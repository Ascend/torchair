"""
align_trace.py - 对 NPU Chrome Tracing JSON 进行时间对齐，生成可在 Chrome://tracing
可视化的 aligned JSON 文件。

================================================================================
背景与目的
================================================================================

原始 NPU trace JSON（graph_N.json）中的 ts/dur 是真实硬件采集时间，受硬件噪声、
负载波动影响，无法直观反映 stream 间的依赖关系（EVENT_RECORD → EVENT_WAIT 的因果
链在时间轴上可能杂乱无章）。

本脚本对 ts 进行重新排布：
  1. 每个 stream 内部的 task 间隔统一置为 0.2（示意间隔，非真实时间）
  2. 跨 stream 的 EVENT_WAIT 的 dur 被拉长，使其结束时间 = 对应 EVENT_RECORD 的
     结束时间，从而在 Chrome Tracing 视图中，WAIT block 的右端与 RECORD block 的
     右端对齐，清晰呈现跨 stream 依赖

经过对齐后，Chrome Tracing 的时间轴即变成"逻辑时序轴"：
  - 左右相邻 = 顺序执行
  - 同列重叠 = 并发执行（但由于 stream 内串行，实际上同 stream 不重叠）

================================================================================
主要处理步骤（_align_trace 函数）
================================================================================

Step 1  修正 EVENT_RESET 的 dur 为 0.2（标准化控制事件长度）

Step 2  重排每个 stream 内的 ts：
        - 第一个 task 从 ts=0 开始
        - 后续每个 task 的 ts = 上一个 task 的 ts + dur + 0.2（最小间隙）
        - dur 保持原值不变（aiv/compute 类 9.5，control 类 4.5 等）

Step 3  扫描所有 EVENT_RECORD/MEM_WRITE_VALUE（record 端）和
        EVENT_WAIT/MEM_WAIT_VALUE（wait 端），按 handle 数字分组成配对。
        区分：
          - 同 tid 配对（如 stream 内部的 MEM_WRITE→MEM_WAIT）：不需要跨 stream 对齐
          - 跨 tid 配对（cross_tid_pairs）：需要对齐

Step 4  迭代对齐（最多 200 轮）：
        对每个跨 tid 配对（record_idx, wait_idx）：
          若 wait["ts"] < record["ts"] + record["dur"]（WAIT 开始早于 RECORD 结束），
          则将 wait["dur"] 拉长使其结束时间 = record 结束时间，并将 wait 之后同 tid
          的所有 task 向右平移相同的 delta，保持内部间隔不变。
        重复直到没有变化（收敛）。

Step 5  生成 flow events（Chrome Tracing 的箭头连线）：
        每对 RECORD→WAIT 生成 4 个辅助事件：
          ph="s"  flow start（起始于 record 的右端）
          ph="f"  flow end  （终止于 wait 的左端）
          ph="i"  瞬时标记 START/END（在各自 tid 上显示小图标）

Step 6  清洗算子名称：
        去掉 name 中夹杂的 hash 片段（小写字母+数字且数字超过 5 个的段）。
        控制类算子（EVENT_RECORD_ 等）不做清洗，保留完整 handle 数字。

================================================================================
_validate_alignment 函数
================================================================================

对齐完成后校验：
  - 所有 ts/dur 必须是有限浮点数
  - 每个 stream 内 task 间隔必须恰好为 0.2（允许 1e-6 误差）
  - 跨 tid 配对的 wait 结束时间必须等于 record 结束时间（skipped_pairs 除外）

================================================================================
虚拟 stream 合并（内置前处理）
================================================================================

本脚本在 align 之前自动调用 _merge_stream_active()，将所有虚拟 stream（通过
STREAM_ACTIVE 激活的 stream）合并到宿主 stream，可通过 --merge-gap 控制合并时的时间间隙（默认 0.0）。

================================================================================
使用示例
================================================================================

# 最简用法：直接传入原始 JSON，输出自动命名为 *.aligned.json
python align_trace.py graph_2.json

# 指定输出路径
python align_trace.py graph_2.json -o graph_2.aligned.json

# 控制虚拟 stream 合并间隙
python align_trace.py graph_2.json --merge-gap 5.0

# 在代码中调用（需手动先做 merge）
from align_trace import _align_trace, _validate_alignment, _merge_stream_active
import json

events = json.load(open("graph_2.json"))
events, moved = _merge_stream_active(events, gap=0.0)
aligned, same_tid_pairs, skipped_pairs = _align_trace(events)
_validate_alignment(aligned, same_tid_pairs, skipped_pairs)
json.dump(aligned, open("out.aligned.json", "w"), indent=2)

================================================================================
定位与工作流
================================================================================

本脚本是【可选的人工可视化辅助工具】，不参与死锁检测的计算逻辑。
其输出（*.aligned.json）可直接送入 deadlock_check.py，也可直接用原始 JSON，
两者检测结果一致（deadlock_check.py 只依赖 task 名称和 args，不依赖 ts/dur）。

推荐工作流：

  【死锁检测（直接使用原始 JSON）】
  graph_N.json  →  deadlock_check.py（内部自动 merge）→  检测报告

  【人工可视化（直接使用原始 JSON）】
  graph_N.json  →  filter_operators.py（内部自动 merge）
               →  align_trace.py（内部自动 merge）
               →  Chrome Tracing
"""

import json
import re
from pathlib import Path as _Path


# ---------------------------------------------------------------------------
# 虚拟 stream 合并（内置前处理，align 前自动执行）
# ---------------------------------------------------------------------------

def _stream_id_to_tid(stream_id):
    """将数字 stream id 转换为 tid 字符串，如 315 -> 'stream315'"""
    return f"stream{stream_id}"


def _tid_to_stream_id(tid):
    """将 tid 字符串转换为数字 stream id，如 'stream315' -> 315；无法解析返回 None"""
    if tid.startswith("stream"):
        try:
            return int(tid[len("stream"):])
        except ValueError:
            return None
    return None


def _merge_stream_active(events, gap=0.0):
    """
    合并被 STREAM_ACTIVE 标记的虚拟 stream 到宿主 stream（in-place 修改）。

    NPU 调度模型中，某个 stream 可通过 STREAM_ACTIVE task 激活另一个 stream，使其
    task 实际运行在宿主 stream 的硬件队列上。若不合并，这些虚拟 stream 的 task 在
    Chrome Tracing 中会显示为独立行，影响可视化效果。

    参数：
        events : list，Chrome Tracing JSON 的事件列表（in-place 修改 tid/ts）
        gap    : float，合并时 source stream 开头与 target stream 结尾之间的时间间隙

    返回：
        (events, moved)
        events : 修改后的同一列表
        moved  : [(source_tid, target_tid), ...] 记录所有发生合并的 stream 对
    """
    tid_events = {}
    for ev in events:
        tid_events.setdefault(ev.get("tid"), []).append(ev)
    for tid, lst in tid_events.items():
        lst.sort(key=lambda e: e.get("ts", 0))

    stream_active = [
        ev for ev in events
        if (
            (ev.get("args") or {}).get("Task Type") == "STREAM_ACTIVE"
            or ev.get("name") == "STREAM_ACTIVE"
        )
        and "Active Stream Id" in (ev.get("args") or {})
    ]
    stream_active.sort(key=lambda e: e.get("ts", 0))

    moved = []
    for ev in stream_active:
        args = ev.get("args") or {}
        active_id = args.get("Active Stream Id")
        if active_id is None:
            continue
        source_tid = _stream_id_to_tid(active_id)
        target_tid = ev.get("tid")
        if not target_tid or source_tid == target_tid:
            continue
        if source_tid not in tid_events:
            continue
        target_evs = tid_events.get(target_tid, [])
        source_evs = tid_events.get(source_tid, [])
        if not source_evs:
            continue
        target_end = max(e.get("ts", 0) + e.get("dur", 0) for e in target_evs) if target_evs else 0
        source_min = min(e.get("ts", 0) for e in source_evs)
        offset = (target_end + gap) - source_min
        target_stream_id = _tid_to_stream_id(target_tid)
        for e in source_evs:
            e["ts"] = e.get("ts", 0) + offset
            e["tid"] = target_tid
            if target_stream_id is not None:
                args_e = e.get("args")
                if isinstance(args_e, dict) and "Stream Id" in args_e:
                    args_e["Stream Id"] = target_stream_id
        tid_events[target_tid] = target_evs + source_evs
        tid_events.pop(source_tid, None)
        moved.append((source_tid, target_tid))

    return events, moved


# 控制类 task 的名称前缀，用于跳过名称清洗
CONTROL_PREFIXES = (
    "EVENT_RECORD_",
    "EVENT_WAIT_",
    "MEM_WRITE_VALUE_",
    "MEM_WAIT_VALUE_",
    "EVENT_RESET_",
)


def _parse_control(name: str):
    """
    解析控制事件名称，返回 (类型, handle字符串)。
    例如 "EVENT_RECORD_20622287575192" -> ("EVENT_RECORD", "20622287575192")
    非控制事件返回 (None, None)。
    """
    if name.startswith("EVENT_RECORD_"):
        return "EVENT_RECORD", name.split("_")[-1]
    if name.startswith("EVENT_WAIT_"):
        return "EVENT_WAIT", name.split("_")[-1]
    if name.startswith("MEM_WRITE_VALUE_"):
        return "MEM_WRITE_VALUE", name.split("_")[-1]
    if name.startswith("MEM_WAIT_VALUE_"):
        return "MEM_WAIT_VALUE", name.split("_")[-1]
    return None, None


def _clean_name(name: str) -> str:
    """
    去掉算子名称中的 hash 片段。
    判断标准：纯小写字母+数字组成，且其中数字个数超过 5 个的段视为 hash。
    例如 "MoeDistributeDispatchV2_e3e2549baf4795bf56c350e970d2218a_1060"
    -> "MoeDistributeDispatchV2_1060"
    """
    parts = name.split("_")
    new_parts = []
    for part in parts:
        if re.fullmatch(r"[a-z0-9]+", part) and sum(c.isdigit() for c in part) > 5:
            continue
        new_parts.append(part)
    return "_".join(new_parts) if new_parts else name


def _align_trace(events):
    """
    对 events 列表进行时间对齐，返回 (aligned_events, same_tid_pairs, skipped_pairs)。

    aligned_events  : 原列表（in-place 修改）+ 新增的 flow 辅助事件
    same_tid_pairs  : [(record_name, wait_name), ...] 同 tid 的配对，不需要对齐
    skipped_pairs   : [(record_name, wait_name), ...] wait 本已晚于 record 的配对，
                      对齐前已满足约束，不需要拉伸
    """
    min_gap = 0.2  # stream 内相邻 task 的最小时间间隔（可视化用）
    eps = 1e-9     # 浮点比较容差

    # Step 1: EVENT_RESET 的 dur 统一置为 0.2
    for ev in events:
        name = ev.get("name", "")
        if name.startswith("EVENT_RESET_"):
            ev["dur"] = 0.2

    # Step 2: 按 tid 分组并按原始 ts 排序，建立 (idx -> 在该 tid 中的位置) 映射
    tid_indices = {}
    for idx, ev in enumerate(events):
        tid = ev.get("tid")
        tid_indices.setdefault(tid, []).append(idx)
    for tid, indices in tid_indices.items():
        indices.sort(key=lambda i: (events[i].get("ts", 0), i))

    # tid_pos[tid][global_idx] = 该 task 在 tid 内的顺序位置（0-based）
    tid_pos = {
        tid: {idx: pos for pos, idx in enumerate(indices)}
        for tid, indices in tid_indices.items()
    }

    # Step 2 续：重写每个 stream 内的 ts，起点为 0，相邻间距 min_gap
    for tid, indices in tid_indices.items():
        prev_end = 0.0
        for pos, idx in enumerate(indices):
            ev = events[idx]
            gap = 0.0 if pos == 0 else min_gap
            ev["ts"] = prev_end + gap
            prev_end = ev["ts"] + ev["dur"]

    # Step 3: 扫描所有控制事件，按 handle 分组为 (record, wait) 配对
    pair_map = {}
    for idx, ev in enumerate(events):
        kind, cid = _parse_control(ev.get("name", ""))
        if not kind:
            continue
        slot = pair_map.setdefault(cid, {})
        if kind in ("EVENT_RECORD", "MEM_WRITE_VALUE"):
            slot["record"] = idx
        elif kind in ("EVENT_WAIT", "MEM_WAIT_VALUE"):
            slot["wait"] = idx

    # 区分同 tid 配对（无需跨 stream 对齐）与跨 tid 配对（需要对齐）
    same_tid_pairs = []
    cross_tid_pairs = []
    for cid, slot in pair_map.items():
        if "record" in slot and "wait" in slot:
            r_idx = slot["record"]
            w_idx = slot["wait"]
            if events[r_idx]["tid"] == events[w_idx]["tid"]:
                same_tid_pairs.append((events[r_idx]["name"], events[w_idx]["name"]))
            else:
                cross_tid_pairs.append((cid, r_idx, w_idx))

    # Step 4: 迭代对齐跨 tid 配对
    # 目标：对每对 (record, wait)，令 wait 的结束时间 = record 的结束时间。
    # 若 wait["ts"] < record_end，则拉伸 wait["dur"]，并将 wait 之后同 tid
    # 的所有 task 向右平移 delta，保持内部间距不变。
    # 由于一个 tid 的平移可能影响其他配对，需要多轮迭代直到收敛。
    max_passes = 200
    for _ in range(max_passes):
        changed = False
        for cid, r_idx, w_idx in cross_tid_pairs:
            record = events[r_idx]
            wait = events[w_idx]
            record_end = record["ts"] + record["dur"]
            if wait["ts"] >= record_end - eps:
                continue  # 已满足：wait 开始时间 >= record 结束时间
            needed_dur = record_end - wait["ts"]
            delta = needed_dur - wait["dur"]
            if abs(delta) > eps:
                wait["dur"] = needed_dur
                # 将 wait 之后同 tid 的 task 全部右移 delta
                tid = wait["tid"]
                start_pos = tid_pos[tid][w_idx]
                indices = tid_indices[tid]
                for idx in indices[start_pos + 1:]:
                    events[idx]["ts"] += delta
                changed = True
        if not changed:
            break
    else:
        raise SystemExit("Alignment did not converge within pass limit.")

    # Step 4 后处理：记录哪些跨 tid 配对已经天然满足约束（wait 不需要拉伸）
    skipped_pairs = []
    for cid, r_idx, w_idx in cross_tid_pairs:
        record = events[r_idx]
        wait = events[w_idx]
        if wait["ts"] >= record["ts"] + record["dur"] - eps:
            skipped_pairs.append((record["name"], wait["name"]))

    # Step 5: 为每对 (record, wait) 生成 Chrome Tracing flow 连线事件
    # ph="s": flow start，绘制箭头起点（位于 record 右端）
    # ph="f": flow end，  绘制箭头终点（位于 wait 左端）
    # ph="i": instant marker，在对应 tid 上显示小标记
    flow_events = []
    for cid, slot in pair_map.items():
        if "record" in slot and "wait" in slot:
            r = events[slot["record"]]
            w = events[slot["wait"]]
            try:
                flow_id = int(cid)
            except ValueError:
                flow_id = abs(hash(cid)) % 1000000000000
            flow_events.append(
                {
                    "name": f"PAIR_{cid}",
                    "cat": "event_record->wait",
                    "ph": "s",
                    "pid": r.get("pid"),
                    "tid": r.get("tid"),
                    "ts": r["ts"] + r["dur"],
                    "id": flow_id,
                    "bp": "e",
                }
            )
            flow_events.append(
                {
                    "name": f"PAIR_{cid}",
                    "cat": "event_record->wait",
                    "ph": "f",
                    "pid": w.get("pid"),
                    "tid": w.get("tid"),
                    "ts": w["ts"],
                    "id": flow_id,
                    "bp": "e",
                }
            )
            flow_events.append(
                {
                    "name": f"PAIR_{cid}_START",
                    "cat": "event_record->wait",
                    "ph": "i",
                    "s": "t",
                    "pid": r.get("pid"),
                    "tid": r.get("tid"),
                    "ts": r["ts"] + r["dur"],
                }
            )
            flow_events.append(
                {
                    "name": f"PAIR_{cid}_END",
                    "cat": "event_record->wait",
                    "ph": "i",
                    "s": "t",
                    "pid": w.get("pid"),
                    "tid": w.get("tid"),
                    "ts": w["ts"],
                }
            )

    # Step 6: 清洗算子名称，去掉 hash 片段（控制类 task 跳过，保留完整 handle）
    for ev in events:
        name = ev.get("name")
        if isinstance(name, str):
            if name.startswith(CONTROL_PREFIXES):
                continue
            ev["name"] = _clean_name(name)

    events.extend(flow_events)

    return events, same_tid_pairs, skipped_pairs


def _validate_alignment(events, same_tid_pairs=None, skipped_pairs=None):
    import math
    if same_tid_pairs is None:
        same_tid_pairs = []
    if skipped_pairs is None:
        skipped_pairs = []
    skipped_set = {tuple(p) for p in skipped_pairs}

    non_finite = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        if isinstance(ts, float) and not math.isfinite(ts):
            non_finite.append(("ts", ev.get("name")))
        if isinstance(dur, float) and not math.isfinite(dur):
            non_finite.append(("dur", ev.get("name")))
    if non_finite:
        preview = "\n".join(f"{field} {name}" for field, name in non_finite[:10])
        raise SystemExit(
            f"Alignment check failed: non-finite values detected ({len(non_finite)}).\n{preview}"
        )

    pairs = {}
    for idx, ev in enumerate(events):
        if ev.get("ph") != "X":
            continue
        kind, cid = _parse_control(ev.get("name", ""))
        if not kind:
            continue
        slot = pairs.setdefault(cid, {})
        if kind in ("EVENT_RECORD", "MEM_WRITE_VALUE"):
            slot["record"] = idx
        elif kind in ("EVENT_WAIT", "MEM_WAIT_VALUE"):
            slot["wait"] = idx

    mismatches = []
    for cid, slot in pairs.items():
        if "record" in slot and "wait" in slot:
            r = events[slot["record"]]
            w = events[slot["wait"]]
            if r.get("tid") == w.get("tid"):
                continue
            if (r.get("name"), w.get("name")) in skipped_set:
                continue
            r_end = r["ts"] + r["dur"]
            w_end = w["ts"] + w["dur"]
            if w["ts"] < r_end - 1e-6 and abs(r_end - w_end) > 1e-6:
                mismatches.append((cid, r_end, w_end))

    min_gap = 0.2
    overlaps = []
    tid_indices = {}
    for idx, ev in enumerate(events):
        if ev.get("ph") != "X":
            continue
        tid = ev.get("tid")
        tid_indices.setdefault(tid, []).append(idx)
    for tid, indices in tid_indices.items():
        indices.sort(key=lambda i: (events[i].get("ts", 0), i))
        prev_end = None
        for idx in indices:
            ev = events[idx]
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            if prev_end is None:
                if abs(ts - 0.0) > 1e-6:
                    overlaps.append((tid, None, ev.get("name")))
                prev_end = ts + dur
                continue
            expected_ts = prev_end + 0.2
            if abs(ts - expected_ts) > 1e-6:
                overlaps.append((tid, None, ev.get("name")))
                if len(overlaps) >= 10:
                    break
            prev_end = ts + dur
        if len(overlaps) >= 10:
            break

    if overlaps:
        preview = "\n".join(f"{tid}: {right}" for tid, _, right in overlaps)
        raise SystemExit(
            f"Alignment check failed: gap mismatch detected ({len(overlaps)}+).\n{preview}"
        )

    return mismatches


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Align trace control tasks and clean task names."
    )
    parser.add_argument("input", help="input trace json path")
    parser.add_argument(
        "-o",
        "--output",
        help="output json path (default: input with .aligned suffix)",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.0,
        help="time gap inserted when merging virtual streams (default: 0.0)",
    )
    args = parser.parse_args()

    in_path = _Path(args.input)
    out_path = _Path(args.output) if args.output else in_path.with_suffix(
        in_path.suffix.replace(".json", "") + ".aligned.json"
    )

    with in_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    # Merge virtual streams before alignment
    events, moved = _merge_stream_active(events, gap=args.merge_gap)
    if moved:
        print("Merged streams:")
        for src, dst in moved:
            print(f"  {src} -> {dst}")

    aligned, same_tid_pairs, skipped_pairs = _align_trace(events)
    mismatches = _validate_alignment(aligned, same_tid_pairs, skipped_pairs)
    if mismatches:
        preview = "\n".join(
            f"{cid} record_end={r_end} wait_end={w_end}"
            for cid, r_end, w_end in mismatches[:10]
        )
        raise SystemExit(
            f"Alignment check failed: {len(mismatches)} mismatches.\n{preview}"
        )
    if same_tid_pairs:
        print("Same-tid pairs (no alignment):")
        for r_name, w_name in same_tid_pairs:
            print(f"{r_name} | {w_name}")
    if skipped_pairs:
        print("Cross-tid pairs skipped (wait starts after record end):")
        for r_name, w_name in skipped_pairs:
            print(f"{r_name} | {w_name}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()
