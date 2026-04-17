#!/usr/bin/env python3
"""
filter_operators.py - 过滤 NPU aclGraph JSON，仅保留通信类算子和控制类算子。

================================================================================
背景与目的
================================================================================

原始 NPU trace（graph_N.json）包含数万条 task，绝大多数是普通计算算子（矩阵乘、
激活函数等）。死锁检测只关心两类 task：

  1. 控制类（Control）：负责跨 stream 同步，决定执行顺序
       - Record 端（非阻塞写）：EVENT_RECORD_*、MEM_WRITE_VALUE_*
       - Wait   端（阻塞读）  ：EVENT_WAIT_*、MEM_WAIT_VALUE_*

  2. 通信类（Communication）：占用 AIVEC/AIC 硬件资源，schemMode=1 时可能引发死锁
       - name 以 aiv_ 开头         （如 aiv_all_to_all_v_bfloat16_t）
       - name 以 hccl_aiv 开头     （如 hccl_aiv_sync）
       - name 含 dispatch 或 combine（如 MoeDistributeDispatchV2、MoeDistributeCombineV2）

其余 task（KERNEL_AICORE 计算、MEMCPY_ASYNC、NOP、EVENT_RESET 等）全部丢弃，
大幅缩减文件体积，便于后续 align_trace.py 和 deadlock_check.py 处理。

================================================================================
额外校验
================================================================================

通信类算子若 schemMode ≠ 1，会打印告警。
schemMode=1 是触发死锁的必要条件（需凑满 numBlocks 个核才能启动），
schemMode≠1 的通信算子无需纳入死锁检测，通常不应出现，值得关注。

================================================================================
虚拟 stream 合并（内置前处理）
================================================================================

本脚本在过滤之前自动调用 _merge_stream_active()，将所有虚拟 stream（通过
STREAM_ACTIVE 激活的 stream）合并到宿主 stream，可通过 --merge-gap 控制合并时的时间间隙（默认 0.0）。

================================================================================
使用示例
================================================================================

# 最简用法：直接传入原始 JSON，输出自动命名为 graph_2_filtered.json
python filter_operators.py graph_2.json

# 指定输出路径
python filter_operators.py graph_2.json -o graph_2_filtered.json

# 控制虚拟 stream 合并间隙
python filter_operators.py graph_2.json --merge-gap 5.0

# 不传参数时默认处理同目录下的 graph_1.json
python filter_operators.py

================================================================================
定位与工作流
================================================================================

本脚本是【可选的人工可视化辅助工具】，不影响死锁检测结果。
deadlock_check.py 可直接处理原始 JSON，无需先经过本脚本过滤。

推荐工作流：

  【死锁检测（直接使用原始 JSON）】
  graph_N.json  →  deadlock_check.py（内部自动 merge）→  检测报告

  【人工可视化（直接使用原始 JSON）】
  graph_N.json  →  filter_operators.py（内部自动 merge）
               →  align_trace.py（内部自动 merge）
               →  Chrome Tracing

filter_operators.py 的作用：去掉大量计算算子（矩阵乘、激活等），将文件体积缩减到
可在 Chrome Tracing 中流畅浏览的大小，同时保留所有死锁相关的通信算子和控制算子，
便于人工对照 deadlock_check.py 的输出结果进行验证。
"""

import argparse
import json
import os
import sys
from collections import defaultdict as _defaultdict


# ---------------------------------------------------------------------------
# 虚拟 stream 合并（内置前处理，过滤前自动执行）
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
    task 实际运行在宿主 stream 的硬件队列上。合并后可视化结果中不再出现碎片 stream。

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


def _classify_operator(name: str, extend_info):
    """
    根据算子 name 判断类型。
    返回 "control_record" / "control_wait" / "communication" / None(不保留)
    """
    lower_name = name.lower()

    # 控制类 - Record
    if lower_name.startswith("event_record") or lower_name.startswith("mem_write"):
        return "control_record"

    # 控制类 - Wait
    if lower_name.startswith("event_wait") or lower_name.startswith("mem_wait"):
        return "control_wait"

    # 通信类
    if lower_name.startswith("aiv_"):
        return "communication"
    if lower_name.startswith("hccl_aiv"):
        return "communication"
    if "dispatch" in lower_name or "combine" in lower_name:
        return "communication"
    if extend_info:
        extend_info_json = json.loads(extend_info)
        if extend_info_json.get("taskType") == "communication":
            return "communication"

    # 其他 -> 过滤掉
    return None


def _filter_operators(input_path: str, output_path: str, merge_gap: float = 0.0):
    """读取原始 JSON，合并虚拟 stream，过滤后输出到新文件。"""
    with open(input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # 合并虚拟 stream（过滤前执行）
    tasks, moved = _merge_stream_active(tasks, gap=merge_gap)
    if moved:
        print("合并虚拟 stream:")
        for src, dst in moved:
            print(f"  {src} -> {dst}")

    print(f"原始算子总数: {len(tasks)}")

    # 统计信息
    stats = _defaultdict(int)
    stream_stats = _defaultdict(lambda: _defaultdict(int))
    filtered = []

    for task in tasks:
        name = task.get("name", "")
        stream = task.get("tid", "unknown")
        extend_info = task.get("args", "").get("ExtendInfo", "")
        op_type = _classify_operator(name, extend_info)

        if op_type is not None:
            filtered.append(task)
            stats[op_type] += 1
            stream_stats[stream][op_type] += 1

            # 通信算子 schemMode 不为 1 时告警
            if op_type == "communication":
                schem_mode = task.get("args", {}).get("schemMode")
                if schem_mode != 1:
                    task_id = task.get("args", {}).get("Task Id", "?")
                    print(f"  [告警] 通信算子 schemMode≠1: {stream} Task={task_id} "
                          f"name=\"{name}\" schemMode={schem_mode}")
        else:
            stats["filtered_out"] += 1

    # 输出统计
    print(f"\n=== 过滤结果 ===")
    print(f"保留算子总数: {len(filtered)}")
    print(f"过滤掉算子数: {stats['filtered_out']}")
    print(f"\n按类型统计:")
    print(f"  控制类 - Record: {stats['control_record']}")
    print(f"  控制类 - Wait:   {stats['control_wait']}")
    print(f"  通信类:          {stats['communication']}")

    print(f"\n按 Stream 统计:")
    for stream in sorted(stream_stats.keys()):
        s = stream_stats[stream]
        parts = []
        if s["control_record"] > 0:
            parts.append(f"Record={s['control_record']}")
        if s["control_wait"] > 0:
            parts.append(f"Wait={s['control_wait']}")
        if s["communication"] > 0:
            parts.append(f"通信={s['communication']}")
        print(f"  {stream}: {', '.join(parts)}")

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"\n已保存过滤结果到: {output_path}")
    return output_path


def filter_comm_ops(input_path, output_path, merge_gap=0.0):
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    # 输出文件名: 在原文件名基础上加 _filtered 后缀
    if output_path:
        filtered_output_path = output_path
    else:
        base, ext = os.path.splitext(input_path)
        filtered_output_path = f"{base}_filtered{ext}"

    return _filter_operators(input_path, filtered_output_path, merge_gap=merge_gap)


if __name__ == "__main__":
    # 默认输入路径（不传参数时使用）
    default_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_1.json")

    parser = argparse.ArgumentParser(
        description="过滤 NPU aclGraph JSON，仅保留通信类算子和控制类算子。"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=default_input,
        help="输入 JSON 文件路径（默认: 同目录下的 graph_1.json）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出 JSON 文件路径（默认: 在输入文件名基础上加 _filtered 后缀）",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.0,
        help="合并虚拟 stream 时的时间间隙（默认: 0.0）",
    )
    args = parser.parse_args()
    filter_comm_ops(args.input, args.output, args.merge_gap)
