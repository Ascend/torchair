#!/usr/bin/env python3
"""
deadlock_check.py - Detect potential deadlock risks in NPU task execution flows.

Analyzes Chrome Tracing format JSON (aligned) files to find concurrent intervals
where AIVEC core requirements exceed the hardware total, causing scheduling deadlocks.

Usage:
    python deadlock_check.py input.json -o output.json --aivec-total 48 --merge-gap 0.0

    input.json 可以是原始 trace JSON，脚本内部会自动完成虚拟 stream 合并（merge）

================================================================================
背景与问题描述
================================================================================

输入 JSON 是 Chrome Tracing 格式，描述一个 AI 模型在 NPU 上多个 stream（tid）的
task 执行序列。每个 stream 内部串行执行，多个 stream 之间通过 EVENT_RECORD/WAIT 和
MEM_WRITE_VALUE/WAIT_VALUE 进行同步。

NPU 硬件资源：
  - 24 个 AIC（AI Core）
  - 48 个 AIVEC（AI Vector Core，默认值，可通过 --aivec-total 修改）

死锁场景：
  若一个 task 的 schemMode=1，代表它必须等到 numBlocks 个对应的核全部空闲后才能
  "一次性"获取所有核并启动。当多个 stream 上的通信算子（schemMode=1）同时处于
  ready 状态，且它们所需的 AIVEC 核数之和超过硬件总数时，NPU 调度器无法同时满足
  所有任务的资源申请，导致死锁（每个任务都在等对方释放核，但谁都无法启动）。

通信算子识别（不区分大小写）：
  - 名称含 "aiv_"
  - 名称含 "dispatch"
  - 名称含 "combine"

AIVEC 占用计算：
  - KERNEL_AIVEC   : numBlocks 个 AIVEC
  - KERNEL_MIX_AIV : numBlocks 个 AIVEC（Task Ration 字段对此类型无效）
  - KERNEL_MIX_AIC : Task Ration=0 → 0 AIVEC；Task Ration=1/2 → numBlocks×2 AIVEC
  - KERNEL_AICORE  : 0 AIVEC

死锁判定条件：两个并发通信算子的 AIVEC 之和 严格大于 aivec_total。

================================================================================
并发区间判定：向量时钟（Vector Clock）算法
================================================================================

由于 task 的 dur/ts 只是可视化用的示意时间，并不代表真实执行耗时，真实的执行约束
完全由控制流事件决定（EVENT_WAIT 必须在对应的 EVENT_RECORD 之后才能继续执行）。
因此不能用 ts 差值来判断并发，必须基于纯逻辑依赖图。

核心定义：
  两个 task T_A（stream_A, pos_i）与 T_B（stream_B, pos_j）并发，当且仅当：
    T_A 不 happens-before T_B，且 T_B 不 happens-before T_A。

向量时钟结构：
  vc[stream_S][pos] 是一个字典，记录"在 stream_S 的第 pos 个 task 执行之前，
  已知 stream_R 上最晚发生到哪个位置"：
    vc[S][pos][R] = k  →  stream_R 上索引 ≤ k 的所有 task 都 happens-before S[pos]

判定规则：
  T_A happens-before T_B  ⟺  vc[stream_B][pos_j][stream_A] >= pos_i
  T_A 与 T_B 并发         ⟺  vc[stream_A][pos_i][stream_B] < pos_j
                              且 vc[stream_B][pos_j][stream_A] < pos_i

传播规则（迭代至不动点）：
  1. 同 stream 顺序继承：S[pos] 的时钟 = max(S[pos], S[pos-1])
  2. WAIT 事件合并：遇到 EVENT_WAIT_H（或 MEM_WAIT_VALUE_H）时，将对应
     EVENT_RECORD_H 所在 task 的向量时钟 merge 进来（取各分量 max），从而
     传递间接依赖（通过中间 stream 传递的 happens-before 关系同样会被捕获）。

================================================================================
典型多 stream 并发场景分析（三流链式结构）
================================================================================

场景描述（用户提出）：
  event1: stream1 发出 RECORD_e1 → stream2 收到（WAIT_e1 解除阻塞）
  event2: stream2 发出 RECORD_e2 → stream3 收到（WAIT_e2 解除阻塞）
  event3: stream3 发出 RECORD_e3 → stream1 收到（WAIT_e3 解除阻塞）

各 stream 在同步事件之间运行的任务：
  stream1: ... RECORD_e1 ... [T1] ... WAIT_e3 ...
  stream2:           WAIT_e1 ... [T2] ... RECORD_e2 ...
  stream3:                       WAIT_e2 ... [T3] ... RECORD_e3 ...

向量时钟分析：

  T1（stream1，在 RECORD_e1 之后、WAIT_e3 之前）：
    vc[T1][stream2] = -1  （stream1 尚未收到 stream2 的任何回信）
    vc[T1][stream3] = -1

  T2（stream2，在 WAIT_e1 之后、RECORD_e2 之前）：
    vc[T2][stream1] = pos(RECORD_e1)  （通过 WAIT_e1 获得）
    vc[T2][stream3] = -1

  T3（stream3，在 WAIT_e2 之后、RECORD_e3 之前）：
    vc[T3][stream1] = pos(RECORD_e1)  （经 stream2 传递：WAIT_e2 merge 了 T2 的时钟）
    vc[T3][stream2] = pos(RECORD_e2)  （通过 WAIT_e2 获得）

并发关系判定：

  T1 与 T2：
    vc[T1][stream2]=-1 < pos(T2) ✓
    vc[T2][stream1]=pos(RECORD_e1) < pos(T1) ✓  （T1 在 RECORD_e1 之后）
    → T1 与 T2 并发 ✓（stream1 发出信号后与 stream2 同时执行）

  T1 与 T3：
    vc[T1][stream3]=-1 < pos(T3) ✓
    vc[T3][stream1]=pos(RECORD_e1) < pos(T1) ✓  （T1 在 RECORD_e1 之后）
    → T1 与 T3 并发 ✓（即使 T3 是经由 stream2 间接解锁，stream1 对此不知情）

  T2 与 T3：
    vc[T3][stream2]=pos(RECORD_e2)
    pos(T2) < pos(RECORD_e2)（T2 在 RECORD_e2 之前）
    → vc[T3][stream2] >= pos(T2)  →  T3 happens-after T2
    → T2 与 T3 不并发 ✓（T3 必须等 stream2 发出 RECORD_e2 才能启动，即 T2 之后）

结论：向量时钟的迭代传播自动处理了任意深度的传递依赖，包括三流链式、
星形拓扑（单协调流）等所有多 stream 同步结构。

================================================================================
虚拟 stream 合并（内置前处理）
================================================================================

NPU 中某些 stream 通过 STREAM_ACTIVE 机制被"激活"到宿主 stream 上运行，但 trace
JSON 中这些 task 仍以独立 tid 记录。若不合并，死锁检测会因 stream 归属错误而产生
漏报或误报。

本脚本在检测前自动调用 _merge_stream_active()，将所有虚拟 stream 合并到宿主 stream，
无需用户手动执行额外步骤。可通过 --merge-gap 控制合并时的时间间隙（默认 0.0）。

================================================================================
工作流（直接使用原始 JSON）
================================================================================

  graph_N.json  →  deadlock_check.py（内部自动 merge）→  检测报告 + 标注 JSON

================================================================================
测试用例
================================================================================

graph_1_filtered.aligned.json（预期：无死锁）
  stream355 作为协调流：先等 stream312 完成第一阶段（WAIT_575160），才发
  RECORD_575192 给 stream311。向量时钟通过传递，使 stream311 的通信算子被判定为
  在 stream312 之后执行，因此两者不并发，不构成死锁。

graph_2_filtered.aligned.json（预期：有死锁）
  stream315 在 pos=7 发 RECORD 给 stream313，在 pos=8 发 RECORD 给 stream310，
  在 pos=22 才等 stream313 回信。两个 stream 同时处于 ready 状态，无依赖约束。
  aiv_all_gather_bfloat16_t（48 AIVEC，stream310）
    + MoeDistributeDispatchV2（16 AIVEC，stream313）= 64 > 48 → 死锁
  aiv_all_gather_bfloat16_t（48 AIVEC，stream310）
    + MoeDistributeCombineV2（16 AIVEC，stream313）= 64 > 48 → 死锁
"""

import os
import json
import argparse
import re
import sys
from collections import defaultdict as _defaultdict


# ---------------------------------------------------------------------------
# 虚拟 stream 合并（内置前处理，检测前自动执行）
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
    task 实际运行在宿主 stream 的硬件队列上。若不合并，这些 task 的 tid 与真实执行
    stream 不符，会导致死锁检测漏报或误报。

    参数：
        events : list，Chrome Tracing JSON 的事件列表（in-place 修改 tid/ts）
        gap    : float，合并时 source stream 开头与 target stream 结尾之间的时间间隙

    返回：
        (events, moved)
        events : 修改后的同一列表
        moved  : [(source_tid, target_tid), ...] 记录所有发生合并的 stream 对
    """
    # 按 tid 建立索引，每组内按 ts 排序
    tid_events = {}
    for ev in events:
        tid_events.setdefault(ev.get("tid"), []).append(ev)
    for tid, lst in tid_events.items():
        lst.sort(key=lambda e: e.get("ts", 0))

    # 收集所有 STREAM_ACTIVE 事件，按 ts 升序处理（保证合并顺序稳定）
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
        # 计算时间偏移：让 source 的起点接在 target 末尾 + gap 处
        target_end = max(e.get("ts", 0) + e.get("dur", 0) for e in target_evs) if target_evs else 0
        source_min = min(e.get("ts", 0) for e in source_evs)
        offset = (target_end + gap) - source_min
        # 修改 source 所有 event 的 ts、tid、Stream Id
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


# ---------------------------------------------------------------------------
# Task classification helpers
# ---------------------------------------------------------------------------

def _is_comm_task(name: str) -> bool:
    """Return True if the task name indicates a communication operator."""
    lower = name.lower()
    return "aiv_" in lower or "dispatch" in lower or "combine" in lower


def _calc_aivec(task: dict) -> int:
    """
    Calculate the number of AIVEC cores occupied by a task.

    Rules:
      KERNEL_AIVEC   : numBlocks
      KERNEL_MIX_AIV : numBlocks  (Task Ration has no effect for this type)
      KERNEL_MIX_AIC : Task Ration == 0 -> 0; else numBlocks * 2
      KERNEL_AICORE  : 0
      others         : 0
    """
    args = task.get("args", {})
    task_type = args.get("Task Type", "")
    num_blocks = args.get("numBlocks", 0)
    task_ration = args.get("Task Ration", -1)

    if task_type == "KERNEL_AIVEC":
        return num_blocks
    if task_type == "KERNEL_MIX_AIV":
        return num_blocks
    if task_type == "KERNEL_MIX_AIC":
        return 0 if task_ration == 0 else num_blocks * 2
    # KERNEL_AICORE and all control-flow types
    return 0


def _get_sync_handle(name: str):
    """
    Extract the numeric handle from a sync task name.
    E.g. 'EVENT_RECORD_20622287575192' -> 20622287575192
         'MEM_WRITE_VALUE_20620228718592' -> 20620228718592
    Returns None if the name does not match.
    """
    prefixes = (
        "EVENT_RECORD_",
        "EVENT_WAIT_",
        "MEM_WRITE_VALUE_",
        "MEM_WAIT_VALUE_",
    )
    for prefix in prefixes:
        if name.startswith(prefix):
            tail = name[len(prefix):]
            if tail.isdigit():
                return int(tail)
    return None


def _is_record_event(name: str) -> bool:
    return name.startswith("EVENT_RECORD_") or name.startswith("MEM_WRITE_VALUE_")


def _is_wait_event(name: str) -> bool:
    return name.startswith("EVENT_WAIT_") or name.startswith("MEM_WAIT_VALUE_")


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _build_stream_tasks(data: list) -> dict:
    """
    Return {tid: [task, ...]} where tasks are ph='X' events sorted by Task Id.
    """
    stream_tasks = _defaultdict(list)
    for item in data:
        if item.get("ph") == "X" and "tid" in item and "args" in item:
            stream_tasks[item["tid"]].append(item)
    for tid in stream_tasks:
        stream_tasks[tid].sort(key=lambda x: x["args"].get("Task Id", 0))
    return dict(stream_tasks)


def _build_sync_pairs(stream_tasks: dict) -> dict:
    """
    Build a mapping from event handle to (record_stream, record_idx, wait_stream, wait_idx).

    For each handle H:
      - Find the task named EVENT_RECORD_H / MEM_WRITE_VALUE_H -> record side
      - Find the task named EVENT_WAIT_H  / MEM_WAIT_VALUE_H  -> wait side

    Returns:
        {handle: (record_stream, record_idx, wait_stream, wait_idx)}
    """
    record_map = {}  # handle -> (stream, idx)
    wait_map = {}    # handle -> (stream, idx)

    for tid, tasks in stream_tasks.items():
        for idx, task in enumerate(tasks):
            name = task["name"]
            handle = _get_sync_handle(name)
            if handle is None:
                continue
            if _is_record_event(name):
                record_map[handle] = (tid, idx)
            elif _is_wait_event(name):
                wait_map[handle] = (tid, idx)

    pairs = {}
    for handle, (rec_tid, rec_idx) in record_map.items():
        if handle in wait_map:
            wait_tid, wait_idx = wait_map[handle]
            pairs[handle] = (rec_tid, rec_idx, wait_tid, wait_idx)

    return pairs


def _compute_vector_clocks(stream_tasks: dict, sync_pairs: dict) -> dict:
    """
    Compute a vector clock for every task position on every stream.

    vc[stream][pos] = {other_stream: latest_synced_index}

    A task T on stream S at position i 'happens-after' task T' on stream R at
    position j if vc[S][i][R] >= j.

    Two tasks are CONCURRENT iff neither happens-after the other.

    The algorithm iterates until no vc value changes (fixed-point), propagating:
      1. Sequential inheritance within a stream (each task inherits from previous).
      2. At EVENT_WAIT / MEM_WAIT_VALUE, merge the clock of the matching RECORD task.
    """
    streams = list(stream_tasks.keys())
    stream_index = {s: i for i, s in enumerate(streams)}
    n_streams = len(streams)

    # Represent clocks as lists of ints for speed: index = stream_index[stream]
    # vc[stream][pos] = list of n_streams ints, default -1
    # vc[stream][pos][k] = latest index on streams[k] that happens-before this task
    vc = {}
    for s in streams:
        n = len(stream_tasks[s])
        si = stream_index[s]
        clocks = []
        for pos in range(n):
            clock = [-1] * n_streams
            clock[si] = pos          # a task always happens-after itself
            clocks.append(clock)
        vc[s] = clocks

    # Build lookup: (wait_stream, wait_idx) -> (record_stream, record_idx)
    wait_to_record = {}
    for handle, (rec_s, rec_i, wait_s, wait_i) in sync_pairs.items():
        wait_to_record[(wait_s, wait_i)] = (rec_s, rec_i)

    # Iterative fixed-point propagation
    changed = True
    iterations = 0
    while changed:
        changed = False
        iterations += 1
        for s in streams:
            tasks = stream_tasks[s]
            n = len(tasks)
            for pos in range(n):
                clock = vc[s][pos]

                # 1. Inherit from previous task on the same stream
                if pos > 0:
                    prev = vc[s][pos - 1]
                    for k in range(n_streams):
                        if prev[k] > clock[k]:
                            clock[k] = prev[k]
                            changed = True

                # 2. At WAIT events, merge from the matching RECORD task's clock
                task_name = tasks[pos]["name"]
                if _is_wait_event(task_name):
                    key = (s, pos)
                    if key in wait_to_record:
                        rec_s, rec_i = wait_to_record[key]
                        rec_clock = vc[rec_s][rec_i]
                        for k in range(n_streams):
                            if rec_clock[k] > clock[k]:
                                clock[k] = rec_clock[k]
                                changed = True

    return vc, stream_index


def _are_concurrent(vc, stream_index, stream_a, pos_a, stream_b, pos_b) -> bool:
    """
    Return True if task (stream_a, pos_a) and (stream_b, pos_b) are concurrent,
    i.e. neither happens-before the other.
    """
    idx_a = stream_index[stream_a]
    idx_b = stream_index[stream_b]

    # Does A happen-after B?  vc[A][pos_a][stream_b] >= pos_b
    a_after_b = vc[stream_a][pos_a][idx_b] >= pos_b
    # Does B happen-after A?  vc[B][pos_b][stream_a] >= pos_a
    b_after_a = vc[stream_b][pos_b][idx_a] >= pos_a

    # Concurrent iff neither is true
    return not a_after_b and not b_after_a


def _find_deadlock_tasks(stream_tasks: dict, vc: dict, stream_index: dict,
                        aivec_total: int) -> list:
    """
    Find all pairs of concurrent communication tasks (schemMode=1) from different
    streams whose combined AIVEC usage strictly exceeds aivec_total.

    Returns a list of (task_a, task_b) dicts representing conflicting pairs.
    """
    streams = list(stream_tasks.keys())

    # Collect candidate tasks per stream: (pos, task, aivec_count)
    candidates = {}
    for s in streams:
        cands = []
        for pos, task in enumerate(stream_tasks[s]):
            args = task.get("args", {})
            if args.get("schemMode") != 1:
                continue
            if not _is_comm_task(task["name"]):
                continue
            aivec = _calc_aivec(task)
            if aivec <= 0:
                continue
            cands.append((pos, task, aivec))
        candidates[s] = cands

    deadlock_pairs = []

    for i, s_a in enumerate(streams):
        for s_b in streams[i + 1:]:
            for pos_a, task_a, aivec_a in candidates[s_a]:
                for pos_b, task_b, aivec_b in candidates[s_b]:
                    if aivec_a + aivec_b <= aivec_total:
                        continue
                    if _are_concurrent(vc, stream_index, s_a, pos_a, s_b, pos_b):
                        deadlock_pairs.append((task_a, task_b))

    return deadlock_pairs


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _mark_deadlock_tasks(data: list, deadlock_tasks: set) -> list:
    """
    Return a new list where tasks in deadlock_tasks have:
      - dur set to 49.5
      - name prefixed with 'dead_lock_'
    Tasks are identified by (tid, Task Id).
    """
    result = []
    for item in data:
        if (
            item.get("ph") == "X"
            and "tid" in item
            and "args" in item
        ):
            key = (item["tid"], item["args"].get("Task Id"))
            if key in deadlock_tasks:
                item = dict(item)          # shallow copy to avoid mutating input
                item["dur"] = 49.5
                if not item["name"].startswith("dead_lock_"):
                    item["name"] = "dead_lock_" + item["name"]
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def deadlock_check(input_path, output_path, aivec_total, merge_gap=0.0):
    # Load input
    print(f"[INFO] Loading {input_path} ...", file=sys.stderr)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Merge virtual streams (prerequisite for correct deadlock detection)
    data, moved = _merge_stream_active(data, gap=merge_gap)
    if moved:
        for src, dst in moved:
            print(f"[INFO] Merged stream: {src} -> {dst}", file=sys.stderr)

    # Build per-stream task lists
    stream_tasks = _build_stream_tasks(data)
    print(
        f"[INFO] Streams found: {sorted(stream_tasks.keys())}",
        file=sys.stderr,
    )

    # Build sync pairs
    sync_pairs = _build_sync_pairs(stream_tasks)
    print(f"[INFO] Sync pairs found: {len(sync_pairs)}", file=sys.stderr)

    # Compute vector clocks
    print("[INFO] Computing vector clocks ...", file=sys.stderr)
    vc, stream_index = _compute_vector_clocks(stream_tasks, sync_pairs)

    # Find deadlock pairs
    print(
        f"[INFO] Checking for AIVEC deadlocks (total={aivec_total}) ...",
        file=sys.stderr,
    )
    deadlock_pairs = _find_deadlock_tasks(
        stream_tasks, vc, stream_index, aivec_total
    )

    if not deadlock_pairs:
        print("[RESULT] No deadlock risks detected.")
        return

    # Collect unique deadlock task keys and details
    deadlock_task_keys = set()   # (tid, task_id)
    deadlock_task_details = []   # list of (tid, task_id, name)

    print(f"\n[RESULT] Deadlock risk detected! {len(deadlock_pairs)} conflicting pair(s):\n")
    for task_a, task_b in deadlock_pairs:
        aivec_a = _calc_aivec(task_a)
        aivec_b = _calc_aivec(task_b)
        tid_a = task_a["tid"]
        tid_b = task_b["tid"]
        task_id_a = task_a["args"].get("Task Id")
        task_id_b = task_b["args"].get("Task Id")
        print(
            f"  CONFLICT: [{tid_a} TaskId={task_id_a}] {task_a['name']} "
            f"({aivec_a} AIVEC)"
            f"  <->  [{tid_b} TaskId={task_id_b}] {task_b['name']} "
            f"({aivec_b} AIVEC)"
            f"  sum={aivec_a + aivec_b} > {aivec_total}"
        )
        for task in (task_a, task_b):
            key = (task["tid"], task["args"].get("Task Id"))
            if key not in deadlock_task_keys:
                deadlock_task_keys.add(key)
                deadlock_task_details.append((
                    task["tid"],
                    task["args"].get("Task Id"),
                    task["name"],
                ))

    print(f"\n[RESULT] Deadlock task list ({len(deadlock_task_details)} tasks):")
    for tid, task_id, name in deadlock_task_details:
        print(f"  - [{tid} TaskId={task_id}] {name}")

    # Write output JSON
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_deadlock_check{ext}"
    
    out_data = _mark_deadlock_tasks(data, deadlock_task_keys)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Output written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect AIVEC deadlock risks in NPU Chrome Tracing JSON files."
    )
    parser.add_argument("input", help="Input aligned Chrome Tracing JSON file")
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file (required when deadlocks are found)",
    )
    parser.add_argument(
        "--aivec-total",
        type=int,
        default=48,
        help="Total number of AIVEC cores on the NPU (default: 48)",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.0,
        help="Time gap inserted when merging virtual streams (default: 0.0)",
    )
    args = parser.parse_args()

    deadlock_check(args.input, output_path=args.output, aivec_total=args.aivec_total, merge_gap=args.merge_gap)
