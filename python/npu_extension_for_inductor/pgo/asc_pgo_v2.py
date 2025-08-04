# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import argparse
import json
import csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

op_data = {}
PROFILING_DIR_PREFIX = 'PROF'
SUMMARY_FILE_PREFIX = 'op_summary'
PROFILER_OUTPUT_DIR = 'mindstudio_profiler_output'


class MsprofDeal:
    @staticmethod
    def execute_test(exec_mode=None):
        """执行脚本文件"""
        full_cmd = []
        try:
            if exec_mode:
                cmd_parts = exec_mode.split()
                if cmd_parts[0] == "msprof":
                    # 格式: "msprof python test.py"
                    run_profiling_mode = cmd_parts[0]
                    args = cmd_parts[1:]
                    full_cmd = [run_profiling_mode] + args
                else:
                    # 格式: "python test.py"
                    full_cmd = cmd_parts

            subprocess.run(full_cmd, check=True)
        except Exception as e:
            logging.error("执行失败 %s: %s", test_path, e, exc_info=True)

    @staticmethod
    def remove_kernel(kernel_so_path: Path):
        """删除缓存的kernel.so文件"""
        if kernel_so_path.exists():
            try:
                os.remove(kernel_so_path)
                logging.info("删除kernel so文件 %s", kernel_so_path)
            except Exception as e:
                logging.error("kernel so文件不存在 %s", kernel_so_path)

    @staticmethod
    def deal_search_file(search_path, op_name, cases_range):
        """获取算子调优配置数据"""
        global op_data

        if op_name in op_data:
            logging.info(f"算子 {op_name} 的数据已存在于内存中，直接使用")
            return op_data[op_name]

        try:
            if not os.path.exists(search_path):
                logging.error(f"未找到 {search_path} 文件")
                raise FileNotFoundError(f"文件 {search_path} 不存在")

            op_dict = ProfilerDeal.parse_file_content(search_path)
            top_n = ProfilerDeal.process_cases(op_dict, cases_range)
            last_case = ProfilerDeal.get_last_case_info(search_path)

            if last_case[0] is not None and last_case[1] is not None:
                top_n = ProfilerDeal.update_with_last_case(top_n, last_case)

            op_data[op_name] = top_n
            logging.info(f"成功读取并存储算子 {op_name} 的数据")
            return op_data[op_name]
        except Exception as e:
            logging.error(f"读取和处理文件 {search_path} 时出错：{e}")
            raise

    @staticmethod
    def parse_file_content(file_path):
        """解析文件内容并返回操作字典"""
        op_dict = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split('#')
                if len(parts) < 2:
                    continue
                case_str = line.strip()
                case_duration_time = parts[1].strip()
                if case_str and case_duration_time:
                    op_dict[case_str] = case_duration_time
        if not op_dict:
            raise ValueError("文件中未提取到有效的数据。")
        return op_dict

    @staticmethod
    def process_cases(op_dict, cases_range):
        """处理案例并返回排序后的top_n"""
        cases = []
        for case_str, duration_str in op_dict.items():
            try:
                cases.append((case_str, float(duration_str)))
            except ValueError:
                continue
        if not cases:
            raise ValueError("文件中未提取到有效的数值数据")
        cases.sort(key=lambda x: x[1])
        return cases[:cases_range]

    @staticmethod
    def get_last_case_info(file_path):
        """获取文件最后一行案例信息"""
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if not lines:
                return (None, None)
            last_line = lines[-1]
            parts = last_line.split('#')
            if len(parts) >= 2:
                try:
                    return (last_line.strip(), float(parts[1].strip()))
                except ValueError:
                    return (last_line.strip(), None)
            return (None, None)

    @staticmethod
    def update_with_last_case(top_n, last_case):
        """使用最后一个案例更新top_n列表"""
        last_case_str, last_duration = last_case
        for i, (case_str, _) in enumerate(top_n):
            if case_str == last_case_str:
                del top_n[i]
                break
        top_n.append((last_case_str, last_duration))
        return top_n

    @staticmethod
    def write_to_config(selected_case, config_path):
        """将最优解写入配置文件"""
        try:
            if not selected_case:
                raise ValueError("selected_case 不能为空")
            with open(config_path, 'w', encoding='utf-8') as file:
                file.write('1\n')
                file.write(f"{selected_case}\n")

            logging.info(f"成功写入配置文件 {config_path}。")

        except Exception as e:
            logging.error(f"写入配置文件 {config_path} 时出错。")
            raise

    @staticmethod
    def get_all_filenames_with_prefix(dir_path: str, prefix: str) -> list:
        """获取指定目录下所有以特定前缀开头的文件名"""
        return [filename for filename in os.listdir(dir_path) if filename.startswith(prefix)]

    @staticmethod
    def get_config_path(lib_dir: Path, op_name: str) -> Path:
        """获取算子对应的config文件路径"""
        op_dir = lib_dir / op_name
        npu_kernel_dir = next(op_dir.iterdir())
        return npu_kernel_dir / "config.txt"

    @staticmethod
    def _should_skip_file(file_path, idx):
        """判断是否应该跳过当前文件"""
        if file_path.upper() == "NULL":
            logging.info(f"跳过NULL文件路径: 索引位置 {idx} 将被跳过")
            return True
        if not os.path.exists(file_path):
            logging.info(f"警告：文件不存在，跳过 - {file_path}")
            return True
        return False

    @staticmethod
    def _process_csv_row(row, idx):
        """处理CSV文件中的一行数据"""
        op_name = row['Op Name']
        if op_name not in op_data:
            return

        op_cases = op_data[op_name]
        if idx < len(op_cases):
            case_str, _ = op_cases[idx]
            op_cases[idx] = (case_str, row['Task Duration(us)'])
        else:
            logging.info(f"警告：算子 {op_name} 在索引 {idx} 处没有对应的case, 跳过")

    def deal_cache_files(self, lib_dir: Path, case_idx: int, cases_range: int):
        """处理缓存目录中的文件"""
        if lib_dir is None:
            return

        for op_name_dir in lib_dir.iterdir():
            if op_name_dir.is_dir() is False:
                continue
            try:
                op_name = op_name_dir.name
                npu_kernel_dir = next(op_name_dir.iterdir())
                search_path = npu_kernel_dir / "search.txt"
                selected_cases = self.deal_search_file(search_path, op_name, cases_range)
                if len(selected_cases) > case_idx:
                    config_path = npu_kernel_dir / "config.txt"
                    self.write_to_config(selected_cases[case_idx][0], config_path)

                    kernel_so_path = npu_kernel_dir / "kernel.so"
                    self.remove_kernel(kernel_so_path)

            except Exception as e:
                logging.error(f"处理缓存下{op_name_dir}目录时出错：{e}")
                continue

    def submit_pgo_tasks(self, inductor_dir: Path, exec_mode=None):
        """提交 任务 到线程池"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.execute_test, exec_mode)

    def get_path_to_time_csv(self, results_dir: str) -> list:
        """获取所有时间统计CSV文件的路径"""
        prof_dirs = self.get_all_filenames_with_prefix(results_dir, PROFILING_DIR_PREFIX)
        if not prof_dirs:
            return []

        prof_dirs_with_time = []
        for prof_dir in prof_dirs:
            dir_path = os.path.join(results_dir, prof_dir)
            ctime = os.path.getctime(dir_path)
            prof_dirs_with_time.append((prof_dir, ctime))

        prof_dirs_with_time.sort(key=lambda x: x[1])

        time_csv_paths = []
        for prof_dir, _ in prof_dirs_with_time:
            prof_dir_path = os.path.join(results_dir, prof_dir, PROFILER_OUTPUT_DIR)

            summary_files = self.get_all_filenames_with_prefix(prof_dir_path, SUMMARY_FILE_PREFIX)
            if not summary_files:
                time_csv_paths.append("NULL")
            else:
                time_csv_name = summary_files[0]
                time_csv_path = os.path.join(prof_dir_path, time_csv_name)
                time_csv_paths.append(time_csv_path)
        return time_csv_paths

    def process_op_data(self, lib_dir: Path):
        """处理op_data, 找到每个算子中执行时间最小的case, 并写入对应的配置文件"""
        for op_name, cases in op_data.items():
            numeric_cases = []
            for case in cases:
                duration = case[1]
                try:
                    duration_float = float(duration)
                    numeric_cases.append((case, duration_float))
                except (ValueError, TypeError):
                    continue

            if not numeric_cases:
                continue

            min_case_tuple = min(numeric_cases, key=lambda x: x[1])
            min_case = min_case_tuple[0]

            selected_case = min_case[0]

            config_path = self.get_config_path(lib_dir, op_name)
            self.write_to_config(selected_case, config_path)

    def read_csv(self, test_dir, results_dir):
        """处理目录中的所有csv文件"""
        file_list = self.get_path_to_time_csv(test_dir)
        valid_file_idx = 0

        for file_path in file_list:
            file_path_str = str(file_path)
            if self._should_skip_file(file_path_str, valid_file_idx):
                valid_file_idx += 1
                continue

            try:
                self._process_single_csv(file_path_str, valid_file_idx)
            except Exception as e:
                logging.info(f"处理文件 {file_path_str} 时出错: {str(e)}", file=sys.stderr)

            valid_file_idx += 1

        return op_data

    def run(self, cases_range=10, test_dir=None, exec_mode=None, results_dir=None):
        cache_dir = test_dir / ".npu_kernels"
        if test_dir is not None:
            for idx in range(cases_range + 1):
                logging.info(f"第 {idx + 1} 次执行")
                self.deal_cache_files(cache_dir, idx, cases_range)

                if test_dir is not None:
                    self.submit_pgo_tasks(test_dir, exec_mode)
        self.read_csv(test_dir, results_dir)
        self.process_op_data(cache_dir)

    def _process_single_csv(self, file_path, idx):
        """处理单个CSV文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._process_csv_row(row, idx)


class ProfilerDeal(MsprofDeal):
    def process_op_data(self, lib_dir: Path):
        """处理算子数据"""
        for op_name, cases in op_data.items():
            numeric_cases = []
            for case in cases:
                duration = case[1]
                try:
                    duration_float = float(duration)
                    numeric_cases.append((case[0], duration_float))
                except (ValueError, TypeError):
                    logging.warning("无法将 %s 转换为浮点数", duration)
                    continue
            if not numeric_cases:
                continue

            min_case = min(numeric_cases, key=lambda x: x[1])

            logging.info(f"Operator Name: {op_name}")
            logging.info(f"Minimum Duration Case: {min_case}")

            config_path = self.get_config_path(lib_dir, op_name)
            self.write_to_config(min_case[0], config_path)

    def read_csv(self, test_dir, results_dir):
        """处理目录中的所有 csv 文件，每个 reader 只更新一个均值"""
        file_list = self.get_path_to_time_csv(results_dir)
        valid_file_idx = 0

        for file_path in file_list:
            file_path_str = str(file_path)
            if self._should_skip_file(file_path_str, valid_file_idx):
                valid_file_idx += 1
                continue

            try:
                self._process_csv_file(file_path_str, valid_file_idx)
            except Exception as e:
                logging.info(f"处理文件{file_path_str}时出错：{str(e)}")

            valid_file_idx += 1

    def get_path_to_time_csv(self, results_dir: str) -> list:
        """获取所有时间统计CSV文件的路径"""
        csv_files = []

        for second_level_dir in os.listdir(results_dir):
            second_level_dir_path = os.path.join(results_dir, second_level_dir)
            if not os.path.isdir(second_level_dir_path):
                continue

            self._process_profiling_dirs(second_level_dir_path, csv_files)

        return csv_files

    def _should_skip_file(self, file_path, idx):
        """判断是否应该跳过当前文件"""
        if file_path.upper() == "NULL":
            logging.info(f"跳过NULL文件路径: 索引位置 {idx} 将被跳过")
            return True
        if not os.path.exists(file_path):
            logging.info(f"文件路径不存在: {file_path}")
            return True
        return False

    def _process_csv_file(self, file_path, idx):
        """处理单个CSV文件并计算平均值"""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            duration_sums, counts = self._calculate_duration_sums(reader)
            self._update_op_data_with_averages(duration_sums, counts, idx)

    def _calculate_duration_sums(self, reader):
        """计算每个op_name的总持续时间和计数"""
        duration_sums = {}
        counts = {}

        for row in reader:
            op_name = row['Op Name']
            duration = row['Task Duration(us)']

            try:
                duration_float = float(duration)
            except (ValueError, TypeError):
                continue

            if op_name in duration_sums:
                duration_sums[op_name] += duration_float
                counts[op_name] += 1
            else:
                duration_sums[op_name] = duration_float
                counts[op_name] = 1

        return duration_sums, counts

    def _update_op_data_with_averages(self, duration_sums, counts, idx):
        """用计算的平均值更新op_data"""
        for op_name, total_duration in duration_sums.items():
            avg_duration = total_duration / counts[op_name]

            if op_name not in op_data:
                continue

            op_cases = op_data[op_name]
            if idx < len(op_cases):
                case_str, _ = op_cases[idx]
                op_cases[idx] = (case_str, str(avg_duration))
            else:
                logging.info(f"警告：算子 {op_name} 在索引 {idx} 处没有对应的case")

    def _process_profiling_dirs(self, base_dir: str, csv_files: list):
        """处理分析目录并收集CSV文件路径"""
        prof_dirs = self.get_all_filenames_with_prefix(base_dir, PROFILING_DIR_PREFIX)
        for prof_dir in prof_dirs:
            prof_dir_path = os.path.join(base_dir, prof_dir, PROFILER_OUTPUT_DIR)
            self._collect_summary_files(prof_dir_path, csv_files)

    def _collect_summary_files(self, dir_path: str, csv_files: list):
        """收集指定目录中的摘要文件"""
        summary_files = self.get_all_filenames_with_prefix(dir_path, SUMMARY_FILE_PREFIX)
        if not summary_files:
            return

        for summary_file in summary_files:
            csv_file_path = os.path.join(dir_path, summary_file)
            csv_files.append(csv_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            '运行 asc_pgo_v2 脚本命令: '
            'python3 asc_pgo_v2.py -n 3 -test_path ./ '
            '-exec_mode "python test.py" -results_dir prof_file'
        )
    )
    parser.add_argument('-n', type=int, default=10, help='top n个case参与回归')
    parser.add_argument('-test_path', type=str, help='指定测试脚本与.npu_kernels缓存路径')
    parser.add_argument('-exec_mode', type=str, help='执行命令模式，例如 "msprof python test.py" 或 "python test.py"')
    parser.add_argument('-results_dir', type=str, help='指定csv文件所在目录, profiler指定到代码中的目录, msprof指定到PROF*的父目录')
    args = parser.parse_args()

    test_dir = Path(args.test_path) if args.test_path else None
    if args.exec_mode and "msprof" in args.exec_mode.split():
        dealer = MsprofDeal()
    else:
        dealer = ProfilerDeal()
    dealer.run(args.n, test_dir, args.exec_mode, args.results_dir)
