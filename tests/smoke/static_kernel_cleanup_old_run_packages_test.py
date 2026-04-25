"""
Test cases for cleanup_old_run_packages core scenarios.

Core scenarios:
1. Dead process (abnormal exit) -> uninstall its run package
2. Alive process (multi-process parallel) -> skip, do not uninstall
3. PID recycled (old process exited, new process reuses PID) -> uninstall
4. Mixed: alive + dead + recycled -> only clean dead and recycled
"""

import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from npugraph_ex._acl_concrete_graph.static_kernel import (
    _is_process_alive,
    _get_process_start_time,
    cleanup_old_run_packages,
)


class TestCleanupOldRunPackages(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._record_file = os.path.join(self._tmp_dir, ".static_kernel_records.json")
        self._fake_uninstall_dir = os.path.join(self._tmp_dir, "uninstall_scripts")
        os.makedirs(self._fake_uninstall_dir, exist_ok=True)
        self._records_patcher = mock.patch(
            "npugraph_ex._acl_concrete_graph.static_kernel._update_static_kernel_records"
        )
        self._mock_update_records = self._records_patcher.start()
        self._mock_update_records.side_effect = self._capture_update_func

    def tearDown(self):
        self._records_patcher.stop()
        for f in os.listdir(self._fake_uninstall_dir):
            os.remove(os.path.join(self._fake_uninstall_dir, f))
        if os.path.exists(self._fake_uninstall_dir):
            os.rmdir(self._fake_uninstall_dir)
        if os.path.exists(self._tmp_dir):
            os.rmdir(self._tmp_dir)

    def _capture_update_func(self, update_func):
        with open(self._record_file, "a+") as f:
            f.seek(0)
            content = f.read(1)
            if content:
                f.seek(0)
                records = json.load(f)
            else:
                records = {}
            records = update_func(records)
            f.seek(0)
            f.truncate()
            json.dump(records, f, indent=4)

    def _read_records(self):
        if not os.path.exists(self._record_file):
            return {}
        with open(self._record_file, "r") as f:
            return json.load(f)

    def _write_records(self, records):
        with open(self._record_file, "w") as f:
            json.dump(records, f, indent=4)

    def _create_fake_uninstall_script(self, name):
        script_path = os.path.join(self._fake_uninstall_dir, name)
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\necho 'uninstall success'\n")
        os.chmod(script_path, stat.S_IRWXU)
        return script_path

    def _build_record(self, pid, uninstall_path, process_start_time=None):
        return {
            "uninstall_path": uninstall_path,
            "filename": "test_pkg",
            "pid": pid,
            "process_start_time": process_start_time,
            "record_create_time": "2026-01-01T00:00:00",
        }

    def test_dead_process_uninstalled(self):
        uninstall_script = self._create_fake_uninstall_script("uninstall_dead.sh")
        dead_pid = 4000000
        self._write_records({
            "pkg_dead": self._build_record(dead_pid, uninstall_script, process_start_time=100),
        })

        with mock.patch("npugraph_ex._acl_concrete_graph.static_kernel._is_process_alive", return_value=False):
            cleanup_old_run_packages()

        self.assertNotIn("pkg_dead", self._read_records())

    def test_alive_process_not_uninstalled(self):
        uninstall_script = self._create_fake_uninstall_script("uninstall_alive.sh")
        alive_pid = os.getpid()
        alive_start_time = _get_process_start_time(alive_pid)
        self._write_records({
            "pkg_alive": self._build_record(alive_pid, uninstall_script, process_start_time=alive_start_time),
        })

        cleanup_old_run_packages()

        self.assertIn("pkg_alive", self._read_records())
        self.assertTrue(os.path.exists(uninstall_script))

    def test_pid_recycled_uninstalled(self):
        uninstall_script = self._create_fake_uninstall_script("uninstall_recycled.sh")
        recycled_pid = os.getpid()
        old_start_time = 1
        self._write_records({
            "pkg_recycled": self._build_record(recycled_pid, uninstall_script, process_start_time=old_start_time),
        })

        cleanup_old_run_packages()

        self.assertNotIn("pkg_recycled", self._read_records())

    def test_mixed_records_only_clean_dead_and_recycled(self):
        uninstall_alive = self._create_fake_uninstall_script("uninstall_alive.sh")
        uninstall_dead = self._create_fake_uninstall_script("uninstall_dead.sh")
        uninstall_recycled = self._create_fake_uninstall_script("uninstall_recycled.sh")

        alive_pid = os.getpid()
        alive_start_time = _get_process_start_time(alive_pid)
        dead_pid = 4000000
        recycled_pid = os.getpid() + 1

        self._write_records({
            "pkg_alive": self._build_record(alive_pid, uninstall_alive, process_start_time=alive_start_time),
            "pkg_dead": self._build_record(dead_pid, uninstall_dead, process_start_time=100),
            "pkg_recycled": self._build_record(recycled_pid, uninstall_recycled, process_start_time=1),
        })

        def mock_is_alive(pid):
            if pid == alive_pid:
                return True
            if pid == dead_pid:
                return False
            if pid == recycled_pid:
                return True
            return _is_process_alive(pid)

        def mock_start_time(pid):
            if pid == alive_pid:
                return alive_start_time
            if pid == recycled_pid:
                return 99999
            return _get_process_start_time(pid)

        with mock.patch("npugraph_ex._acl_concrete_graph.static_kernel._is_process_alive", side_effect=mock_is_alive), \
             mock.patch("npugraph_ex._acl_concrete_graph.static_kernel._get_process_start_time", side_effect=mock_start_time):
            cleanup_old_run_packages()

        remaining = self._read_records()
        self.assertIn("pkg_alive", remaining)
        self.assertNotIn("pkg_dead", remaining)
        self.assertNotIn("pkg_recycled", remaining)


if __name__ == "__main__":
    unittest.main()