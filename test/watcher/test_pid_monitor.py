import os
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from graphsignal.watcher.pid_monitor import PidMonitor


class _RecordingListener:
    def __init__(self):
        self.known = []
        self.created = 0
        self.children_created = []
        self.children_terminated = []
        self.terminated = 0

    def on_target_known(self, pid):
        self.known.append(pid)

    def on_target_created(self, args):
        self.created += 1

    def on_child_created(self, pid, args):
        self.children_created.append(pid)

    def on_child_terminated(self, pid):
        self.children_terminated.append(pid)

    def on_target_terminated(self):
        self.terminated += 1


class PidMonitorTest(unittest.TestCase):
    def test_create_time_mismatch_treats_reused_pid_as_terminated(self):
        class _Process:
            def __init__(self, create_times):
                self._create_times = iter(create_times)

            def is_running(self):
                return True

            def status(self):
                return 'running'

            def create_time(self):
                return next(self._create_times)

            def children(self, recursive=True):
                return []

            def cmdline(self):
                return ['python', 'target.py']

        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=1234)
        monitor.add_listener(listener)
        process = _Process([100.0, 200.0])
        with patch('graphsignal.watcher.pid_monitor.psutil.Process', return_value=process):
            monitor._tick()
            monitor._tick()

        self.assertEqual(listener.created, 1)
        self.assertEqual(listener.terminated, 1)

    def test_matching_create_time_keeps_target_alive(self):
        class _Process:
            def is_running(self):
                return True

            def status(self):
                return 'running'

            def create_time(self):
                return 100.0

            def children(self, recursive=True):
                return []

            def cmdline(self):
                return ['python', 'target.py']

        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=1234)
        monitor.add_listener(listener)
        with patch('graphsignal.watcher.pid_monitor.psutil.Process', return_value=_Process()):
            monitor._tick()
            monitor._tick()

        self.assertEqual(listener.created, 1)
        self.assertEqual(listener.terminated, 0)

    def test_on_target_known_emitted_at_setup_before_polling(self):
        # on_target_known fires synchronously from setup(), so a consumer of a
        # pid-derived artifact is wired up even if the target never polls alive.
        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=os.getpid(), poll_interval=0.05)
        monitor.add_listener(listener)
        try:
            monitor.setup()
            self.assertEqual(listener.known, [os.getpid()])
        finally:
            monitor.shutdown()

    def test_on_target_created_for_live_target(self):
        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=os.getpid(), poll_interval=0.05)
        monitor.add_listener(listener)
        try:
            monitor.setup()
            deadline = time.time() + 2.0
            while listener.created == 0 and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(listener.created, 1)
            self.assertEqual(listener.terminated, 0)
        finally:
            monitor.shutdown()

    def test_on_target_known_fires_even_for_dead_target(self):
        # Pick a pid that is (almost certainly) not running.
        dead_pid = 999999
        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=dead_pid, poll_interval=0.05)
        monitor.add_listener(listener)
        try:
            monitor.setup()
            # Known fires from setup regardless of liveness; the poll then sees
            # it dead and reports terminated without ever reporting created.
            self.assertEqual(listener.known, [dead_pid])
            time.sleep(0.2)
            self.assertEqual(listener.created, 0)
            self.assertGreaterEqual(listener.terminated, 1)
        finally:
            monitor.shutdown()

    def test_child_created_and_terminated(self):
        listener = _RecordingListener()
        monitor = PidMonitor(target_pid=os.getpid(), poll_interval=0.05)
        monitor.add_listener(listener)
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])
        try:
            monitor.setup()
            deadline = time.time() + 3.0
            while child.pid not in listener.children_created \
                    and time.time() < deadline:
                time.sleep(0.02)
            self.assertIn(child.pid, listener.children_created)

            child.terminate()
            child.wait()
            deadline = time.time() + 3.0
            while child.pid not in listener.children_terminated \
                    and time.time() < deadline:
                time.sleep(0.02)
            self.assertIn(child.pid, listener.children_terminated)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            monitor.shutdown()

    def test_listener_exception_in_on_target_known_does_not_break_setup(self):
        class _Boom:
            def on_target_known(self, pid):
                raise RuntimeError('boom')

        monitor = PidMonitor(target_pid=os.getpid(), poll_interval=0.05)
        monitor.add_listener(_Boom())
        try:
            monitor.setup()  # must not raise
        finally:
            monitor.shutdown()


if __name__ == '__main__':
    unittest.main()
