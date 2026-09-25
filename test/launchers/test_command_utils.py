import os
import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

from graphsignal.launchers import command_utils
from graphsignal.launchers.command_utils import (
    extract_host, extract_port, resolve_metrics_host, resolve_metrics_port,
    start_watcher)


class RemovedHelpersTest(unittest.TestCase):
    def test_workload_and_engine_helpers_removed(self):
        self.assertFalse(hasattr(command_utils, 'hash_workload_id'))
        self.assertFalse(hasattr(command_utils, 'engine_version'))


class ExtractHostTest(unittest.TestCase):
    def test_space_form(self):
        self.assertEqual(extract_host(['trtllm-serve', '--host', '0.0.0.0']),
                         '0.0.0.0')

    def test_equals_form(self):
        self.assertEqual(extract_host(['trtllm-serve', '--host=localhost']),
                         'localhost')

    def test_absent_returns_default(self):
        self.assertIsNone(extract_host(['trtllm-serve']))
        self.assertEqual(extract_host(['trtllm-serve'], default='localhost'),
                         'localhost')


class ResolveMetricsHostTest(unittest.TestCase):
    def test_explicit_host_wins(self):
        self.assertEqual(
            resolve_metrics_host('10.0.0.1', ['trtllm-serve', '--host', '0.0.0.0'],
                                 default='localhost'),
            '10.0.0.1')

    def test_falls_back_to_engine_host(self):
        self.assertEqual(
            resolve_metrics_host(None, ['trtllm-serve', '--host', '0.0.0.0'],
                                 default='localhost'),
            '0.0.0.0')

    def test_falls_back_to_default(self):
        self.assertEqual(
            resolve_metrics_host(None, ['trtllm-serve'], default='localhost'),
            'localhost')


class ExtractPortTest(unittest.TestCase):
    def test_space_form(self):
        self.assertEqual(extract_port(['vllm', 'serve', '--port', '8001']), 8001)

    def test_equals_form(self):
        self.assertEqual(extract_port(['vllm', 'serve', '--port=8001']), 8001)

    def test_absent_returns_default(self):
        self.assertIsNone(extract_port(['vllm', 'serve']))
        self.assertEqual(extract_port(['vllm', 'serve'], default=8000), 8000)

    def test_non_int_returns_default(self):
        self.assertEqual(extract_port(['vllm', '--port', 'abc'], default=8000), 8000)

    def test_trailing_port_flag_without_value(self):
        self.assertEqual(extract_port(['vllm', '--port'], default=8000), 8000)


class ResolveMetricsPortTest(unittest.TestCase):
    def test_explicit_port_wins(self):
        # Explicit --metrics-port overrides both the engine --port and default.
        self.assertEqual(
            resolve_metrics_port(9999, ['vllm', '--port', '8001'], default=8000), 9999)

    def test_falls_back_to_engine_port(self):
        self.assertEqual(
            resolve_metrics_port(None, ['vllm', '--port', '8001'], default=8000), 8001)

    def test_falls_back_to_default(self):
        self.assertEqual(
            resolve_metrics_port(None, ['vllm', 'serve'], default=8000), 8000)

    def test_no_default_returns_none(self):
        self.assertIsNone(resolve_metrics_port(None, ['app.py'], default=None))


class StartWatcherTest(unittest.TestCase):
    def _user_args(self, cmd):
        """The argv after the leading `python -m ... --pid PID` prefix, minus
        the `--status-fd N` pair the parent always appends (its own tests
        cover that)."""
        self.assertEqual(cmd[:4], [sys.executable,
                                   '-m', 'graphsignal.commands.graphsignal_watch',
                                   '--pid'])
        rest = cmd[5:]
        if '--status-fd' in rest:
            self.assertEqual(rest[-2], '--status-fd')
            rest = rest[:-2]
        return rest

    def test_spawn_args_default(self):
        fake_popen = MagicMock(name='Popen')
        with patch.object(subprocess, 'Popen', return_value=fake_popen) as popen_m:
            result = start_watcher(12345)

        self.assertIs(result, fake_popen)
        popen_m.assert_called_once()
        cmd = popen_m.call_args[0][0]
        kwargs = popen_m.call_args[1]

        self.assertEqual(cmd[:5], [sys.executable,
                                   '-m', 'graphsignal.commands.graphsignal_watch',
                                   '--pid', '12345'])
        self.assertEqual(self._user_args(cmd), [])
        self.assertTrue(kwargs.get('start_new_session'))

    def test_spawns_with_a_status_pipe(self):
        # The child's stderr is discarded, so the /signals endpoint's bind
        # failures need a pipe back to this process. The read end is pumped by
        # a thread, which is stubbed here so no real fd outlives the test.
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            with patch.object(os, 'pipe', return_value=(21, 22)) as pipe_m:
                with patch.object(os, 'close') as close_m:
                    with patch.object(command_utils.threading, 'Thread') as thread_m:
                        start_watcher(12345)

        pipe_m.assert_called_once()
        cmd = popen_m.call_args[0][0]
        kwargs = popen_m.call_args[1]
        self.assertEqual(cmd[-2:], ['--status-fd', '22'])
        self.assertEqual(tuple(kwargs['pass_fds']), (22,))
        # The parent closes its copy of the write end, or the reader thread
        # would never see EOF when the child exits.
        self.assertIn(22, [call.args[0] for call in close_m.call_args_list])
        # ... and pumps the read end on a background thread.
        thread_m.assert_called_once()
        self.assertEqual(thread_m.call_args.kwargs['args'], (21,))

    def test_spawn_args_with_metrics_port(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(self._user_args(cmd), ['--metrics-port', '8000'])

    def test_spawn_args_with_metrics_path(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000,
                          metrics_path='/prometheus/metrics')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            self._user_args(cmd),
            ['--metrics-port', '8000', '--metrics-path', '/prometheus/metrics'])

    def test_spawn_args_with_metrics_host(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000, metrics_host='localhost')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            self._user_args(cmd), ['--metrics-port', '8000', '--metrics-host', 'localhost'])

    def test_spawn_args_with_listen_port(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, listen_port=18400)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(self._user_args(cmd), ['--listen-port', '18400'])

    def test_spawn_args_with_listen_host(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, listen_host='0.0.0.0')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(self._user_args(cmd), ['--listen-host', '0.0.0.0'])

    def test_spawn_args_with_all_kwargs(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000,
                          metrics_path='/prometheus/metrics',
                          metrics_host='localhost', listen_host='0.0.0.0',
                          listen_port=18400)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            self._user_args(cmd),
            ['--metrics-port', '8000', '--metrics-path', '/prometheus/metrics',
             '--metrics-host', 'localhost', '--listen-host', '0.0.0.0',
             '--listen-port', '18400'])

    def test_returns_none_on_failure(self):
        with patch.object(subprocess, 'Popen', side_effect=OSError('boom')):
            with patch.object(os, 'close') as close_m:
                result = start_watcher(1)
        self.assertIsNone(result)
        # Both pipe ends are released when there is no child to write one.
        closed = [call.args[0] for call in close_m.call_args_list]
        self.assertEqual(len(closed), 2)


if __name__ == '__main__':
    unittest.main()
