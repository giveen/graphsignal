import sys
import unittest
from unittest.mock import patch

from graphsignal.commands import graphsignal_run
from graphsignal.launchers.vllm_launcher import VllmLauncher
from graphsignal.launchers.sglang_launcher import SglangLauncher
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher
from graphsignal.launchers.fallback_launcher import FallbackLauncher


class GraphsignalRunDispatchTest(unittest.TestCase):
    """`graphsignal-run` walks the launcher list and dispatches to the first
    one whose match() returns True."""

    def _run_with_argv(self, argv):
        with patch.object(VllmLauncher, 'launch') as v, \
             patch.object(SglangLauncher, 'launch') as s, \
             patch.object(TrtllmLauncher, 'launch') as t, \
             patch.object(FallbackLauncher, 'launch') as f, \
             patch.object(sys, 'argv', argv):
            try:
                graphsignal_run.main()
            except SystemExit:
                pass
        return v, s, t, f

    def test_no_args_exits(self):
        with patch.object(sys, 'argv', ['graphsignal-run']):
            with self.assertRaises(SystemExit) as cm:
                graphsignal_run.main()
        self.assertEqual(cm.exception.code, 1)

    def test_version_flag_prints_and_exits_zero(self):
        # `--version` counts only as the FIRST argument: it is how an install
        # is verified. Later in argv it belongs to the workload.
        from graphsignal.version import __version__
        with patch.object(sys, 'argv', ['graphsignal-run', '--version']), \
             patch('builtins.print') as out:
            with self.assertRaises(SystemExit) as cm:
                graphsignal_run.main()
        self.assertEqual(cm.exception.code, 0)
        out.assert_called_once_with(f'graphsignal-run {__version__}')

    def test_workload_version_flag_is_forwarded(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'python', '--version'])
        f.assert_called_once()

    def test_vllm_wins_over_fallback(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_sglang_executable_wins(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', 'sglang', 'serve', '--model', 'm'])
        s.assert_called_once()
        v.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_sglang_python_module_form_wins(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', 'python', '-m', 'sglang.launch_server'])
        s.assert_called_once()
        f.assert_not_called()

    def test_trtllm_wins(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'trtllm-serve', '--model', 'm'])
        t.assert_called_once()
        v.assert_not_called()
        s.assert_not_called()
        f.assert_not_called()

    def test_unrecognised_falls_back(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'python', 'my_app.py'])
        f.assert_called_once()
        v.assert_not_called()
        s.assert_not_called()
        t.assert_not_called()

    def test_leading_flags_stripped_and_command_still_matches(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', '--metrics-port', '8000',
             '--listen-port', '18400', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()


class ExtractFlagsTest(unittest.TestCase):
    def test_no_flags(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['vllm', 'serve']),
            (None, None, None, ['vllm', 'serve']))

    def test_extracts_metrics_port_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port', '8000', 'trtllm-serve', 'm']),
            (8000, None, None, ['trtllm-serve', 'm']))

    def test_extracts_metrics_port_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port=8000', 'trtllm-serve', 'm']),
            (8000, None, None, ['trtllm-serve', 'm']))

    def test_extracts_listen_port_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port', '18400', 'vllm', 'serve']),
            (None, None, 18400, ['vllm', 'serve']))

    def test_extracts_listen_port_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port=18400', 'vllm', 'serve']),
            (None, None, 18400, ['vllm', 'serve']))

    def test_extracts_listen_host_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-host', '0.0.0.0', 'vllm', 'serve']),
            (None, '0.0.0.0', None, ['vllm', 'serve']))

    def test_extracts_listen_host_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-host=10.0.0.5', 'vllm', 'serve']),
            (None, '10.0.0.5', None, ['vllm', 'serve']))

    def test_listen_host_missing_value_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(['--listen-host'])

    def test_extracts_both_leading_flags_in_any_order(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port', '18400', '--metrics-port', '9001',
                 'vllm', 'serve']),
            (9001, None, 18400, ['vllm', 'serve']))

    def test_unknown_leading_token_starts_workload_command(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['python', '--metrics-port', '8000']),
            (None, None, None, ['python', '--metrics-port', '8000']))

    def test_flags_after_command_left_for_workload(self):
        # Only leading flags are parsed; identically named workload flags
        # later in argv are forwarded untouched.
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['vllm', 'serve', '--metrics-port', '8000']),
            (None, None, None, ['vllm', 'serve', '--metrics-port', '8000']))
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['vllm', 'serve', '--listen-port', '18400']),
            (None, None, None, ['vllm', 'serve', '--listen-port', '18400']))

    def test_invalid_metrics_port_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port', 'abc', 'vllm', 'serve'])

    def test_metrics_port_missing_value_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(['--metrics-port'])

    def test_invalid_listen_port_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(
                ['--listen-port', 'abc', 'vllm', 'serve'])

    def test_listen_port_missing_value_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(['--listen-port'])


class MainFlagForwardingTest(unittest.TestCase):
    def test_main_passes_flags_to_launchers(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, metrics_port=None, listen_host=None,
                         listen_port=None):
            created.append((list(args), metrics_port, listen_host, listen_port))
            return orig_init(self, args, metrics_port=metrics_port,
                             listen_host=listen_host, listen_port=listen_port)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(sys, 'argv',
                          ['graphsignal-run', '--metrics-port', '9001',
                           '--listen-host', '0.0.0.0',
                           '--listen-port', '18400', 'vllm', 'serve']):
            graphsignal_run.main()

        self.assertEqual(created, [(['vllm', 'serve'], 9001, '0.0.0.0', 18400)])

    def test_main_passes_none_flags_when_absent(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, metrics_port=None, listen_host=None,
                         listen_port=None):
            created.append((metrics_port, listen_host, listen_port))
            return orig_init(self, args, metrics_port=metrics_port,
                             listen_host=listen_host, listen_port=listen_port)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(sys, 'argv', ['graphsignal-run', 'vllm', 'serve']):
            graphsignal_run.main()

        self.assertEqual(created, [(None, None, None)])


if __name__ == '__main__':
    unittest.main()
