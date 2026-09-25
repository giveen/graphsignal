import os
import tempfile
import unittest
from unittest.mock import patch

from graphsignal.launchers import ninfer_launcher as ninfer_mod
from graphsignal.launchers.ninfer_launcher import NinferLauncher

from test.launchers._helpers import LaunchFixture


class NinferMatchTest(unittest.TestCase):
    def test_matches_supported_executables(self):
        self.assertTrue(NinferLauncher(['ninfer', 'run', 'm']).match())
        self.assertTrue(NinferLauncher(['ninfer-serve', 'm']).match())
        self.assertTrue(NinferLauncher(['ninfer-perplexity', 'm']).match())
        self.assertTrue(NinferLauncher(['/opt/ninfer/bin/ninfer']).match())
        # The benchmark harness, in both spellings the build produces.
        self.assertTrue(NinferLauncher(['ninfer-bench', 'm']).match())
        self.assertTrue(NinferLauncher(['build/bench/ninfer_bench']).match())

    def test_does_not_match_other_commands(self):
        self.assertFalse(NinferLauncher(['python', 'app.py']).match())
        self.assertFalse(NinferLauncher(['ninfer-other']).match())


class NinferLaunchTest(unittest.TestCase):
    ENV_VAR = 'GRAPHSIGNAL_NINFER_JSONL'

    def test_regular_ninfer_preserves_argv_and_has_no_prometheus_config(self):
        launcher = NinferLauncher(
            ['ninfer', 'run', '--model', 'm'], metrics_port=9999,
            listen_host='127.0.0.1', listen_port=18303)
        with LaunchFixture(ninfer_mod) as fx:
            launcher.launch()

        fx.cupti_env_m.assert_called_once_with(cuda_graph_trace=None)
        fx.rocm_env_m.assert_called_once_with(cuda_graph_trace=None)
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'run', '--model', 'm'],
            metrics_port=None, listen_host='127.0.0.1', listen_port=18303)

    def test_perplexity_preserves_argv(self):
        launcher = NinferLauncher(
            ['/opt/bin/ninfer-perplexity', '--foo', 'bar'])
        with LaunchFixture(ninfer_mod) as fx:
            launcher.launch()
        self.assertEqual(
            fx.launched_argv,
            ['/abs/exec', '--foo', 'bar'])

    def test_bench_preserves_argv_and_adds_no_request_log(self):
        # The request log is a serve-only flag: appending it would make the
        # harness exit on an unknown argument. Its report path is the user's
        # to choose, so argv passes through byte-for-byte.
        launcher = NinferLauncher([
            'build/bench/ninfer_bench', '--weights', 'm.ninfer',
            '-o', 'json', '--output-file', 'out.json'])
        with LaunchFixture(ninfer_mod) as fx:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(NinferLaunchTest.ENV_VAR, None)
                launcher.launch()
        self.assertEqual(
            fx.launched_argv,
            ['/abs/exec', '--weights', 'm.ninfer', '-o', 'json',
             '--output-file', 'out.json'])
        self.assertNotIn(NinferLaunchTest.ENV_VAR, os.environ)

    def test_bench_still_enables_the_ninfer_nvtx_domain(self):
        # The bench shares the engine, so it needs the same NVTX opt-in the
        # serve path gets.
        launcher = NinferLauncher(['build/bench/ninfer_bench', 'm.ninfer'])
        with LaunchFixture(ninfer_mod) as fx:
            launcher.launch()
        fx.cupti_env_m.assert_called_once_with(cuda_graph_trace=None)
        self.assertEqual(os.environ.get('GRAPHSIGNAL_NINFER_NVTX'), '1')

    def test_serve_injects_unique_request_log_and_exports_it(self):
        paths = []
        with patch.dict(os.environ, {}, clear=False):
            for _ in range(2):
                launcher = NinferLauncher(['ninfer-serve', '--model', 'm'])
                with LaunchFixture(ninfer_mod) as fx:
                    launcher.launch()
                argv = fx.launched_argv
                flag_index = argv.index('--request-log-jsonl')
                self.assertEqual(flag_index, len(argv) - 2)
                path = argv[-1]
                paths.append(path)
                self.assertFalse(os.path.exists(path))
                self.assertEqual(os.environ[self.ENV_VAR], path)
            # Read inside the patch: the launcher sets these, and the patch
            # rolls them back on exit, so asserting outside it depends on
            # whatever the ambient environment happened to hold. The NVTX
            # library comparison stays an equality because the launcher
            # setdefault()s it, and an already-set value legitimately wins.
            self.assertEqual(os.environ['GRAPHSIGNAL_NINFER_NVTX'], '1')
            self.assertEqual(os.environ.get('NVTX_INJECTION64_PATH'),
                             os.environ.get('CUDA_INJECTION64_PATH'))

        self.assertNotEqual(paths[0], paths[1])

    def test_serve_keeps_user_request_log_and_does_not_delete_it(self):
        with tempfile.TemporaryDirectory() as directory:
            request_log = os.path.join(directory, 'requests.jsonl')
            with open(request_log, 'w') as stream:
                stream.write('user data\n')

            with patch.dict(os.environ, {}, clear=False):
                launcher = NinferLauncher([
                    'ninfer-serve', '--model', 'm',
                    '--request-log-jsonl', request_log])
                with LaunchFixture(ninfer_mod) as fx:
                    launcher.launch()
                self.assertEqual(os.environ[self.ENV_VAR], request_log)

            self.assertEqual(fx.launched_argv, [
                '/abs/exec', '--model', 'm',
                '--request-log-jsonl', request_log])
            self.assertTrue(os.path.exists(request_log))
            with open(request_log) as stream:
                self.assertEqual(stream.read(), 'user data\n')

    def test_serve_keeps_equals_form_request_log(self):
        with tempfile.TemporaryDirectory() as directory:
            request_log = os.path.join(directory, 'requests.jsonl')
            with open(request_log, 'w'):
                pass

            with patch.dict(os.environ, {}, clear=False):
                launcher = NinferLauncher([
                    'ninfer-serve',
                    '--request-log-jsonl=%s' % request_log])
                with LaunchFixture(ninfer_mod) as fx:
                    launcher.launch()
                self.assertEqual(os.environ[self.ENV_VAR], request_log)

            self.assertEqual(fx.launched_argv, [
                '/abs/exec', '--request-log-jsonl=%s' % request_log])
            self.assertTrue(os.path.exists(request_log))

    def test_auto_request_log_is_cleaned_when_launch_fails(self):
        with patch.dict(os.environ, {}, clear=False):
            launcher = NinferLauncher(['ninfer-serve'])
            with LaunchFixture(ninfer_mod) as fx:
                fx.launch_supervised_m.side_effect = RuntimeError('failed')
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    launcher.launch()

            path = fx.launched_argv[-1]
            self.assertFalse(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
