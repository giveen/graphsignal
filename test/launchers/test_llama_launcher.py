import unittest

from graphsignal.launchers import llama_launcher as llama_mod
from graphsignal.launchers.llama_launcher import LlamaLauncher

from test.launchers._helpers import LaunchFixture


class LlamaMatchTest(unittest.TestCase):
    def test_matches(self):
        self.assertTrue(LlamaLauncher(['llama-server', 'model.gguf']).match())
        self.assertTrue(LlamaLauncher(['/opt/llama.cpp/build/bin/llama-server']).match())

    def test_does_not_match_other_commands(self):
        self.assertFalse(LlamaLauncher(['llama-cli', 'model.gguf']).match())
        self.assertFalse(LlamaLauncher(['python', 'app.py']).match())


class LlamaLaunchTest(unittest.TestCase):
    def test_enables_metrics_and_preserves_argv(self):
        launcher = LlamaLauncher(
            ['llama-server', 'model.gguf', '--host', '0.0.0.0', '--port', '8081'],
            listen_host='127.0.0.1', listen_port=18304)
        with LaunchFixture(llama_mod) as fx:
            launcher.launch()

        fx.cupti_env_m.assert_called_once_with(cuda_graph_trace=None)
        fx.rocm_env_m.assert_called_once_with(cuda_graph_trace=None)
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'model.gguf', '--host', '0.0.0.0', '--port', '8081',
             '--metrics'],
            metrics_port=8081, metrics_path='/metrics', metrics_host='127.0.0.1',
            listen_host='127.0.0.1', listen_port=18304)

    def test_preserves_explicit_metrics_flag(self):
        launcher = LlamaLauncher(['llama-server', 'model.gguf', '--metrics'])
        with LaunchFixture(llama_mod) as fx:
            launcher.launch()
        self.assertEqual(fx.launched_argv, ['/abs/exec', 'model.gguf', '--metrics'])
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'model.gguf', '--metrics'],
            metrics_port=8080, metrics_path='/metrics', metrics_host='127.0.0.1',
            listen_host=None, listen_port=None)

    def test_explicit_metrics_port_wins(self):
        launcher = LlamaLauncher(
            ['llama-server', 'model.gguf', '--port', '8081'], metrics_port=9999)
        with LaunchFixture(llama_mod) as fx:
            launcher.launch()
        self.assertEqual(
            fx.launch_supervised_m.call_args.kwargs['metrics_port'], 9999)

    def test_equals_form_metrics_flag_is_preserved(self):
        launcher = LlamaLauncher(['llama-server', 'model.gguf', '--metrics=true'])
        with LaunchFixture(llama_mod) as fx:
            launcher.launch()
        self.assertEqual(fx.launched_argv,
                         ['/abs/exec', 'model.gguf', '--metrics=true'])

    def test_api_prefix_moves_metrics_path(self):
        launcher = LlamaLauncher(['llama-server', 'model.gguf', '--api-prefix', 'api'])
        with LaunchFixture(llama_mod) as fx:
            launcher.launch()
        self.assertEqual(fx.launch_supervised_m.call_args.kwargs['metrics_path'],
                         '/api/metrics')

    def test_auth_and_tls_modes_warn(self):
        for flag in ('--api-key', '--api-keys-file', '--ssl-key-file'):
            launcher = LlamaLauncher(['llama-server', 'model.gguf', flag, 'value'])
            with LaunchFixture(llama_mod) as fx:
                with self.assertLogs('graphsignal', level='WARNING') as logs:
                    launcher.launch()
            self.assertTrue(any('metrics may be unavailable' in message
                                for message in logs.output), flag)


if __name__ == '__main__':
    unittest.main()
