import unittest

import pytest

import graphsignal.watcher
from graphsignal.recorders.nvml_recorder import NVMLRecorder
from test.test_utils import configure_test_watcher, find_metric


def has_nvidia_gpu() -> bool:
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count > 0
    except Exception:
        return False


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class NVMLRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def test_setup_without_gpu_is_noop(self):
        if has_nvidia_gpu():
            self.skipTest("NVIDIA GPU available; no-op path not applicable")
        recorder = NVMLRecorder()
        recorder.setup()  # must not raise
        recorder.on_tick()  # must not raise

        self.assertEqual(self.watcher.metric_store().export(), [])
        recorder.shutdown()

    @pytest.mark.cuda
    def test_record(self):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")
        if not has_torch():
            self.skipTest("torch not installed in the active Python environment")

        recorder = NVMLRecorder()
        recorder.setup()

        import torch
        model = torch.nn.Conv2d(1, 1, kernel_size=(1, 1))
        if torch.cuda.is_available():
            model = model.cuda()

        x = torch.arange(-5, 5, 0.1).view(1, 1, -1, 1)
        if torch.cuda.is_available():
            x = x.cuda()
        _ = model(x)

        recorder.on_tick()

        exported = self.watcher.metric_store().export()
        gpu_metrics = [m for m in exported if m.name.startswith('gpu_')]
        self.assertGreater(len(gpu_metrics), 0)
        for metric in gpu_metrics:
            self.assertIn(metric.type, ('gauge', 'counter'))

        expected_gauges = [
            'gpu_utilization_percent', 'gpu_memory_used_bytes', 'gpu_memory_total_bytes',
            'gpu_temperature_celsius', 'gpu_power_usage_watts',
        ]
        emitted = {m.name for m in gpu_metrics}
        self.assertTrue(emitted & set(expected_gauges),
                        f'no expected gpu gauges among {sorted(emitted)}')

        resources = self.watcher.resource_store().export()
        device_resources = [r for r in resources if r['kind'] == 'device']
        self.assertGreaterEqual(len(device_resources), 1)
        attrs = device_resources[0]['attributes']
        self.assertIn('device.name', attrs)
        self.assertIn('architecture', attrs)
        self.assertIn('compute_capability', attrs)

    def test_xid_errors_emitted_as_running_total_counter(self):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")

        recorder = NVMLRecorder()
        recorder.setup()

        self.watcher.log_store().clear()

        recorder._pending_xid_error_codes[0] = [1, 2, 3]
        recorder.on_tick()

        metric = find_metric(self.watcher.metric_store().export(),
                             'gpu_xid_critical_errors')
        self.assertIsNotNone(metric)
        self.assertEqual(metric.type, 'counter')
        self.assertEqual(metric.datapoint['total'], 3)

        # A tick with no new XID errors must not re-count the drained ones.
        recorder.on_tick()
        metric = find_metric(self.watcher.metric_store().export(),
                             'gpu_xid_critical_errors')
        self.assertEqual(metric.datapoint['total'], 3)

        # Recorder-held running total: further errors add to the total.
        recorder._pending_xid_error_codes[0] = [4]
        recorder.on_tick()
        metric = find_metric(self.watcher.metric_store().export(),
                             'gpu_xid_critical_errors')
        self.assertEqual(metric.datapoint['total'], 4)

        xid_entries = [e for e in self.watcher.log_store().export()
                       if 'XID error' in (e['message'] or '')]
        self.assertEqual(len(xid_entries), 4)
        for entry in xid_entries:
            self.assertEqual(entry['level'], 'error')
        codes = {int(e['message'].split('XID error')[1].strip())
                 for e in xid_entries}
        self.assertEqual(codes, {1, 2, 3, 4})


if __name__ == '__main__':
    unittest.main()
