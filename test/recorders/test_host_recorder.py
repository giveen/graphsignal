import unittest

import graphsignal.watcher
from graphsignal.recorders.host_recorder import HostRecorder
from graphsignal.version import __version__
from test.test_utils import configure_test_watcher, find_metric


class HostRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def test_setup_sets_watcher_tags(self):
        recorder = HostRecorder()
        recorder.setup()
        self.assertIsNotNone(self.watcher.get_tag('host.name'))

    def test_on_tick_emits_host_memory_usage(self):
        recorder = HostRecorder()
        recorder.setup()
        recorder.on_tick()

        metric = find_metric(self.watcher.metric_store().export(),
                             'host_memory_usage_bytes')
        self.assertIsNotNone(metric)
        self.assertEqual(metric.type, 'gauge')
        self.assertGreater(metric.datapoint['value'], 0)

    def test_on_tick_emits_host_resource(self):
        recorder = HostRecorder()
        recorder.setup()
        recorder.on_tick()

        resources = self.watcher.resource_store().export()
        host_resources = [r for r in resources if r['kind'] == 'host']
        self.assertEqual(len(host_resources), 1)
        resource = host_resources[0]

        self.assertIn('platform.name', resource['attributes'])
        self.assertIn('platform.version', resource['attributes'])
        self.assertIn('platform.machine', resource['attributes'])
        self.assertEqual(resource['attributes'].get('profiler.version'),
                         __version__)
        self.assertGreater(resource['first_seen_ts'], 0)


if __name__ == '__main__':
    unittest.main()
