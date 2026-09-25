import os
import random
import time
import unittest

import graphsignal.watcher
import psutil
from graphsignal.recorders.process_recorder import ProcessRecorder
from test.test_utils import configure_test_watcher, find_metric

mem = []


class ProcessRecorderTest(unittest.TestCase):
    def setUp(self):
        self.watcher = configure_test_watcher()

    def tearDown(self):
        graphsignal.watcher.shutdown()

    def test_record(self):
        pid = os.getpid()
        recorder = ProcessRecorder(root_pid=pid, pid=pid,
                                   args='python -m unittest')
        recorder.setup()

        time.sleep(0.2)
        for _ in range(100000):
            random.random()
        global mem
        mem = [1] * 100000

        recorder.on_tick()

        exported = self.watcher.metric_store().export()
        pid_tag = {'process.pid': str(pid)}

        rss = find_metric(exported, 'process_memory_usage_bytes', pid_tag)
        self.assertEqual(rss.type, 'gauge')
        self.assertGreater(rss.datapoint['value'], 0)
        vms = find_metric(exported, 'process_memory_virtual_bytes', pid_tag)
        self.assertEqual(vms.type, 'gauge')
        self.assertGreater(vms.datapoint['value'], 0)

        resources = self.watcher.resource_store().export()
        process_resources = [r for r in resources if r['kind'] == 'process']
        self.assertEqual(len(process_resources), 1)
        resource = process_resources[0]
        self.assertEqual(resource['tags'].get('process.pid'), str(pid))
        # The root's own resource links to itself.
        self.assertEqual(resource['tags'].get('process.root_pid'), str(pid))
        self.assertIn('process.command_line', resource['attributes'])
        # When observing self, runtime attrs are also reported.
        self.assertIn('runtime.name', resource['attributes'])
        self.assertIn('runtime.version', resource['attributes'])

    def test_records_host_side_triage_signals(self):
        # These answer "is this process CPU-saturated, blocked, or starved?"
        # without stacks or privileges, which is the first question when an
        # engine is host-bound.
        pid = os.getpid()
        recorder = ProcessRecorder(root_pid=pid, pid=pid, args='python -m unittest')
        recorder.setup()
        recorder.on_tick()

        exported = self.watcher.metric_store().export()
        pid_tag = {'process.pid': str(pid)}

        user_cpu = find_metric(exported, 'process_user_cpu_seconds', pid_tag)
        self.assertEqual(user_cpu.type, 'gauge')
        self.assertGreaterEqual(user_cpu.datapoint['value'], 0)
        system_cpu = find_metric(exported, 'process_system_cpu_seconds', pid_tag)
        self.assertEqual(system_cpu.type, 'gauge')
        self.assertGreaterEqual(system_cpu.datapoint['value'], 0)

        threads = find_metric(exported, 'process_threads', pid_tag)
        self.assertEqual(threads.type, 'gauge')
        self.assertGreaterEqual(threads.datapoint['value'], 1)

        for name in ('process_context_switches_voluntary_total',
                     'process_context_switches_involuntary_total'):
            metric = find_metric(exported, name, pid_tag)
            if metric is None:
                continue  # not implemented on this platform
            self.assertEqual(metric.type, 'counter')
            self.assertGreaterEqual(metric.datapoint['total'], 0)

    def test_missing_psutil_fields_do_not_break_the_tick(self):
        # num_ctx_switches is not implemented everywhere; the rest of the tick
        # must still land.
        pid = os.getpid()
        recorder = ProcessRecorder(root_pid=pid, pid=pid, args='python -m unittest')
        recorder.setup()

        real = psutil.Process.num_ctx_switches

        def boom(self):
            raise NotImplementedError('not available on this platform')

        psutil.Process.num_ctx_switches = boom
        try:
            recorder.on_tick()
        finally:
            psutil.Process.num_ctx_switches = real

        exported = self.watcher.metric_store().export()
        pid_tag = {'process.pid': str(pid)}
        self.assertIsNotNone(find_metric(exported, 'process_threads', pid_tag))
        self.assertIsNone(find_metric(
            exported, 'process_context_switches_voluntary_total', pid_tag))


if __name__ == '__main__':
    unittest.main()
