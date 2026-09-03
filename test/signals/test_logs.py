import os
import unittest

from graphsignal import version
from graphsignal.signals.logs import LogStore


class LogStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = LogStore()

    def test_entry_shape(self):
        self.store.log_message(
            tags={'t1': '1'}, level='INFO', message='msg1',
            exception='exc1', timestamp_ns=123)

        entries = self.store.export()
        self.assertEqual(entries, [{
            'level': 'info',
            'ts': 123,
            'message': 'msg1',
            'exception': 'exc1',
            'tags': {'t1': '1'},
        }])

    def test_defaults(self):
        self.store.log_message(message='msg1')
        entry = self.store.export()[0]
        self.assertEqual(entry['level'], 'info')
        self.assertGreater(entry['ts'], 0)
        self.assertIsNone(entry['exception'])
        self.assertEqual(entry['tags'], {})

    def test_none_message_dropped(self):
        self.store.log_message(message=None)
        self.assertEqual(self.store.export(), [])

    def test_oversized_message_dropped(self):
        self.store.log_message(message='x' * (LogStore.MESSAGE_SIZE_LIMIT + 1))
        self.assertEqual(self.store.export(), [])

    def test_oversized_exception_dropped(self):
        self.store.log_message(
            message='msg', exception='x' * (LogStore.STACK_TRACE_SIZE_LIMIT + 1))
        self.assertEqual(self.store.export(), [])

    def test_ring_cap(self):
        for i in range(LogStore.MAX_ENTRIES + 50):
            self.store.log_message(message=f'msg{i}', timestamp_ns=i)

        entries = self.store.export()
        self.assertEqual(len(entries), LogStore.MAX_ENTRIES)
        self.assertEqual(entries[0]['message'], 'msg50')
        self.assertEqual(entries[-1]['message'],
                         f'msg{LogStore.MAX_ENTRIES + 49}')

    def test_last_entries_level_filter_order_and_limit(self):
        self.store.log_message(message='d1', level='debug', timestamp_ns=1)
        self.store.log_message(message='i1', level='info', timestamp_ns=2)
        for i in range(12):
            level = 'warning' if i % 2 == 0 else 'error'
            self.store.log_message(message=f'w{i}', level=level,
                                   timestamp_ns=10 + i)
        self.store.log_message(message='c1', level='critical', timestamp_ns=100)

        entries = self.store.last_entries(min_level='warning', limit=10)
        self.assertEqual(len(entries), 10)
        # Newest N, chronological order.
        self.assertEqual([e['message'] for e in entries],
                         [f'w{i}' for i in range(3, 12)] + ['c1'])

    def test_last_entries_min_level_error(self):
        self.store.log_message(message='w1', level='warning', timestamp_ns=1)
        self.store.log_message(message='e1', level='error', timestamp_ns=2)
        entries = self.store.last_entries(min_level='error')
        self.assertEqual([e['message'] for e in entries], ['e1'])

    def test_log_watcher_message_tags_and_version_prefix(self):
        self.store.log_watcher_message(
            tags={'t1': '1'}, level='error', message='boom', timestamp_ns=5)

        entry = self.store.export()[0]
        self.assertEqual(
            entry['message'], f'Graphsignal {version.__version__}: boom')
        self.assertEqual(entry['tags']['t1'], '1')
        self.assertEqual(entry['tags']['scope.name'], 'watcher')
        self.assertEqual(entry['tags']['logger'], 'graphsignal')
        self.assertEqual(entry['tags']['watcher.pid'], os.getpid())

    def test_export_non_destructive(self):
        self.store.log_message(message='msg1', timestamp_ns=1)
        first = self.store.export()
        second = self.store.export()
        self.assertEqual(first, second)

        first[0]['message'] = 'mutated'
        first.append({'level': 'info'})
        self.assertEqual(self.store.export()[0]['message'], 'msg1')
        self.assertEqual(len(self.store.export()), 1)


if __name__ == '__main__':
    unittest.main()
