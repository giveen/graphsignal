import time
import unittest
from unittest.mock import patch

from graphsignal.collector.signal_uploader import SignalUploader
from graphsignal.watcher.env_vars import DEFAULT_API_BASE
from graphsignal.proto import signals_pb2
from test.http_server import HttpTestServer


def _metric_proto():
    metric = signals_pb2.Metric()
    metric.name = 'metric1'
    metric.type = signals_pb2.Metric.MetricType.GAUGE_METRIC
    dp = metric.datapoints.add()
    dp.gauge = 42.0
    dp.measurement_ts = int(time.time())
    return metric


class SignalUploaderTest(unittest.TestCase):
    def test_default_api_base(self):
        uploader = SignalUploader('k1')
        self.assertEqual(uploader._api_base, DEFAULT_API_BASE)

    @patch.object(SignalUploader, '_post')
    def test_flush_empties_buffer(self, mocked_post):
        uploader = SignalUploader('k1', api_base='http://127.0.0.1:1')
        uploader.upload_metric(_metric_proto())
        uploader.flush()

        mocked_post.assert_called_once()
        self.assertEqual(len(uploader._buffer), 0)

    @patch.object(SignalUploader, '_post')
    def test_flush_failure_rebuffers(self, mocked_post):
        mocked_post.side_effect = Exception('Ex1')

        uploader = SignalUploader('k1', api_base='http://127.0.0.1:1')
        uploader.upload_metric(_metric_proto())
        uploader.upload_metric(_metric_proto())
        uploader.flush()

        self.assertEqual(len(uploader._buffer), 2)

    @patch.object(SignalUploader, '_post')
    def test_flush_with_empty_buffer_does_not_post(self, mocked_post):
        uploader = SignalUploader('k1', api_base='http://127.0.0.1:1')
        uploader.flush()
        mocked_post.assert_not_called()

    def test_upload_signals(self):
        server = HttpTestServer(None)
        server.set_response_data(b'')
        server.start()
        server.wait_ready()

        uploader = SignalUploader(
            'k1', api_base=f'http://localhost:{server.get_port()}')

        log_batch = signals_pb2.LogBatch()
        tag = log_batch.tags.add()
        tag.key = 't1'
        tag.value = '1'
        entry = log_batch.log_entries.add()
        entry.level = signals_pb2.LogEntry.LogLevel.INFO_LEVEL
        entry.message = 'test log'
        entry.log_ts = time.time_ns()

        resource = signals_pb2.Resource()
        resource.kind = 'process'
        resource.first_seen_ts = 1
        resource.last_seen_ts = 2

        uploader.upload_metric(_metric_proto())
        uploader.upload_log_batch(log_batch)
        uploader.upload_resource(resource)
        uploader.flush()

        request_data = server.get_request_data()
        request_path = server.get_request_path()
        server.join(timeout=2.0)

        self.assertEqual(request_path, '/api/v1/ingest')

        upload_request = signals_pb2.UploadRequest()
        upload_request.ParseFromString(request_data)

        self.assertGreater(upload_request.upload_ts, 0)

        self.assertEqual(len(upload_request.metrics), 1)
        self.assertEqual(upload_request.metrics[0].name, 'metric1')
        self.assertEqual(len(upload_request.metrics[0].datapoints), 1)
        self.assertEqual(upload_request.metrics[0].datapoints[0].gauge, 42.0)

        self.assertEqual(len(upload_request.log_batches), 1)
        batch = upload_request.log_batches[0]
        self.assertEqual(len(batch.log_entries), 1)
        self.assertEqual(batch.log_entries[0].message, 'test log')
        self.assertEqual(batch.log_entries[0].level,
                         signals_pb2.LogEntry.LogLevel.INFO_LEVEL)

        self.assertEqual(len(upload_request.resources), 1)
        self.assertEqual(upload_request.resources[0].kind, 'process')

        self.assertEqual(len(uploader._buffer), 0)


if __name__ == '__main__':
    unittest.main()
