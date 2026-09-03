import json
import logging
import os
import unittest

from graphsignal.version import __version__
from graphsignal.watcher.version_check import newer_version, start_version_check

from test.http_server import HttpTestServer, RequestHandler
from test.test_utils import clear_graphsignal_env, configure_test_watcher, shutdown_test_watcher

logger = logging.getLogger('graphsignal')


class VersionComparisonTest(unittest.TestCase):
    """What counts as worth telling somebody about."""

    def test_newer_minor_is_reported(self):
        self.assertEqual(newer_version('1.0.0', '1.1.0'), '1.1.0')

    def test_newer_major_is_reported(self):
        self.assertEqual(newer_version('1.9.0', '2.0.0'), '2.0.0')

    def test_newer_patch_is_not(self):
        # The rule the whole comparison exists for: a patch is not worth
        # interrupting a run for.
        self.assertIsNone(newer_version('1.0.0', '1.0.9'))

    def test_current_and_older_are_not(self):
        self.assertIsNone(newer_version('1.1.0', '1.1.0'))
        self.assertIsNone(newer_version('1.2.0', '1.1.0'))
        self.assertIsNone(newer_version('2.0.0', '1.9.0'))

    def test_unreadable_versions_report_nothing(self):
        # Includes the absent answer: the server says null when there is
        # nothing to report.
        self.assertIsNone(newer_version('1.0.0', None))
        self.assertIsNone(newer_version('1.0.0', 'latest'))
        self.assertIsNone(newer_version('nightly', '2.0.0'))

    def test_suffixed_releases_read_as_the_releases_they_are(self):
        self.assertEqual(newer_version('1.0.0rc1', '1.1.0'), '1.1.0')
        self.assertIsNone(newer_version('1.0.0+cu12', '1.0.5'))


class VersionCheckRequestTest(unittest.TestCase):
    def setUp(self):
        clear_graphsignal_env()
        # Class attributes on the shared handler; a leftover from another test
        # would decide this one's outcome.
        RequestHandler.request_path = None
        RequestHandler.response_code = 200
        RequestHandler.response_data = None

    def tearDown(self):
        clear_graphsignal_env()

    def _serve(self, latest_version=None, response_code=200, body=None):
        server = HttpTestServer()
        RequestHandler.response_code = response_code
        if body is not None:
            RequestHandler.response_data = body
        else:
            RequestHandler.response_data = json.dumps(
                {'latest_version': latest_version}).encode()
        server.start()
        server.wait_ready()
        return server

    def _check(self, api_base, stop_event=None):
        thread = start_version_check(api_base=api_base, stop_event=stop_event)
        if thread is not None:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        return thread

    def test_asks_the_version_check_endpoint_for_this_version(self):
        server = self._serve(latest_version=None)
        self._check(f'http://localhost:{server.get_port()}')

        path = server.get_request_path()
        self.assertTrue(path.startswith('/api/v1/version_check'), path)
        self.assertIn(f'version={__version__}', path)

    def test_disabled_makes_no_request_at_all(self):
        os.environ['GRAPHSIGNAL_DISABLE_VERSION_CHECK'] = '1'
        server = self._serve(latest_version='99.0.0')

        self.assertIsNone(self._check(f'http://localhost:{server.get_port()}'))
        self.assertIsNone(server.get_request_path())

    def test_a_newer_release_is_announced(self):
        server = self._serve(latest_version='99.0.0')

        with self.assertLogs('graphsignal', level='WARNING') as captured:
            self._check(f'http://localhost:{server.get_port()}')

        self.assertIn('99.0.0', '\n'.join(captured.output))

    def test_a_check_the_watcher_is_shutting_down_says_nothing(self):
        import threading
        server = self._serve(latest_version='99.0.0')
        stop_event = threading.Event()
        stop_event.set()

        with self.assertNoLogs('graphsignal', level='WARNING'):
            self._check(f'http://localhost:{server.get_port()}', stop_event=stop_event)


class VersionCheckSilenceTest(unittest.TestCase):
    """The half that is easier to regress: nothing that fails may be heard.

    A version check nobody asked for must never make a working run look broken —
    not on stderr, and not in the `errors` array of GET /signals.
    """

    def setUp(self):
        clear_graphsignal_env()
        RequestHandler.request_path = None
        RequestHandler.response_code = 200
        RequestHandler.response_data = None

    def tearDown(self):
        clear_graphsignal_env()

    def _run(self, api_base):
        thread = start_version_check(api_base=api_base)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_an_unreachable_endpoint_is_silent(self):
        from test.test_utils import free_port
        with self.assertNoLogs('graphsignal', level='WARNING'):
            self._run(f'http://127.0.0.1:{free_port()}')

    def test_a_server_error_is_silent(self):
        server = HttpTestServer()
        RequestHandler.response_code = 500
        RequestHandler.response_data = b'nope'
        server.start()
        server.wait_ready()

        with self.assertNoLogs('graphsignal', level='WARNING'):
            self._run(f'http://localhost:{server.get_port()}')

    def test_a_body_that_is_not_json_is_silent(self):
        server = HttpTestServer()
        RequestHandler.response_code = 200
        RequestHandler.response_data = b'<html>proxy sign-in page</html>'
        server.start()
        server.wait_ready()

        with self.assertNoLogs('graphsignal', level='WARNING'):
            self._run(f'http://localhost:{server.get_port()}')


class VersionCheckSignalsPayloadTest(unittest.TestCase):
    """The notice reaches GET /signals, and a failure does not.

    Pinned rather than left to chance: the `errors` array is what an AI agent
    reads, and a stale profiler is something it can act on.
    """

    def setUp(self):
        self.watcher = configure_test_watcher()
        # configure_test_watcher disables the check so no test phones home; the
        # cases below drive it explicitly against a local server.
        os.environ.pop('GRAPHSIGNAL_DISABLE_VERSION_CHECK', None)
        RequestHandler.request_path = None
        RequestHandler.response_code = 200
        RequestHandler.response_data = None

    def tearDown(self):
        shutdown_test_watcher()
        clear_graphsignal_env()

    def _run(self, api_base):
        thread = start_version_check(api_base=api_base)
        thread.join(timeout=5)

    def test_the_notice_is_in_the_errors_an_agent_reads(self):
        server = HttpTestServer()
        RequestHandler.response_data = json.dumps({'latest_version': '99.0.0'}).encode()
        server.start()
        server.wait_ready()

        self._run(f'http://localhost:{server.get_port()}')

        messages = [entry['message'] for entry in
                    self.watcher.log_store().last_entries(min_level='warning')]
        self.assertTrue(any('99.0.0' in message for message in messages), messages)

    def test_a_failed_check_leaves_the_errors_empty(self):
        from test.test_utils import free_port
        self._run(f'http://127.0.0.1:{free_port()}')

        self.assertEqual(self.watcher.log_store().last_entries(min_level='warning'), [])
