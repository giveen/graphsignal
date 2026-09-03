import os
import unittest

from graphsignal.watcher.env_vars import read_config_param, read_config_tags
from test.test_utils import clear_graphsignal_env


class EnvVarsTest(unittest.TestCase):
    def setUp(self):
        clear_graphsignal_env()

    def tearDown(self):
        clear_graphsignal_env()

    def test_provided_value_wins(self):
        self.assertEqual(read_config_param('arg1', str, 'val1', required=True),
                         'val1')
        self.assertEqual(read_config_param('arg2', int, 1, required=True), 1)

    def test_optional_missing_returns_none(self):
        self.assertIsNone(read_config_param('arg3', int, None, required=False))

    def test_default_value(self):
        self.assertEqual(
            read_config_param('arg3', int, None, default_value=7), 7)

    def test_env_value_parsed(self):
        os.environ['GRAPHSIGNAL_ARG4'] = '2'
        self.assertEqual(read_config_param('arg4', int, None), 2)

    def test_required_missing_raises(self):
        with self.assertRaises(ValueError):
            read_config_param('arg5', str, None, required=True)

    def test_required_from_env(self):
        os.environ['GRAPHSIGNAL_ARG6'] = '10'
        self.assertEqual(read_config_param('arg6', int, None, required=True), 10)

    def test_invalid_env_type_raises(self):
        os.environ['GRAPHSIGNAL_ARG7'] = 'str'
        with self.assertRaises(ValueError):
            read_config_param('arg7', int, None, required=True)

    def test_bool_parsing(self):
        os.environ['GRAPHSIGNAL_DEBUG'] = 'true'
        self.assertIs(read_config_param('debug', bool, None), True)
        os.environ['GRAPHSIGNAL_DEBUG'] = '0'
        self.assertIs(read_config_param('debug', bool, None), False)

    def test_list_parsing(self):
        os.environ['GRAPHSIGNAL_ARG9'] = 'v1,v2'
        self.assertEqual(read_config_param('arg9', list, None), ['v1', 'v2'])

    def test_provided_tags_win(self):
        os.environ['GRAPHSIGNAL_TAG_IGNORED'] = 'x'
        self.assertEqual(read_config_tags({'arg8': 'v1', 'arg9': '2.0'}),
                         {'arg8': 'v1', 'arg9': '2.0'})

    def test_tags_from_env(self):
        os.environ['GRAPHSIGNAL_TAG_ARG10'] = 'v1'
        os.environ['GRAPHSIGNAL_TAG_ARG11'] = '2.0'
        self.assertEqual(read_config_tags(None), {'arg10': 'v1', 'arg11': '2.0'})


if __name__ == '__main__':
    unittest.main()
