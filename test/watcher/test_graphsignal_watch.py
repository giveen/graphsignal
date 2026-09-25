import argparse
import unittest

from graphsignal.commands.graphsignal_watch import _port


class PortArgumentTest(unittest.TestCase):
    def test_accepts_valid_ports(self):
        self.assertEqual(_port('1'), 1)
        self.assertEqual(_port('18259'), 18259)
        self.assertEqual(_port('65535'), 65535)

    def test_rejects_invalid_ports(self):
        for value in ('abc', '0', '-1', '65536'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                _port(value)


if __name__ == '__main__':
    unittest.main()
