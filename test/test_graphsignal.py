import unittest


class GraphsignalPackageTest(unittest.TestCase):
    def test_exposes_version_only(self):
        import graphsignal
        self.assertTrue(hasattr(graphsignal, '__version__'))
        self.assertIsInstance(graphsignal.__version__, str)
        self.assertFalse(hasattr(graphsignal, 'watch'))
        self.assertEqual(graphsignal.__all__, ['__version__'])

    def test_import_is_side_effect_free(self):
        import graphsignal  # noqa: F401
        import graphsignal.watcher
        self.assertFalse(graphsignal.watcher.is_configured())


if __name__ == '__main__':
    unittest.main()
