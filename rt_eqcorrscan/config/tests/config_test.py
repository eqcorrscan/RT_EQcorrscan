"""
Test for the configuration of RT_EQcorrscan
"""

import unittest
import os

from rt_eqcorrscan.config.config import Config, read_config


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files_to_remove = []
        cls.test_path = os.path.abspath(os.path.dirname(__file__))

    def test_default_config(self):
        config = Config()
        # Check that we can get attributes
        self.assertIsInstance(config.rt_match_filter.n_stations, int)
        self.assertIsInstance(config.rt_match_filter.plot, bool)
        self.assertIsInstance(config.rt_match_filter.seedlink_server_url, str)
        client = config.rt_match_filter.get_client()
        self.assertTrue(hasattr(client, "get_events"))

        self.assertIsInstance(config.reactor.magnitude_threshold, float)

        self.assertIsInstance(config.plot.low_cut, float)

        self.assertIsInstance(config.database_manager.event_path, str)

    def test_read_config(self):
        config = read_config(
            os.path.join(self.test_path, "default_config.yml"))
        self.assertIsInstance(config, Config)

    def test_read_no_file(self):
        config = read_config()
        self.assertIsInstance(config, Config)

    def test_fail_read(self):
        with self.assertRaises(FileNotFoundError):
            read_config("missing_file")

    def test_round_trip(self):
        config = Config()
        tmp_file = "tmp_config.yml"
        config.write(tmp_file)
        self.files_to_remove.append(tmp_file)

        config_back = read_config(tmp_file)
        self.assertEqual(config, config_back)

    @classmethod
    def tearDownClass(cls):
        for f in cls.files_to_remove:
            if os.path.isfile(f):
                os.remove(f)


if __name__ == "__main__":
    unittest.main()
