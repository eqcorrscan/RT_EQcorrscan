"""
Test for the configuration of RT_EQcorrscan
"""

import unittest
import os

from obspy.clients.fdsn import Client

from rt_eqcorrscan.config.config import Config, read_config, PlotConfig


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
        self.assertIsInstance(config.streaming.rt_client_url, str)
        client = config.rt_match_filter.get_client()
        self.assertTrue(hasattr(client, "get_events"))

        self.assertIsInstance(config.reactor.magnitude_threshold, float)

        self.assertIsInstance(config.plot.lowcut, float)

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

    def test_logging_setup(self):
        """ Just check that no error is raised """
        config = Config()
        config.setup_logging()

    def test_get_client(self):
        config = read_config(
            os.path.join(self.test_path, "default_config.yml"))
        client = config.rt_match_filter.get_client()
        self.assertIsInstance(client, Client)
        config.rt_match_filter.client = "WaLrOuS"
        client = config.rt_match_filter.get_client()
        self.assertEqual(client, None)
        config.rt_match_filter.client_type = "aLbATrOsS"
        client = config.rt_match_filter.get_client()
        self.assertEqual(client, None)

    def test_get_waveform_client(self):
        config = read_config(
            os.path.join(self.test_path, "default_config.yml"))
        client = config.rt_match_filter.get_waveform_client()
        self.assertEqual(client, None)
        config.rt_match_filter.waveform_client_type = "FDSN"
        client = config.rt_match_filter.get_waveform_client()
        self.assertEqual(client, None)
        config.rt_match_filter.waveform_client = "GEONET"
        client = config.rt_match_filter.get_waveform_client()
        self.assertIsInstance(client, Client)

    def test_get_streaming_client(self):
        config = Config()
        rt_client = config.streaming.get_streaming_client()
        self.assertEqual(
            rt_client.server_url, config.streaming.rt_client_url)

    def test_bad_init(self):
        with self.assertRaises(NotImplementedError):
            Config(wilf="bob")

    def test_init_with_dict(self):
        config = Config(plot={"walrous": True})
        self.assertEqual(config.plot.walrous, True)
        self.assertEqual(config.plot.plot_length, 600.)

    def test_init_with_object(self):
        plot = PlotConfig(walrous=True, plot_length=500.)
        config = Config(plot=plot)
        self.assertEqual(config.plot.walrous, True)
        self.assertEqual(config.plot.plot_length, 500.)

    def test_equality(self):
        config = Config()
        self.assertEqual(config, config)
        self.assertNotEqual(config, "walrous")
        altered_config = Config(plot={"plot_length": 500.})
        self.assertNotEqual(config, altered_config)

        plot = PlotConfig()
        plot_extras = PlotConfig(animal="albatross")
        self.assertNotEqual(plot, plot_extras)

    @classmethod
    def tearDownClass(cls):
        for f in cls.files_to_remove:
            if os.path.isfile(f):
                os.remove(f)


if __name__ == "__main__":
    unittest.main()
