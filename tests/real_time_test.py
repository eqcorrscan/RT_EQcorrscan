"""
Tests for real-time matched-filtering.
"""

import unittest
import os
import shutil
import glob
import logging

from eqcorrscan import Tribe, Party
from eqcorrscan.utils import catalog_utils
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from rt_eqcorrscan.rt_match_filter import RealTimeTribe
from rt_eqcorrscan.streaming.clients.seedlink import RealTimeClient
from rt_eqcorrscan.reactor import get_inventory
from rt_eqcorrscan.plugins.lag_calc import LagCalcConfig
from rt_eqcorrscan.plugins.relocation import HypConfig


Logger = logging.getLogger(__name__)


class RealTimeTribeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detect_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), ".test_detections")
        client = Client('GEONET')
        cls.t1 = UTCDateTime(2016, 9, 4, 18)
        cls.t2 = UTCDateTime(2016, 9, 5)
        catalog = client.get_events(
            starttime=cls.t1, endtime=cls.t2, minmagnitude=4,
            minlatitude=-49, maxlatitude=-35,
            minlongitude=175.0, maxlongitude=180.0)
        catalog = catalog_utils.filter_picks(
            catalog, channels=['EHZ'], top_n_picks=2)
        cls.tribe = Tribe().construct(
            method='from_client', catalog=catalog, client_id='GEONET',
            lowcut=2.0, highcut=9.0, samp_rate=100.0, filt_order=4,
            length=3.0, prepick=0.15, swin='all', process_len=300)
        cls.client = "GEONET"
        cls.inventory = get_inventory(
            client=client, tribe=cls.tribe,
            location=dict(latitude=-42, longitude=177.5), starttime=cls.t1)

    def test_init_with_tribe(self):
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=1200)
        rt_tribe = RealTimeTribe(tribe=self.tribe, rt_client=rt_client)
        self.assertEqual(rt_tribe.templates, self.tribe.templates)
        self.assertEqual(rt_tribe.rt_client.server_hostname,
                         "link.geonet.org.nz")

    def test_buffer_too_short(self):
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz",  buffer_capacity=20)
        with self.assertRaises(AssertionError):
            RealTimeTribe(tribe=self.tribe, rt_client=rt_client)

    def test_run(self):
        tribe = self.tribe.copy()
        for template in tribe:
            template.process_length = 60
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=90,)
        rt_tribe = RealTimeTribe(
            tribe=tribe, rt_client=rt_client, detect_interval=5, plot=False,
            name="test_run_tribe")
        party = rt_tribe.run(
            threshold=6, threshold_type="MAD", trig_int=3, max_run_length=100,
            detect_directory=self.detect_dir)
        if os.path.isdir(rt_tribe.running_template_dir):
            shutil.rmtree(rt_tribe.running_template_dir)
        self.assertTrue(isinstance(party, Party))
        Logger.critical("RESTARTING")
        # Test re-starting
        party += rt_tribe.run(
            threshold=6, threshold_type="MAD", trig_int=3,
            max_run_length=20, detect_directory=self.detect_dir)
        if os.path.isdir(rt_tribe.running_template_dir):
            shutil.rmtree(rt_tribe.running_template_dir)

    def test_station_overlap(self):
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=1200)
        rt_tribe = RealTimeTribe(
            tribe=self.tribe, rt_client=rt_client, inventory=self.inventory,
            name="test_station_overlap_tribe")
        self.assertGreaterEqual(
            len(rt_tribe.used_stations), len(self.inventory))
        if os.path.isdir(rt_tribe.running_template_dir):
            shutil.rmtree(rt_tribe.running_template_dir)

    def test_run_zero_threshold(self):
        """ Test to ensure some detections are made an handled correctly."""
        tribe = self.tribe.copy()
        for template in tribe:
            template.process_length = 60
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=90)
        rt_tribe = RealTimeTribe(
            tribe=tribe, rt_client=rt_client, detect_interval=5, plot=False,
            name="test_run_zero_threshold_tribe")
        party = rt_tribe.run(
            threshold=0.9, threshold_type="MAD", trig_int=3,
            max_run_length=100, detect_directory=self.detect_dir)
        self.assertTrue(isinstance(party, Party))
        self.assertTrue(os.path.isdir(self.detect_dir))
        detect_files = glob.glob(os.path.join(
            self.detect_dir, "????", "???", "*.xml"))
        self.assertGreater(len(detect_files), 0)
        if os.path.isdir(rt_tribe.running_template_dir):
            shutil.rmtree(rt_tribe.running_template_dir)

    def test_run_lag_calc(self):
        tribe = self.tribe.copy()
        for template in tribe:
            template.process_length = 60
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=90,)

        rt_tribe = RealTimeTribe(
            tribe=tribe, rt_client=rt_client, detect_interval=5, plot=False,
            name="test_run_tribe", plugin_config={'lag_calc': LagCalcConfig(
                sleep_interval=20
            )})
        party = rt_tribe.run(
            threshold=1, threshold_type="MAD", trig_int=3, max_run_length=600,
            detect_directory="test_run_tribe/detections")
        if os.path.isdir(rt_tribe.running_template_dir):
            shutil.rmtree(rt_tribe.running_template_dir)
        self.assertTrue(isinstance(party, Party))
        # TODO: This doesn't actually test that lag-calc was running properly

    def test_run_hyp(self):
        tribe = self.tribe.copy()
        for template in tribe:
            template.process_length = 60
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=90,)

        self.inventory.write("test_inv.xml", format="STATIONXML")

        rt_tribe = RealTimeTribe(
            tribe=tribe, rt_client=rt_client, detect_interval=5, plot=False,
            name="test_run_tribe", plugin_config={
                'lag_calc': LagCalcConfig(sleep_interval=20),
                'hyp': HypConfig(sleep_interval=20,
                                 station_file="test_inv.xml")
            })
        party = rt_tribe.run(
            threshold=1, threshold_type="MAD", trig_int=3, max_run_length=600,
            detect_directory="test_run_tribe/detections")
        if os.path.isdir(rt_tribe.running_template_dir):
            shutil.rmtree(rt_tribe.running_template_dir)
        self.assertTrue(isinstance(party, Party))

    # No tear downs with parallel testing :(
    # @classmethod
    # def tearDownClass(cls) -> None:
    #     if os.path.isdir(cls.detect_dir):
    #         shutil.rmtree(cls.detect_dir)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level="DEBUG")
    unittest.main()
