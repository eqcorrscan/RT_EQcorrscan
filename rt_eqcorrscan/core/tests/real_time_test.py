"""
Tests for real-time matched-filtering.
"""

import unittest

from eqcorrscan import Tribe, Party
from eqcorrscan.utils import catalog_utils
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from rt_eqcorrscan.core.rt_match_filter import RealTimeTribe


class RealTimeTribeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client = Client('GEONET')
        cls.t1 = UTCDateTime(2016, 9, 4)
        cls.t2 = cls.t1 + 86400
        catalog = client.get_events(
            starttime=cls.t1, endtime=cls.t2, minmagnitude=4,
            minlatitude=-49, maxlatitude=-35,
            minlongitude=175.0, maxlongitude=180.0)
        catalog = catalog_utils.filter_picks(
            catalog, channels=['EHZ'], top_n_picks=5)
        cls.tribe = Tribe().construct(
            method='from_client', catalog=catalog, client_id='GEONET',
            lowcut=2.0, highcut=9.0, samp_rate=50.0, filt_order=4,
            length=3.0, prepick=0.15, swin='all', process_len=120)
        cls.client = "GEONET"

    def test_init_with_tribe(self):
        rt_tribe = RealTimeTribe(
            tribe=self.tribe, server_url="link.geonet.org.nz",
            buffer_capacity=1200)
        self.assertEqual(rt_tribe.templates, self.tribe.templates)
        self.assertEqual(rt_tribe.client.server_hostname,
                         "link.geonet.org.nz")

    def test_buffer_too_short(self):
        with self.assertRaises(AssertionError):
            RealTimeTribe(tribe=self.tribe, server_url="link.geonet.org.nz",
                          buffer_capacity=20)

    def test_run(self):
        rt_tribe = RealTimeTribe(
            tribe=self.tribe, server_url="link.geonet.org.nz",
            buffer_capacity=120)
        party = rt_tribe.run(threshold=8, threshold_type="MAD", trig_int=3,
                             max_run_length=200)
        self.assertTrue(isinstance(party, Party))


if __name__ == "__main__":
    unittest.main()
