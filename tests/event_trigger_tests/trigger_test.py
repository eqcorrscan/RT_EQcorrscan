"""
Tests for the standard trigger functions.
"""

import unittest

from obspy import UTCDateTime, Catalog
from obspy.clients.fdsn import Client
from eqcorrscan import Detection

from rt_eqcorrscan.event_trigger import (
    magnitude_rate_trigger_func, average_rate)


class StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client = Client("GEONET")
        # Central Southern Alps
        cls.quiet_catalog = client.get_events(
            starttime=UTCDateTime(2019, 6, 1),
            endtime=UTCDateTime(2019, 7, 1),
            latitude=-43.5, longitude=169.4, maxradius=0.5)
        # Taupo swarm
        cls.high_rate_catalog = client.get_events(
            starttime=UTCDateTime(2019, 6, 20),
            endtime=UTCDateTime(2019, 6, 21),
            latitude=-38.8, longitude=175.8, maxradius=0.2)
        # AF/Puysegur junction mainshock: mainshock eventid=2019p428761
        cls.mainshock_catalog = client.get_events(
            starttime=UTCDateTime(2019, 6, 7),
            endtime=UTCDateTime(2019, 6, 10),
            latitude=-44.3, longitude=168.1, maxradius=0.4)

    def test_no_trigger(self):
        self.assertTrue(len(self.quiet_catalog) > 0)
        triggered_events = magnitude_rate_trigger_func(self.quiet_catalog)
        self.assertTrue(len(triggered_events) == 0)

    def test_magnitude_trigger(self):
        mainshock = sorted(
            self.mainshock_catalog.events,
            key=lambda e: e.preferred_magnitude().mag, reverse=True)[0]
        triggered_events = magnitude_rate_trigger_func(self.mainshock_catalog)
        self.assertTrue(mainshock in triggered_events)

    def test_rate_trigger(self):
        triggered_events = magnitude_rate_trigger_func(self.high_rate_catalog)
        self.assertTrue(len(triggered_events) > 1)

    def test_missing_magnitudes(self):
        missing_mag_cat = self.high_rate_catalog.copy()
        missing_mag_cat[2].magnitudes = []
        missing_mag_cat[2].preferred_magnitude_id = None
        triggered_events = magnitude_rate_trigger_func(missing_mag_cat)
        self.assertTrue(len(triggered_events) > 1)

    def test_missing_origins(self):
        missing_origin_cat = self.mainshock_catalog.copy()
        missing_origin_cat[2].origins = []
        missing_origin_cat[2].preferred_origin_id = None
        triggered_events = magnitude_rate_trigger_func(missing_origin_cat)
        self.assertTrue(len(triggered_events) > 1)

    def test_rate_calculation_no_events(self):
        self.assertEqual(average_rate(Catalog()), 0.)

    def test_rate_for_list_of_events(self):
        dets = [Detection(
            detect_time=UTCDateTime() + (i * 100),
            template_name="wilf", no_chans=10, detect_val=15,
            threshold=3, threshold_type="MAD", threshold_input=5,
            typeofdet="correlation")
                for i in range(10)]
        rate = average_rate(dets)
        self.assertAlmostEqual(rate, 960, 1)


if __name__ == "__main__":
    unittest.main()
