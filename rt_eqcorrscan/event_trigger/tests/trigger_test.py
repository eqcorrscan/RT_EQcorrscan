"""
Tests for the standard trigger functions.
"""

import unittest

from obspy.clients.fdsn import Client

from rt_eqcorrscan.event_trigger.triggers import (
    magnitude_rate_trigger_func)


class StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client = Client("GEONET")
        cls.quiet_catalog = client.get_events()
        cls.high_rate_catalog = client.get_events()
        cls.mainshock_catalog = client.get_events()

    def test_no_trigger(self):
        triggered_events = magnitude_rate_trigger_func(self.quiet_catalog)
        self.assertEqual(triggered_events, [])

    def test_magnitude_trigger(self):
        mainshock = sorted(self.mainshock_catalog.event,
                           key=lambda e: e.preferred_magnitude().mag)[0]
        triggered_events = magnitude_rate_trigger_func(self.mainshock_catalog)
        self.assertTrue(mainshock in triggered_events)

    def test_rate_trigger(self):
        triggered_events = magnitude_rate_trigger_func(self.high_rate_catalog)
        self.assertTrue(len(triggered_events) > 1)


if __name__ == "__main__":
    unittest.main()
