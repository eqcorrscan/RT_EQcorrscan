"""
Tests for the event listener.
"""

import unittest
import time

from obspy import Catalog
from obspy.clients.fdsn import Client

from rt_eqcorrscan.utils.event_trigger.listener import (
    trigger_func, CatalogListener)


class StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client = Client("GEONET")
        cls.quiet_catalog = client.get_events()
        cls.high_rate_catalog = client.get_events()
        cls.mainshock_catalog = client.get_events()

    def test_no_trigger(self):
        triggered_events = trigger_func(self.quiet_catalog)
        self.assertEqual(triggered_events, [])

    def test_magnitude_trigger(self):
        mainshock = sorted(self.mainshock_catalog.event,
                           key=lambda e: e.preferred_magnitude().mag)[0]
        triggered_events = trigger_func(self.mainshock_catalog)
        self.assertTrue(mainshock in triggered_events)

    def test_rate_trigger(self):
        triggered_events = trigger_func(self.high_rate_catalog)
        self.assertTrue(len(triggered_events) > 1)


class ListeningTest(unittest.TestCase):
    def test_listener(self):
        Listener = CatalogListener(
            client=Client("GEONET"), catalog=Catalog(),
            callback_func=dummy_callback, interval=20,
            latitude=-42.0, longitude=175.2, maxradius=3.)
        Listener.background_run()
        self.assertTrue(Listener.busy)
        time.sleep(10)
        Listener.background_stop()
        self.assertFalse(Listener.busy)


def dummy_callback(event):
    """
    Dummy callback function that just prints the event
    """
    print(event)


if __name__ == "__main__":
    unittest.main()
