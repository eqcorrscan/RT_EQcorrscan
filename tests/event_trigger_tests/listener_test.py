"""
Tests for the event listener.
"""

import unittest
import time
import os

from obspy import Catalog, UTCDateTime
from obspy.clients.fdsn import Client

from rt_eqcorrscan.event_trigger.catalog_listener import (
    CatalogListener, filter_events)
from rt_eqcorrscan.database import TemplateBank


class ListeningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_path = os.path.abspath(os.path.dirname(__file__))

    def test_listener(self):
        listener = CatalogListener(
            client=Client("GEONET"), catalog=Catalog(),
            interval=20, template_bank=TemplateBank(base_path=self.test_path),
            catalog_lookup_kwargs=dict(
                latitude=-42.0, longitude=175.2, maxradius=3.), keep=600,
        )
        listener.background_run()
        self.assertTrue(listener.busy)
        time.sleep(10)
        listener.background_stop()
        self.assertFalse(listener.busy)


class StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        client = Client("GEONET")
        cls.catalog = client.get_events(
            starttime=UTCDateTime(2019, 6, 21),
            endtime=UTCDateTime(2019, 6, 22),
            latitude=-38.8, longitude=175.8, maxradius=0.2)

    def test_filter_events(self):
        cat = filter_events(
            self.catalog, min_stations=5, auto_event=False,
            auto_picks=False, event_type=["earthquake"])
        for ev in cat:
            auto_picks = [p for p in ev.picks
                          if p.evaluation_mode == 'automatic']
            self.assertEqual(len(auto_picks), 0)

    def test_remove_events(self):
        cat = filter_events(self.catalog, event_type="explosion")
        self.assertEqual(len(cat), 0)


if __name__ == "__main__":
    unittest.main()
