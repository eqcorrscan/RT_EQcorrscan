"""
Tests for the event listener.
"""

import unittest
import time
import os
import shutil
import copy

from obspy import Catalog, UTCDateTime
from obspy.core.event import Event, Pick, Origin
from obspy.clients.fdsn import Client

from rt_eqcorrscan.event_trigger.catalog_listener import (
    CatalogListener, filter_events)
from rt_eqcorrscan.event_trigger.listener import event_time
from rt_eqcorrscan.database import TemplateBank


class ListeningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "listener_db")
        if not os.path.isdir(cls.test_path):
            os.makedirs(cls.test_path)

    def test_listener(self):
        listener = CatalogListener(
            client=Client("GEONET"), catalog=Catalog(),
            interval=20, template_bank=TemplateBank(base_path=self.test_path),
            catalog_lookup_kwargs=dict(
                latitude=-38.8, longitude=175.8, maxradius=3.), keep=600,
        )
        # Use a period of known seismicity: 2019-07-06, swarm under Taupo
        seconds_before_now = UTCDateTime.now() - UTCDateTime(2019, 7, 6, 9, 12)
        listener._test_start_step = seconds_before_now
        listener.background_run(
            filter_func=filter_events, auto_pick=False, make_templates=False,
            template_kwargs=dict(lowcut=2., highcut=15., samp_rate=50.,
                                 filt_order=4, prepick=0.5, length=3,
                                 swin="P"))
        self.assertTrue(listener.busy)
        time.sleep(120)
        listener.background_stop()
        self.assertFalse(listener.busy)
        self.assertEqual(len(listener.old_events), 1)

    def test_expire_old_events(self):
        old_cat = Client("GEONET").get_events(
            latitude=-38.8, longitude=175.8, maxradius=3.,
            starttime=UTCDateTime(2019, 7, 6, 9, 12),
            endtime=UTCDateTime(2019, 7, 6, 12))
        listener = CatalogListener(
            client=Client("GEONET"), catalog=old_cat,
            interval=20, template_bank=TemplateBank(base_path=self.test_path),
            catalog_lookup_kwargs=dict(
                latitude=-38.8, longitude=175.8, maxradius=3.), keep=600,
        )
        self.assertEqual(len(old_cat), len(listener.old_events))
        original_old_events = copy.deepcopy(listener.old_events)
        endtime = UTCDateTime(2019, 7, 6, 10)
        listener._remove_old_events(endtime=endtime)
        for _, _event_time in listener.old_events:
            self.assertGreater(_event_time, endtime - listener.keep)
        for event in original_old_events:
            if event not in listener.old_events:
                self.assertLess(event[1], endtime - listener.keep)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(cls.test_path):
            shutil.rmtree(cls.test_path)


class StaticFilterTests(unittest.TestCase):
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
        cat = filter_events(
            self.catalog, min_stations=None, auto_event=True,
            auto_picks=True, event_type="explosion")
        self.assertEqual(len(cat), 0)


class EventTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.event = Event(
            origins=[Origin(time=UTCDateTime(2001, 3, 26))],
            picks=[Pick(time=UTCDateTime(2001, 3, 26, 1, 1, 1)),
                   Pick(time=UTCDateTime(2001, 3, 26, 1, 1, 5))])
        cls.event.preferred_origin_id = cls.event.origins[0].resource_id

    def test_extract_origin_time(self):
        self.assertEqual(self.event.origins[0].time, event_time(self.event))

    def test_extract_pick_time(self):
        no_origin = self.event.copy()
        no_origin.origins = []
        no_origin.preferred_origin_id = None
        self.assertEqual(self.event.picks[0].time, event_time(no_origin))

    def test_extract_zero_time(self):
        self.assertEqual(UTCDateTime(0), event_time(Event()))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level="DEBUG")
    unittest.main()
