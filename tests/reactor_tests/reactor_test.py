"""
Tests for the reactor module of rt_eqcorrscan.
"""
import os
import unittest
import shutil
import time

from functools import partial

from obspy import UTCDateTime, Catalog
from obspy.core.event import Event, Origin, Magnitude
from obspy.geodetics import kilometer2degrees
from obspy.clients.fdsn import Client

from eqcorrscan.core.match_filter import read_tribe

from rt_eqcorrscan.reactor import get_inventory, estimate_region, Reactor
from rt_eqcorrscan.event_trigger import CatalogListener
from rt_eqcorrscan.database import TemplateBank
from rt_eqcorrscan.event_trigger import magnitude_rate_trigger_func


class ReactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.testing_path = os.path.abspath(
            os.path.dirname(__file__)) + os.path.sep + "db"
        cls.template_bank = TemplateBank(cls.testing_path)
        cls.listener = CatalogListener(
            client=Client("GEONET"), catalog=Catalog(),
            catalog_lookup_kwargs=dict(
                latitude=-45., longitude=178., maxradius=3.0),
            template_bank=cls.template_bank, interval=5)
        cls.trigger_func = partial(
            magnitude_rate_trigger_func, magnitude_threshold=4,
            rate_threshold=20, rate_bin=0.5)

    def test_up_time(self):
        reactor = Reactor(
            client=Client("GEONET"),
            seedlink_server_url="link.geonet.org.nz",
            listener=self.listener, trigger_func=self.trigger_func,
            template_database=self.template_bank,
            real_time_tribe_kwargs=dict(),
            plot_kwargs=dict(),
            listener_kwargs=dict(make_templates=False))
        self.assertEqual(reactor.up_time, 0)
        reactor._run_start = UTCDateTime(2000, 1, 1)
        reactor.up_time = UTCDateTime(2000, 1, 2)
        self.assertEqual(reactor.up_time, 86400)

    def test_run(self):
        reactor = Reactor(
            client=Client("GEONET"),
            seedlink_server_url="link.geonet.org.nz",
            listener=self.listener, trigger_func=self.trigger_func,
            template_database=self.template_bank,
            real_time_tribe_kwargs=dict(),
            plot_kwargs=dict(),
            listener_kwargs=dict(make_templates=False))
        reactor.run(max_run_length=30)
        self.assertGreaterEqual(reactor.up_time, 30)

    def test_reactor_spin_up(self):
        reactor = Reactor(
            client=Client("GEONET"),
            seedlink_server_url="link.geonet.org.nz",
            listener=self.listener, trigger_func=self.trigger_func,
            template_database=self.template_bank,
            real_time_tribe_kwargs=dict(),
            plot_kwargs=dict(),
            listener_kwargs=dict(make_templates=False))
        trigger_event = Event(
            origins=[Origin(
                time=UTCDateTime(2019, 1, 1), latitude=-45.,
                longitude=178.0, depth=10000.)],
            magnitudes=[Magnitude(mag=7.4)])
        reactor.background_spin_up(triggering_event=trigger_event)
        time.sleep(10)
        reactor.stop()

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(cls.testing_path):
            shutil.rmtree(cls.testing_path)


class GetInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = Client("GEONET")
        cls.tribe = read_tribe(os.path.join(
            os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
            "test_data", "test_tribe.tgz"))
        cls.trigger_event = Event(
            origins=[Origin(
                time=UTCDateTime(2019, 1, 1), latitude=-45.,
                longitude=178.0, depth=10000.)],
            magnitudes=[Magnitude(mag=7.4)])

    def test_get_inventory_no_trigger(self):
        inventory = get_inventory(
            client=self.client, tribe=self.tribe,
            location=dict(
                latitude=self.trigger_event.origins[0].latitude,
                longitude=self.trigger_event.origins[0].longitude),
            starttime=self.trigger_event.origins[0].time)
        self.assertEqual(len(inventory), 10)

    def test_get_inventory_using_event(self):
        inventory = get_inventory(
            client=self.client, tribe=self.tribe,
            triggering_event=self.trigger_event)
        self.assertEqual(len(inventory), 10)

    def test_no_origin(self):
        event = self.trigger_event.copy()
        event.preferred_origin_id = None
        event.origins = []
        inventory = get_inventory(
            client=self.client, tribe=self.tribe,
            triggering_event=event)
        self.assertEqual(len(inventory), 0)

    def test_small_region(self):
        inventory = get_inventory(
            client=self.client, tribe=self.tribe,
            triggering_event=self.trigger_event, max_distance=50.)
        self.assertLess(len(inventory), 10)


class RegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.event = Event(
            origins=[Origin(
                time=UTCDateTime(2019, 1, 1), latitude=-45.,
                longitude=90.0, depth=10000.)],
            magnitudes=[Magnitude(mag=7.4)]
        )
        cls.event.preferred_origin_id = cls.event.origins[0].resource_id

    def test_good_estimate_region(self):
        region = estimate_region(self.event, min_length=50.)
        self.assertEqual(
            self.event.preferred_origin().latitude, region["latitude"])
        self.assertEqual(
            self.event.preferred_origin().longitude, region["longitude"])
        self.assertGreater(
            region["maxradius"], kilometer2degrees(50) / 2)

    def test_estimate_region_no_origin(self):
        event = self.event.copy()
        event.preferred_origin_id = None
        event.origins = []
        self.assertIsNone(estimate_region(event))

    def test_estimate_region_no_magnitude(self):
        event = self.event.copy()
        event.magnitudes = []
        region = estimate_region(event)
        self.assertEqual(
            self.event.preferred_origin().latitude, region["latitude"])
        self.assertEqual(
            self.event.preferred_origin().longitude, region["longitude"])
        self.assertEqual(
            region["maxradius"], kilometer2degrees(50) / 2)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level="DEBUG")

    unittest.main()
