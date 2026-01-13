"""
Tests for the reactor module of rt_eqcorrscan.
"""
import os
import unittest
import shutil
import time
import copy
import logging
import warnings
from copy import deepcopy

from functools import partial

from obspy import UTCDateTime, Catalog
from obspy.core.event import Event, Origin, Magnitude, ResourceIdentifier
from obspy.geodetics import kilometer2degrees
from obspy.clients.fdsn import Client

from eqcorrscan.core.match_filter import read_tribe

from rt_eqcorrscan.config import Config
from rt_eqcorrscan.reactor import get_inventory, estimate_region, Reactor
from rt_eqcorrscan.event_trigger import CatalogListener
from rt_eqcorrscan.database.database_manager import (
    TemplateBank, remove_unreferenced)
from rt_eqcorrscan.event_trigger import magnitude_rate_trigger_func
from eqcorrscan.utils.catalog_utils import filter_picks


class ReactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.testing_path = os.path.abspath(
            os.path.dirname(__file__)) + os.path.sep + "db"

    def setUp(self) -> None:
        if not os.path.isdir(self.testing_path):
            os.makedirs(self.testing_path)
        self.template_bank = TemplateBank(self.testing_path)
        self.listener = CatalogListener(
            client=Client("GEONET"), catalog=Catalog(),
            catalog_lookup_kwargs=dict(
                latitude=-45., longitude=178., maxradius=3.0),
            template_bank=self.template_bank, interval=5)
        self.trigger_func = partial(
            magnitude_rate_trigger_func, magnitude_threshold=4,
            rate_threshold=20, rate_bin=0.5)
        config = Config()
        config.rt_match_filter.rt_client_url = "link.geonet.org.nz"
        config.rt_match_filter.rt_client_type = "seedlink"
        config.rt_match_filter.threshold = 8
        config.rt_match_filter.threshold_type = "MAD"
        config.rt_match_filter.trig_int = 2
        config.rt_match_filter.plot = False
        self.config = config

    def test_up_time(self):
        reactor = Reactor(
            client=Client("GEONET"),
            listener=self.listener, trigger_func=self.trigger_func,
            template_database=self.template_bank,
            config=self.config)
        self.assertEqual(reactor.up_time, 0)
        reactor._run_start = UTCDateTime(2000, 1, 1)
        reactor.up_time = UTCDateTime(2000, 1, 2)
        self.assertEqual(reactor.up_time, 86400)

    def test_run(self):
        reactor = Reactor(
            client=Client("GEONET"),
            listener=self.listener, trigger_func=self.trigger_func,
            template_database=self.template_bank, config=self.config)
        with self.assertRaises(SystemExit):
            reactor.run(max_run_length=30)
        self.assertGreaterEqual(reactor.up_time, 30)

    def test_reactor_spin_up(self):
        reactor = Reactor(
            client=Client("GEONET"),
            listener=self.listener, trigger_func=self.trigger_func,
            template_database=self.template_bank,
            config=self.config)
        trigger_event = Event(
            origins=[Origin(
                time=UTCDateTime(2019, 1, 1), latitude=-45.,
                longitude=178.0, depth=10000.)],
            magnitudes=[Magnitude(mag=7.4)])
        lookup_starttime = reactor._trigger_region(trigger_event)['starttime']
        self.assertEqual(UTCDateTime(0), lookup_starttime)
        reactor.spin_up(triggering_event=trigger_event)
        time.sleep(10)
        with self.assertRaises(SystemExit):
            reactor.stop()

    def test_reactor_spin_up_starttime_from_trigger(self):
        config = deepcopy(self.config)
        config.database_manager.lookup_starttime = 2 * 86400
        reactor = Reactor(
            client=Client("GEONET"),
            listener=self.listener, trigger_func=self.trigger_func,
            template_database=self.template_bank,
            config=config)
        trigger_event = Event(
            origins=[Origin(
                time=UTCDateTime(2019, 1, 1), latitude=-45.,
                longitude=178.0, depth=10000.)],
            magnitudes=[Magnitude(mag=7.4)])
        lookup_starttime = reactor._trigger_region(trigger_event)['starttime']
        self.assertEqual(UTCDateTime(2019, 1, 1) - (2 * 86400),
                         lookup_starttime)


class IncreasingMagnitude(Magnitude):
    def __init__(self, magnitude: Magnitude, mag_step: float = 0.1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._previous_mag = magnitude.mag - mag_step
            self.mag_step = mag_step
        super().__init__(**magnitude.__dict__)
        self.resource_id = ResourceIdentifier(
            id=f"{self.resource_id.id}_increasing")

    @property
    def mag(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._previous_mag += self.mag_step
        return self._previous_mag


class IncreasingMagnitudeClient:
    """
    Hack of a "client" that returns the same event but with an increasing
    magnitude for every query.
    """
    def __init__(self, event: Event):
        self.event = event.copy()
        self.event.magnitudes = [IncreasingMagnitude(
            self.event.preferred_magnitude())]
        self.event.preferred_magnitude_id = self.event.magnitudes[0].resource_id

    def get_events(self, *args, **kwargs):
        """ We don't care about args. """
        return Catalog([self.event])


class ExpandingRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.testing_path = os.path.abspath(
            os.path.dirname(__file__)) + os.path.sep + "expanding_db"
        if os.path.isdir(cls.testing_path):
            shutil.rmtree(cls.testing_path)
        os.makedirs(cls.testing_path)
        logging.debug("Making the bank")
        cls.bank = TemplateBank(
            base_path=cls.testing_path, name_structure="{event_id_short}")
        cls.client = Client("GEONET")
        logging.debug("Downloading the catalog")
        catalog = cls.client.get_events(
            starttime=UTCDateTime(2019, 6, 21),
            endtime=UTCDateTime(2019, 6, 23),
            latitude=-38.8, longitude=175.8, maxradius=0.5)
        cls.catalog = remove_unreferenced(
            filter_picks(catalog=catalog, top_n_picks=5))
        cls.catalog.events.sort(
            key=lambda ev: ev.preferred_magnitude().mag)
        cls.bank.put_events(cls.catalog)
        logging.debug("Making templates")
        _ = cls.bank.make_templates(
            catalog=cls.catalog, client=cls.client, lowcut=2., highcut=15.,
            samp_rate=50., filt_order=4, prepick=0.5, length=3, swin="all")

    def setUp(self) -> None:
        self.trigger_event = IncreasingMagnitudeClient(self.catalog[-1])
        self.listener = CatalogListener(
            client=self.trigger_event, catalog=Catalog(),
            catalog_lookup_kwargs=dict(
                latitude=-45., longitude=178., maxradius=3.0),
            template_bank=self.bank, interval=5)
        self.trigger_func = partial(
            magnitude_rate_trigger_func, magnitude_threshold=2.9,
            rate_threshold=20, rate_bin=0.5)
        config = Config()
        config.rt_match_filter.rt_client_url = "link.geonet.org.nz"
        config.rt_match_filter.rt_client_type = "seedlink"
        config.rt_match_filter.threshold = 8
        config.rt_match_filter.threshold_type = "MAD"
        config.rt_match_filter.trig_int = 2
        config.rt_match_filter.plot = False
        config.reactor.minimum_lookup_radius = 0
        self.config = config

    def test_increasing_region(self):
        reactor = Reactor(
            client=Client("GEONET"),
            listener=self.listener, trigger_func=self.trigger_func,
            template_database=self.bank, config=self.config)
        trigger_1 = self.trigger_event.get_events()[0]
        region_1 = reactor._trigger_region(
            triggering_event=trigger_1)
        trigger_2 = self.trigger_event.get_events()[0]
        region_2 = reactor._trigger_region(
            triggering_event=trigger_2)
        self.assertGreater(trigger_2.preferred_magnitude().mag,
                           trigger_1.preferred_magnitude().mag)
        self.assertGreater(region_2["maxradius"], region_1["maxradius"])
        self.assertEqual(region_2["latitude"], region_1["latitude"])
        self.assertEqual(region_2["longitude"], region_1["longitude"])




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
        region = estimate_region(self.event, min_radius=50.)
        self.assertEqual(
            self.event.origins[0].latitude, region["latitude"])
        self.assertEqual(
            self.event.origins[0].longitude, region["longitude"])
        self.assertGreater(
            region["maxradius"], kilometer2degrees(50))

    def test_estimate_region_no_origin(self):
        event = copy.deepcopy(self.event)
        event.preferred_origin_id = None
        event.origins = []
        self.assertIsNone(estimate_region(event))

    def test_estimate_region_no_magnitude(self):
        event = copy.deepcopy(self.event)
        event.magnitudes = []
        region = estimate_region(event)
        self.assertEqual(
            self.event.origins[0].latitude, region["latitude"])
        self.assertEqual(
            self.event.origins[0].longitude, region["longitude"])
        self.assertEqual(
            region["maxradius"], kilometer2degrees(50))

#
# class BackfillTest(unittest.TestCase):
#     def test_basic(self):
#         raise NotImplementedError


if __name__ == "__main__":
    import logging
    logging.basicConfig(level="DEBUG")

    unittest.main()
