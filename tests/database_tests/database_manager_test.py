"""
Test for database management in rt_eqcorrscan
"""

import unittest
import os
import shutil
import logging

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from eqcorrscan.utils.catalog_utils import filter_picks
from eqcorrscan.core.match_filter import read_tribe

from rt_eqcorrscan.database.database_manager import (
    TemplateBank, check_tribe_quality, remove_unreferenced)


class TestTemplateBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_path = (
            os.path.abspath(os.path.dirname(__file__)) + os.path.sep + "db")
        if os.path.isdir(cls.test_path):
            shutil.rmtree(cls.test_path)
        os.makedirs(cls.test_path)
        logging.debug("Making the bank")
        cls.bank = TemplateBank(
            base_path=cls.test_path, name_structure="{event_id_short}")
        cls.client = Client("GEONET")
        logging.debug("Downloading the catalog")
        catalog = cls.client.get_events(
            starttime=UTCDateTime(2019, 6, 21),
            endtime=UTCDateTime(2019, 6, 23),
            latitude=-38.8, longitude=175.8, maxradius=0.2)
        cls.catalog = remove_unreferenced(
            filter_picks(catalog=catalog, top_n_picks=5))
        cls.bank.put_events(cls.catalog)

    def test_build_db(self):
        logging.debug("Making templates")
        _ = self.bank.make_templates(
            catalog=self.catalog, client=self.client, lowcut=2., highcut=15.,
            samp_rate=50., filt_order=4, prepick=0.5, length=3, swin="all")
        cat_back = self.bank.get_events()
        cat_back.events.sort(key=lambda e: e.origins[0].time)
        self.assertEqual(self.catalog, cat_back)
        tribe = self.bank.get_templates()
        self.assertEqual(len(tribe), len(self.catalog))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.test_path)


class TestTribeFiltering(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tribe = read_tribe(os.path.join(
            os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
            "test_data", "test_tribe.tgz"))

    def test_good_tribe_quality(self):
        checked_tribe = check_tribe_quality(self.tribe)
        for template in checked_tribe:
            for tr in template.st:
                self.assertEqual(tr.stats.npts, 150)

    def test_too_short_chans(self):
        tribe = self.tribe.copy()
        tribe[0].st[0].data = tribe[0].st[0].data[0:-10]
        checked_tribe = check_tribe_quality(tribe)
        for template in checked_tribe:
            for tr in template.st:
                self.assertEqual(tr.stats.npts, 150)

    def test_removing_channels(self):
        chans = {tr.id for template in self.tribe for tr in template.st}
        removed_chan = chans.pop()
        checked_tribe = check_tribe_quality(self.tribe, seed_ids=chans)
        for template in checked_tribe:
            template_ids = {tr.id for tr in template.st}
            self.assertNotIn(removed_chan, template_ids)

    def test_removing_short_templates(self):
        checked_tribe = check_tribe_quality(self.tribe, min_stations=5)
        self.assertLess(len(checked_tribe), len(self.tribe))
        for template in checked_tribe:
            stas = {tr.stats.station for tr in template.st}
            self.assertGreaterEqual(len(stas), 5)


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    unittest.main()
