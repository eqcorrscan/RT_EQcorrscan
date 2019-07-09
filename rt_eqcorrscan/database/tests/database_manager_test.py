"""
Test for database management in rt_eqcorrscan
"""

import unittest
import os
import shutil

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from rt_eqcorrscan.database.database_manager import (
    TemplateBank, check_tribe_quality)


class TestTemplateBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_path = (
            os.path.abspath(os.path.dirname(__file__)) + os.path.sep + "db")
        if not os.path.isdir(cls.test_path):
            os.makedirs(cls.test_path)
        cls.bank = TemplateBank(
            base_path=cls.test_path, event_name_structure="{event_id_short}",
            template_name_structure="{event_id_short}")
        client = Client("GEONET")
        cls.catalog = client.get_events(
            starttime=UTCDateTime(2019, 6, 20),
            endtime=UTCDateTime(2019, 6, 24),
            latitude=-38.8, longitude=175.8, maxradius=0.2)
        cls.bank.make_templates(
            catalog=cls.catalog, client=client, lowcut=2., highcut=15.,
            samp_rate=50., filt_order=4, prepick=0.5, length=3, swin="all")

    def test_build_db(self):
        cat_back = self.bank.get_events()
        self.assertEqual(self.catalog, cat_back)
        tribe = self.bank.get_templates()
        self.assertEqual(len(tribe), len(self.catalog))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.test_path)


if __name__ == "__main__":
    unittest.main()
