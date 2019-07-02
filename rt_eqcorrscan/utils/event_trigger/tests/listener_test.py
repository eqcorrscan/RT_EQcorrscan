"""
Tests for the event listener.
"""

import unittest
import time
import os

from obspy import Catalog
from obspy.clients.fdsn import Client

from rt_eqcorrscan.utils.event_trigger.catalog_listener import CatalogListener
from rt_eqcorrscan.core.database_manager import _test_template_bank


class ListeningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_path = os.path.abspath(os.path.dirname(__file__))

    def test_listener(self):
        listener = CatalogListener(
            client=Client("GEONET"), catalog=Catalog(),
            interval=20, template_bank=_test_template_bank(self.test_path),
            catalog_lookup_kwargs=dict(
                latitude=-42.0, longitude=175.2, maxradius=3.), keep=600,
        )
        listener.background_run()
        self.assertTrue(listener.busy)
        time.sleep(10)
        listener.background_stop()
        self.assertFalse(listener.busy)


if __name__ == "__main__":
    unittest.main()
