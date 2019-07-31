"""
Tests for simulating a real-time client.
"""

import unittest
import time

from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

from rt_eqcorrscan.streaming.simulate import SimulateRealTimeClient


class FDSNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = Client("GEONET")
        cls.rt_client = SimulateRealTimeClient(
            client=cls.client, buffer_capacity=10,
            starttime=UTCDateTime(2017, 1, 1), speed_up=4., query_interval=5.)
        cls.rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")

    def test_background_streaming(self):
        self.rt_client.background_run()
        time.sleep(20)
        self.rt_client.background_stop()
        self.assertEqual(self.rt_client.buffer_length,
                         self.rt_client.buffer_capacity)

    def test_full_buffer(self):
        self.rt_client.clear_buffer()
        self.rt_client.background_run()
        self.assertFalse(self.rt_client.buffer_full)
        time.sleep(20)  # GeoNet is slow
        self.rt_client.background_stop()
        self.assertTrue(self.rt_client.buffer_full)


if __name__ == "__main__":
    unittest.main()
