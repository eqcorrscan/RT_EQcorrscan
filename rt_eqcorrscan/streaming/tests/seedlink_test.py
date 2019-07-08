"""
Tests for real-time matched-filtering.
"""

import unittest
import time

from obspy import Stream

from rt_eqcorrscan.streaming.seedlink import RealTimeClient


class SeedLinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=10)
        cls.rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")

    def test_background_streaming(self):
        self.rt_client.background_run()
        time.sleep(20)
        self.rt_client.background_stop()
        self.assertEqual(self.rt_client.buffer_length,
                         self.rt_client.buffer_capacity)

    def test_full_buffer(self):
        self.rt_client.buffer = Stream()
        self.rt_client.background_run()
        self.assertFalse(self.rt_client.buffer_full)
        time.sleep(4)
        print(self.rt_client.buffer)
        self.assertFalse(self.rt_client.buffer_full)
        time.sleep(8)
        self.rt_client.background_stop()
        self.assertTrue(self.rt_client.buffer_full)


if __name__ == "__main__":
    unittest.main()
