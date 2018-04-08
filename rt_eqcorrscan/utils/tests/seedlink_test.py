"""
Tests for real-time matched-filtering.
"""

import unittest
import time

from rt_eqcorrscan.utils.seedlink import RealTimeClient


class RealTimeTribeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=10)

    def test_select_and_stream(self):
        self.rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        self.rt_client.background_run()
        time.sleep(20)
        self.rt_client.background_stop()
        self.assertEqual(self.rt_client.buffer_length(),
                         self.rt_client.buffer_capacity)


if __name__ == "__main__":
    unittest.main()
