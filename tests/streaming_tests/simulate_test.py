"""
Tests for simulating a real-time client.
"""

import unittest
import time

from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

from rt_eqcorrscan.streaming.simulate import SimulateRealTimeClient

SLEEP_STEP = 20


class FDSNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = Client("GEONET")
        cls.rt_client = SimulateRealTimeClient(
            client=cls.client, buffer_capacity=10,
            starttime=UTCDateTime(2017, 1, 1), speed_up=4., query_interval=5.)

    def test_background_streaming(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        time.sleep(SLEEP_STEP)
        rt_client.background_stop()
        self.assertEqual(rt_client.buffer_length,
                         rt_client.buffer_capacity)

    def test_full_buffer(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        self.assertFalse(rt_client.buffer_full)
        time.sleep(SLEEP_STEP)
        rt_client.background_stop()
        self.assertTrue(rt_client.buffer_full)

    def test_always_started(self):
        rt_client = self.rt_client.copy()
        rt_client.start()
        self.assertTrue(rt_client.started)
        rt_client.stop()
        self.assertFalse(rt_client.started)

    def test_can_add_streams(self):
        self.assertTrue(self.rt_client.can_add_streams)

    def test_copy_with_buffer(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        time.sleep(SLEEP_STEP)
        rt_client.background_stop()
        new_client = rt_client.copy(empty_buffer=False)
        new_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        self.assertEqual(new_client.get_stream(), rt_client.get_stream())
        new_client.background_run()
        time.sleep(SLEEP_STEP * 2)
        new_client.background_stop()
        # Make sure that we don't change the old buffer.
        self.assertNotEqual(new_client.get_stream(), rt_client.get_stream())

    def test_reprint(self):
        print_str = self.rt_client.__repr__()
        self.assertEqual(print_str, "Client at http://service.geonet.org.nz, "
                                    "status: Stopped, buffer capacity: 10.0s\n	"
                                    "Current Buffer:\nBuffer(0 traces, "
                                    "maxlen=10)")


if __name__ == "__main__":
    unittest.main()
