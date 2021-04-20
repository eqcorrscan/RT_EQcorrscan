"""
Tests for simulating a real-time client.
"""

import unittest
import time
import logging

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from rt_eqcorrscan.streaming.clients.obspy import RealTimeClient

SLEEP_STEP = 20

logging.basicConfig(
    level="INFO",
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


class FDSNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = Client("GEONET")
        cls.rt_client = RealTimeClient(
            server_url="Unreal-streamer",
            client=cls.client, buffer_capacity=10,
            starttime=UTCDateTime(2017, 1, 1), speed_up=4., query_interval=5.)
        cls.clients = [cls.rt_client]

    def test_background_streaming(self):
        rt_client = self.rt_client.copy()
        self.clients.append(rt_client)  # To allow tear-down
        rt_client.select_stream(net="NZ", station="JCZ", selector="HHZ")
        rt_client.background_run()
        self.assertFalse(rt_client.buffer_full)
        time.sleep(SLEEP_STEP)
        rt_client.background_stop()
        print(rt_client.stream)
        self.assertTrue(rt_client.buffer_full)
        self.assertEqual(rt_client.buffer_length,
                         rt_client.buffer_capacity)

    def test_always_started(self):
        rt_client = self.rt_client.copy()
        self.clients.append(rt_client)  # To allow tear-down
        rt_client.start()
        self.assertTrue(rt_client.started)
        rt_client.stop()
        self.assertFalse(rt_client.started)

    def test_can_add_streams(self):
        self.assertTrue(self.rt_client.can_add_streams)

    # def test_copy_with_buffer(self):
    #     rt_client = self.rt_client.copy()
    #     rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
    #     rt_client.background_run()
    #     time.sleep(SLEEP_STEP)
    #     rt_client.background_stop()
    #     new_client = rt_client.copy(empty_buffer=False)
    #     new_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
    #     self.assertEqual(new_client.get_stream(), rt_client.stream)
    #     new_client.background_run()
    #     time.sleep(SLEEP_STEP * 2)
    #     new_client.background_stop()
    #     # Make sure that we don't change the old buffer.
    #     self.assertNotEqual(new_client.get_stream(), rt_client.stream)

    def test_reprint(self):
        print_str = self.rt_client.__repr__()
        self.assertEqual(print_str, "Client at http://service.geonet.org.nz, "
                                    "status: Stopped, buffer capacity: 10.0s\n	"
                                    "Current Buffer:\nBuffer(0 traces, "
                                    "maxlen=10)")

    # @classmethod
    # def tearDownClass(cls):
    #     n = len(cls.clients)
    #     for i, rt_client in enumerate(cls.clients):
    #         print(f"Killing client {i + 1} of {n}")
    #         rt_client.background_stop()


if __name__ == "__main__":
    unittest.main()
