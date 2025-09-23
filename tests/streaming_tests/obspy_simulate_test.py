"""
Tests for simulating a real-time client.
"""

import unittest
import time
import logging

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from rt_eqcorrscan.streaming.clients.obspy import RealTimeClient, StreamClient

SLEEP_STEP = 20

Logger = logging.getLogger(__name__)


class TestStreamBuffer(unittest.TestCase):
    def test_maintain(self):
        client = Client("GEONET")
        buffer = StreamClient(
            client=client, buffer_length=60, min_buffer_fraction=0.25)
        buffer.initiate_buffer(
            ["NZ.WVZ.10.HHZ", "NZ.RPZ.10.HHZ"],
            UTCDateTime(2020, 1, 1))
        time.sleep(1)
        initial_stream = buffer.stream
        Logger.info(f"Initialised buffer as: \n{initial_stream}")
        buffer.maintain_buffer()
        st = buffer.get_waveforms_bulk(
            [("NZ", "WVZ", "10", "HHZ",
              UTCDateTime(2020, 1, 1),
              UTCDateTime(2020, 1, 1, 0, 0, 6)),
             ("NZ", "RPZ", "10", "HHZ",
              UTCDateTime(2020, 1, 1),
              UTCDateTime(2020, 1, 1, 0, 0, 6))])
        try:
            self.assertEqual(len(st), 2)
            trimmed_stream = buffer.stream
            Logger.info(f"Cut out first 6 seconds to give buffer: \n{trimmed_stream}")
            self.assertLess(
                trimmed_stream[0].stats.npts, initial_stream[0].stats.npts)
            # remove more than 75% of buffer
            st2 = buffer.get_waveforms_bulk(
                [("NZ", "WVZ", "10", "HHZ",
                  UTCDateTime(2020, 1, 1),
                  UTCDateTime(2020, 1, 1, 0, 0, 50)),
                 ("NZ", "RPZ", "10", "HHZ",
                  UTCDateTime(2020, 1, 1),
                  UTCDateTime(2020, 1, 1, 0, 0, 50))])
            self.assertEqual(len(st), 2)
            Logger.info("Sleeping to allow buffer to refill")
            time.sleep(20)  # Sleep for more than half the min buffer fraction
            out_buffer = buffer.stream

            for tr in out_buffer:
                self.assertGreaterEqual(
                    tr.stats.endtime - tr.stats.starttime, 60)
        finally:
            buffer.background_stop()


class FDSNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = Client("GEONET")
        cls.buffer = StreamClient(
            client=cls.client, buffer_length=60, min_buffer_fraction=0.25)
        cls.rt_client = RealTimeClient(
            server_url="Unreal-streamer",
            client=cls.buffer, buffer_capacity=10,
            starttime=UTCDateTime(2018, 1, 1), speed_up=2., query_interval=5.,
            pre_empt_data=True)

    def test_background_streaming(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="JCZ", selector="HHZ")
        rt_client.select_stream(net="NZ", station="RPZ", selector="HHZ")
        rt_client.background_run()
        try:
            self.assertFalse(rt_client.buffer_full)
        except Exception as e:
            rt_client.background_stop()
            raise e
        Logger.info(f"Sleeping for {SLEEP_STEP}s")
        time.sleep(SLEEP_STEP)
        Logger.info("Stopping buffer")
        rt_client.background_stop()
        Logger.info("Running checks")
        self.assertTrue(rt_client.buffer_full)
        self.assertEqual(rt_client.buffer_length,
                         rt_client.buffer_capacity)

    def test_always_started(self):
        rt_client = self.rt_client.copy()
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


if __name__ == "__main__":
    unittest.main()
