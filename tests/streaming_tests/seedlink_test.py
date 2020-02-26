"""
Tests for real-time matched-filtering.
"""

import unittest
import time
import shutil
import numpy as np

from obspy import Stream
from obsplus import WaveBank

from rt_eqcorrscan.streaming.clients.seedlink import RealTimeClient


class SeedLinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=10.)

    def test_background_streaming(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        time.sleep(20)
        rt_client.background_stop()
        self.assertEqual(rt_client.buffer_length,
                         rt_client.buffer_capacity)

    def test_full_buffer(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.clear_buffer()
        rt_client.background_run()
        self.assertFalse(rt_client.buffer_full)
        time.sleep(18)
        rt_client.background_stop()
        self.assertTrue(rt_client.buffer_full)

    def test_can_add_streams(self):
        rt_client = self.rt_client.copy()
        self.assertTrue(rt_client.can_add_streams)
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        self.assertFalse(rt_client.can_add_streams)
        rt_client.background_stop()
        self.assertFalse(rt_client.can_add_streams)
        rt_client = self.rt_client.copy(empty_buffer=False)
        self.assertTrue(rt_client.can_add_streams)

    def test_start_connection(self):
        rt_client = self.rt_client.copy()
        rt_client.start()
        # This next one should warn
        with self.assertLogs(level="WARNING") as cm:
            rt_client.start()
            self.assertIn("connection already started", cm.output[0])
        rt_client.stop()

    def test_get_stream(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        time.sleep(10)
        stream = rt_client.get_stream()
        self.assertIsInstance(stream, Stream)
        time.sleep(10)
        stream2 = rt_client.get_stream()
        self.assertNotEqual(stream, stream2)

    def test_wavebank_integration(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.wavebank = WaveBank(base_path="test_wavebank")
        rt_client.background_run()
        time.sleep(20)
        rt_client.background_stop()
        self.assertTrue(rt_client.buffer_full)  # Need a full buffer to work
        wavebank_traces = rt_client.wavebank.get_waveforms()
        wavebank_stream = wavebank_traces.merge()
        buffer_stream = rt_client.get_stream()
        wavebank_stream.sort()
        buffer_stream.sort()
        self.assertEqual(buffer_stream[0].stats.endtime,
                         wavebank_stream[0].stats.endtime)
        self.assertTrue(
            np.all(wavebank_stream.slice(starttime=buffer_stream[0].stats.starttime)[0].data == buffer_stream[0].data))
        shutil.rmtree("test_wavebank")


if __name__ == "__main__":
    unittest.main()
