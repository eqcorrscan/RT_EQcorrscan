"""
Tests for real-time matched-filtering.
"""

import unittest
import time
import shutil
import os
import numpy as np

from obspy import Stream
from obsplus import WaveBank

from rt_eqcorrscan.streaming.clients.seedlink import RealTimeClient


class SeedLinkTest(unittest.TestCase):
    def setUp(self):
        self.rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=10.)

    # Can't run tear down class properly with xdist
    # @classmethod
    # def tearDownClass(cls):
    #     if os.path.isdir("test_wavebank"):
    #         shutil.rmtree("test_wavebank")

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
        stream = rt_client.stream
        self.assertIsInstance(stream, Stream)
        time.sleep(10)
        stream2 = rt_client.stream
        self.assertNotEqual(stream, stream2)
        rt_client.background_stop()

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
        buffer_stream = rt_client.stream
        wavebank_stream.sort()
        buffer_stream.sort()
        self.assertEqual(buffer_stream[0].id, wavebank_stream[0].id)
        print(buffer_stream[0])
        print(wavebank_stream[0])
        self.assertLessEqual(
            abs(buffer_stream[0].stats.endtime -
                wavebank_stream[0].stats.endtime),
            buffer_stream[0].stats.delta * 10)
        endtime = min(buffer_stream[0].stats.endtime,
                      wavebank_stream[0].stats.endtime)
        starttime = max(buffer_stream[0].stats.starttime,
                        wavebank_stream[0].stats.starttime)
        self.assertTrue(
            np.all(wavebank_stream.slice(
                starttime=starttime, endtime=endtime)[0].data ==
                   buffer_stream.slice(
                       starttime=starttime, endtime=endtime)[0].data))
        # shutil.rmtree("test_wavebank")


class SeedLinkThreadedTests(unittest.TestCase):
    """ Checks that operations are thread-safe. """
    @classmethod
    def setUpClass(cls):
        cls.rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=10.)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("test_wavebank"):
            shutil.rmtree("test_wavebank")

    def test_read_write_from_multiple_threads(self):
        """ Check that one thread can read while the other writes. """
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        tic, toc = time.time(), time.time()
        lock_count = 0
        while toc - tic < 10.0:
            if rt_client.lock.locked():
                lock_count += 1
                _ = rt_client.stream  # This waits then acquires the lock
                self.assertFalse(rt_client.lock.locked())
            else:
                _ = rt_client.stream  # This waits then acquires the lock
            toc = time.time()
        rt_client.background_stop()
        # If we got here without error, then we should be safe.


if __name__ == "__main__":
    unittest.main()
