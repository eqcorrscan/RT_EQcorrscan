"""
Tests for real-time matched-filtering.
"""

import unittest
import time
import shutil
import os
import pytest

from obspy import Stream

from rt_eqcorrscan.streaming.clients.seedlink import RealTimeClient

import logging

logging.basicConfig(
    level="INFO",
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

SLEEP_STEP = 30


# Note:: Must always have try: finally: to stop the streamer to avoid
# continuous running on fail!
class SeedLinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=10.)

    @pytest.mark.flaky(reruns=1)
    def test_background_streaming(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        time.sleep(SLEEP_STEP)
        try:
            self.assertEqual(rt_client.buffer_length,
                             rt_client.buffer_capacity)
        finally:
            rt_client.background_stop()

    @pytest.mark.flaky(reruns=1)
    def test_full_buffer(self):
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.clear_buffer()
        rt_client.background_run()
        try:
            self.assertFalse(rt_client.buffer_full)
            time.sleep(SLEEP_STEP)
            self.assertTrue(rt_client.buffer_full)
        finally:
            rt_client.background_stop()

    @pytest.mark.flaky(reruns=1)
    def test_can_add_streams(self):
        rt_client = self.rt_client.copy()
        self.assertTrue(rt_client.can_add_streams)
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        try:
            self.assertFalse(rt_client.can_add_streams)
        finally:
            rt_client.background_stop()
        self.assertFalse(rt_client.can_add_streams)
        rt_client = self.rt_client.copy(empty_buffer=False)
        self.assertTrue(rt_client.can_add_streams)

    @pytest.mark.flaky(reruns=1)
    def test_get_stream(self):
        initial_sleep = 10
        rt_client = self.rt_client.copy()
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        time.sleep(initial_sleep)
        stream = rt_client.stream
        try:
            self.assertIsInstance(stream, Stream)
            time.sleep(SLEEP_STEP - initial_sleep)
        finally:
            rt_client.background_stop()
        stream2 = rt_client.stream
        self.assertNotEqual(stream, stream2)


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
        st = Stream()
        while toc - tic < SLEEP_STEP:
            st = rt_client.stream
            toc = time.time()
        rt_client.background_stop()
        self.assertNotEqual(len(st), 0)
        # If we got here without error, then we should be safe.

    def test_add_trace_from_mainprocess(self):
        """ Check that adding a trace from the main process works. """
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=600.)
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        # Wait for some data to perform the test
        tic, toc = time.time(), time.time()
        while toc - tic < SLEEP_STEP:
            time.sleep(1)
            st = rt_client.stream.split()
            if len(st):
                break
            toc = time.time()
        else:
            rt_client.background_stop()  # Must stop the streamer!
            raise NotImplementedError("Did not accumulate any data")
        tr = st[0]
        tr.stats.starttime -= 100
        rt_client.on_data(tr)
        time.sleep(SLEEP_STEP)
        st = rt_client.stream.split().merge()
        try:
            self.assertEqual(len(st), 1)  # "More than one trace in stream!"
            self.assertLess(st[0].stats.starttime - tr.stats.starttime, 0.01)
            self.assertGreater(st[0].stats.endtime, tr.stats.endtime)
        finally:
            rt_client.background_stop()

    def test_add_data_not_streaming(self):
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=600.)
        rt_client.select_stream(net="NZ", station="FOZ", selector="HHZ")
        rt_client.background_run()
        # Wait for some data to perform the test
        tic, toc = time.time(), time.time()
        while toc - tic < SLEEP_STEP:
            time.sleep(1)
            st = rt_client.stream.split()
            if len(st):
                break
            toc = time.time()
        else:
            rt_client.background_stop()  # Must stop the streamer!
            raise NotImplementedError("Did not accumulate any data")
        # Stop the streamer here.
        rt_client.background_stop()

        tr = st[0].copy()
        tr.stats.starttime -= 100
        # Try to add the trace.
        rt_client.on_data(tr)
        st2 = rt_client.stream.split().merge()
        st2.sort(key=lambda _tr: _tr.stats.starttime)
        # Check that the starttimes are essentially the same
        if st2[0].stats.starttime - tr.stats.starttime >= 0.01:
            st2.write("borked_seedlinker.ms", format="MSEED")
        self.assertLess(st2[0].stats.starttime - tr.stats.starttime, 0.01)
        # Check that the endtime of the buffer is after the trace that was
        # shifted by 100 s
        self.assertGreater(st2[0].stats.endtime, tr.stats.endtime)


if __name__ == "__main__":
    unittest.main()
