"""
Long running test to make sure that streamers can consistently get data.
"""


import unittest
import time
import numpy as np
import os

from obspy import UTCDateTime
from obsplus import WaveBank

from rt_eqcorrscan.streaming.clients.seedlink import RealTimeClient

import logging

logging.basicConfig(
    level="INFO",
    format="%(asctime)s\t[%(processName)s:%(threadName)s]: %(name)s\t%(levelname)s\t%(message)s")

RUN_LENGTH = 600.0    # Run length in seconds.
SLEEP_INTERVAL = 30.0  # Interval to check the stream at.


@unittest.skipIf("CI" in os.environ and os.environ["CI"] == "true",
                 "Skipping this test on CI.")
class LongSeedLinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=600.)
        cls.selectors = [
            ("NZ", "FOZ", "HHZ"),
            ("NZ", "JCZ", "HHZ"),
            ("NZ", "WVZ", "HHZ"),
            ("NZ", "RPZ", "HHZ"),
            ("NZ", "PYZ", "HHZ"),
            ("NZ", "URZ", "HHZ"),
            ("NZ", "WEL", "HHZ"),
            ("NZ", "INZ", "HHZ"),
            ("NZ", "APZ", "HHZ"),
            ("NZ", "SYZ", "HHZ"),
            ("NZ", "MQZ", "HHZ"),
        ]

    def run_streamer(self, rt_client, logger):
        for net, sta, chan in self.selectors:
            rt_client.select_stream(net=net, station=sta, selector=chan)

        rt_client.background_run()
        sleepy_time = 0
        try:
            while sleepy_time <= RUN_LENGTH:
                time.sleep(SLEEP_INTERVAL)
                now = UTCDateTime.now()
                st = rt_client.stream.split().merge()
                logger.info(f"Currently (at {now}) have the stream: \n{st}")
                for tr in st:
                    if np.ma.is_masked(tr.data):
                        # Check that the data are not super gappy.
                        self.assertLess(tr.data.mask.sum(), len(tr.data) / 4)
                    # Check that data are recent.
                    self.assertLess(abs(now - rt_client.last_data), 60.0)
                    self.assertLess(abs(now - tr.stats.endtime), 120.0)
                sleepy_time += SLEEP_INTERVAL
        finally:  # We MUST stop the streamer even if we fail.
            rt_client.background_stop()

    def test_no_wavebank(self):
        """ Run without a wavebank. """
        logger = logging.getLogger("NoWaveBank")
        rt_client = self.rt_client.copy()
        self.run_streamer(rt_client=rt_client, logger=logger)


if __name__ == "__main__":
    unittest.main()
