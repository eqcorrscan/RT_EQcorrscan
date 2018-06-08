"""
Test the real-time plotting routines.

Not to be run on CI.
"""

import unittest
import time

from rt_eqcorrscan.utils.seedlink import RealTimeClient


class SeedLinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=60,
            log_level='info')
        cls.rt_client.select_stream(net="NZ", station="FOZ", selector="HH?")

    def test_background_plotting(self):
        self.rt_client.background_run(plot=True, plot_length=60)
        time.sleep(70)
        self.rt_client.background_stop()
        print(self.rt_client.buffer)
        for tr in self.rt_client.buffer:
            self.assertEqual(self.rt_client.buffer_capacity,
                             tr.stats.endtime - tr.stats.starttime)


if __name__ == "__main__":
    unittest.main()
