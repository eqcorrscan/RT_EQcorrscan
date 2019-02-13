"""
Test the real-time plotting routines.

Not to be run on CI.
"""

import unittest
import time

from obspy import Catalog, Inventory

from rt_eqcorrscan.utils.seedlink import RealTimeClient
from rt_eqcorrscan.plotting.plot_buffer import EQcorrscanPlot


SLEEP_STEP = 0.5
MAX_DURATION = 120


# TODO: Test template and inventory map plotting and detection plotting on waveforms and alpha changes on map.
class SeedLinkTest(unittest.TestCase):
    def test_real_time_plotting(self):
        """Test the real-time plotter - must be run interactively."""
        buffer_capacity = 1200
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=buffer_capacity,
            log_level='info')
        rt_client.select_stream(net="NZ", station="JCZ", selector="HH?")
        rt_client.select_stream(net="NZ", station="FOZ", selector="HH?")
        rt_client.select_stream(net="NZ", station="GCSZ", selector="EH?")
        rt_client.select_stream(net="NZ", station="WVZ", selector="HH?")

        rt_client.background_run()
        while len(rt_client.buffer) < 12:
            # Wait until we have some data
            time.sleep(SLEEP_STEP)

        plotter = EQcorrscanPlot(
            rt_client=rt_client, plot_length=600,
            template_catalog=Catalog(),
            inventory=Inventory(networks=[], source="bob"), update_interval=100)
        plotter.background_run()

        duration = 0
        while duration < MAX_DURATION:
            time.sleep(SLEEP_STEP)
            duration += SLEEP_STEP
        rt_client.background_stop()
        self.assertEqual(len(rt_client.get_stream()[0]), buffer_capacity)


if __name__ == "__main__":
    unittest.main()
