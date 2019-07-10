"""
Test the real-time plotting routines.

Not to be run on CI.
"""

import unittest
import time
import logging

from itertools import cycle

from obspy import UTCDateTime
from obspy.core.event import Event, Pick, WaveformStreamID
from obspy.clients.fdsn import Client

from eqcorrscan.core.match_filter import Tribe, Detection, Template

from rt_eqcorrscan.streaming.seedlink import RealTimeClient
from rt_eqcorrscan.plotting.plot_buffer import EQcorrscanPlot


logging.basicConfig(
    level="INFO",
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

SLEEP_STEP = .5
MAX_DURATION = 3600


class SeedLinkTest(unittest.TestCase):
    def test_real_time_plotting(self):
        """Test the real-time plotter - must be run interactively."""

        seed_list = [
            "NZ.INZ.10.HHZ", "NZ.JCZ.10.HHZ", "NZ.FOZ.11.HHZ", "NZ.MSZ.10.HHZ",
            "NZ.PYZ.10.HHZ", "NZ.DCZ.10.HHZ", "NZ.WVZ.10.HHZ"]
        client = Client("GEONET")
        inv = client.get_stations(
            network=seed_list[0].split(".")[0],
            station=seed_list[0].split(".")[1],
            location=seed_list[0].split(".")[2],
            channel=seed_list[0].split(".")[3])
        for seed_id in seed_list[1:]:
            net, sta, loc, chan = seed_id.split('.')
            inv += client.get_stations(
                network=net, station=sta, channel=chan, location=loc)

        now = UTCDateTime.now()
        template_cat = client.get_events(
            starttime=now - 3600, endtime=now)
        tribe = Tribe(templates=[
            Template(event=event, name=event.resource_id.id.split("/")[-1])
            for event in template_cat])
        template_names = cycle([t.name for t in tribe])

        buffer_capacity = 1200
        rt_client = RealTimeClient(
            server_url="link.geonet.org.nz", buffer_capacity=buffer_capacity)
        for seed_id in seed_list:
            net, station, _, selector = seed_id.split(".")
            rt_client.select_stream(
                net=net, station=station, selector=selector)

        rt_client.background_run()
        while len(rt_client.buffer) < 7:
            # Wait until we have some data
            time.sleep(SLEEP_STEP)

        detections = []
        plotter = EQcorrscanPlot(
            rt_client=rt_client, plot_length=600,
            tribe=tribe, inventory=inv, update_interval=1000,
            detections=detections)
        plotter.background_run()

        duration = 0
        while duration < MAX_DURATION:
            detections.append(
                Detection(
                    template_name=next(template_names),
                    detect_time=UTCDateTime.now(), no_chans=999,
                    detect_val=999, threshold=999, threshold_type="MAD",
                    threshold_input=999, typeofdet="unreal",
                    event=Event(picks=[
                        Pick(time=UTCDateTime.now(),
                             waveform_id=WaveformStreamID(seed_string=seed_id))
                        for seed_id in seed_list])))
            time.sleep(20)
            duration += 20
        rt_client.background_stop()
        self.assertEqual(
            len(rt_client.get_stream()[0]) *
            rt_client.get_stream()[0].stats.delta,
            buffer_capacity)


if __name__ == "__main__":
    unittest.main()