"""
Tests for the hyp plugin.
"""

import unittest
import logging
import time
import os
import shutil

from typing import List
from multiprocessing import Process

from obspy import Catalog, UTCDateTime, read_events

from rt_eqcorrscan.plugins.relocation.hyp_runner import HypConfig, _cleanup
from rt_eqcorrscan.plugins.relocation.hyp_runner import main as hyp_run_main

Logger = logging.getLogger(__name__)


def _write_detections_for_sim(
    catalog: Catalog,
    outdir: str,
    poison_dir: str,
    sleep_step: float = 20.0,
):
    slices = [slice(0, 1), slice(1, 5), slice(5, None)]
    for _slice in slices:
        events = catalog[_slice]
        Logger.info(events)
        for event in events:
            Logger.debug(f"Writing event {event.resource_id.id.split('/')[-1]}")
            event.write(f"{outdir}/{event.resource_id.id.split('/')[-1]}.xml",
                        format="QUAKEML")
        time.sleep(sleep_step)
    Logger.info("Writing poison file")
    with open(f"{poison_dir}/poison", "w") as f:
        f.write("Poisoned at end of detection writing")
    return


def _get_bulk_for_cat(cat: Catalog) -> List[tuple]:
    picks = sorted([p for ev in cat for p in ev.picks], key=lambda p: p.time)
    starttime = picks[0].time
    endtime = picks[-1].time
    bulk = {
        (p.waveform_id.network_code or "*",
         p.waveform_id.station_code or "*",
         p.waveform_id.location_code or "*",
         p.waveform_id.channel_code or "*")
        for p in picks}
    bulk = [(n, s, l, c, starttime, endtime) for n, s, l, c in bulk]
    return bulk


def setup_testcase(
    stationxml: str,
):
    """
    Download some event files and a stationxml to run.
    """
    from obspy.clients.fdsn import Client

    client = Client("GEONET")

    Logger.info("Setting up testcase")
    # Kaikōura - Cape Campbell ~ 200 events
    cat = client.get_events(
        starttime=UTCDateTime(2016, 11, 13),
        endtime=UTCDateTime(2016, 11, 14),
        minlatitude=-41.9,
        maxlatitude=-41.6,
        minlongitude=174.0,
        maxlongitude=174.4,
        maxdepth=40
    )
    Logger.info(f"Downloaded {len(cat)} events")

    bulk = _get_bulk_for_cat(cat=cat)
    inv = client.get_stations_bulk(bulk, level="channel")

    inv.write(stationxml, format="STATIONXML")
    Logger.info("Completed test set-up")
    return cat


class TestHypPlugin(unittest.TestCase):
    clean_up = []
    @classmethod
    def setUpClass(cls):
        cls.stations_file = "test_stations.xml"
        cls.cat = setup_testcase(stationxml=cls.stations_file)
        cls.clean_up.append(cls.stations_file)

    def test_runner(self):
        config = HypConfig(
            station_file=self.stations_file,
            sleep_interval=10,
        )
        in_dir, out_dir, config_file = (
            "hyp_test_in", "hyp_test_out", "hyp_test_config.yml")
        config.in_dir = in_dir
        config.out_dir = out_dir
        self.clean_up.extend([in_dir, out_dir, config_file])

        config.write(config_file)

        for _dir in [in_dir, out_dir]:
            if not os.path.isdir(_dir):
                Logger.info(f"Making directory {_dir}")
                os.makedirs(_dir)

        event_writer = Process(
            target=_write_detections_for_sim,
            args=(self.cat, in_dir, out_dir, 20.),
            name="EventWriter")

        event_writer.start()
        failed = False
        try:
            hyp_run_main(config_file)
        except Exception as e:
            Logger.error(f"Failed due to {e}")
            failed = True
        finally:
            event_writer.kill()

        self.assertFalse(failed)

        Logger.info("Reading the catalog back in")
        cat_back = read_events(f"{out_dir}/*.xml")
        Logger.info(f"Read in {len(cat_back)} events")
        self.assertEqual(len(self.cat), len(cat_back))

    @classmethod
    def tearDownClass(cls, clean=True):
        if clean:
            for thing in cls.clean_up:
                if os.path.isdir(thing):
                    shutil.rmtree(thing)
                else:
                    os.remove(thing)
            _cleanup()


if __name__ == "__main__":
    unittest.main()
