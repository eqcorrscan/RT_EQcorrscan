"""
Test for growclust runner
"""

import unittest
import logging
import time
import os
import shutil
import pytest

from typing import List
from multiprocessing import Process

from obspy import Catalog, UTCDateTime, read_events

from eqcorrscan.utils.catalog_utils import filter_picks

from rt_eqcorrscan.plugins.relocation.growclust_runner import (
    _cleanup, GrowClustConfig, GrowClust)

Logger = logging.getLogger(__name__)


def _write_detections_for_sim(
    catalog: Catalog,
    outdir: str,
    poison_dir: str,
    sleep_step: float = 20.0,
):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    slices = [slice(0, 1), slice(1, 5), slice(5, 50), slice(50, 125),
              slice(125, None)]
    for _slice in slices:
        events = catalog[_slice]
        Logger.info(events)
        for event in events:
            Logger.debug(f"Writing event {event.resource_id.id.split('/')[-1]}")
            event.write(f"{outdir}/{event.resource_id.id.split('/')[-1]}.xml",
                        format="QUAKEML")
        time.sleep(sleep_step)
    Logger.info(f"Detection writing complete, waiting "
                f"{sleep_step * 4}s to poison")
    time.sleep(sleep_step * 4)

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
    wavedir: str,
    eventdir: str,
) -> Catalog:
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

    # Just use the top n most picked stations
    cat = filter_picks(cat, top_n_picks=15)
    cat.events = [ev for ev in cat if len(ev.picks)]

    bulk = _get_bulk_for_cat(cat=cat)
    inv = client.get_stations_bulk(bulk, level="channel")

    st = client.get_waveforms_bulk(bulk)
    st = st.merge()

    if not os.path.isdir(wavedir):
        os.makedirs(wavedir)
    if not os.path.isdir(eventdir):
        os.makedirs(eventdir)
    inv.write(f"{eventdir}/stations.xml", format="STATIONXML")

    for ev in cat:
        ev.write(f"{eventdir}/{ev.resource_id.id.split('/')[-1]}.xml",
                 format="QUAKEML")

    for tr in st:
        tr.split().write(f"{wavedir}/{tr.id}.ms", format="MSEED")

    Logger.info("Completed test set-up")
    return cat


class TestGrowclustPlugin(unittest.TestCase):
    clean_up = []
    @classmethod
    def setUpClass(cls):
        cls.eventdir = "growclust_test_events"
        cls.wavedir = "growclust_test_streams"
        cls.outdir = "growclust_test_output"
        cls.cat = setup_testcase(
            eventdir=cls.eventdir,
            wavedir=cls.wavedir
        )
        cls.clean_up.extend([cls.eventdir, cls.wavedir, cls.outdir])

    @pytest.mark.growclust
    def test_main(self):
        out_dir, config_file = "gc_test_output", "gc_config.yml"
        config = GrowClustConfig(
            sleep_interval=5, tt_zmax=50.0, ttxmax=500.0,
            in_dir=self.eventdir, out_dir=out_dir, wavebank_dir=self.wavedir,
            station_file=f"{self.eventdir}/stations.xml",
            rmin=0.2)

        self.clean_up.extend([out_dir, config_file])

        config.write(config_file)

        gc_runner = GrowClust(config_file=config_file)
        gc_runner.run(loop=False)


        cat_back = read_events(f"{out_dir}/*.xml")
        # We don't write out all the events, only the relocated, so they may
        # not all relocate
        self.assertGreater(len(cat_back), 150)
        # self.assertEqual(len(cat_back), len(self.cat))
        # All events in cat_back should be relocated
        for ev in cat_back:
            method_id = ev.origins[-1].method_id.id.split('/')[-1]
            if method_id != "GrowClust":
                print(f"Event: {ev.resource_id.id} not relocated")
                cat_back.write("Growclustcat_failed.xml",
                               format="QUAKEML")
            self.assertTrue(method_id == "GrowClust")

    @pytest.mark.growclust
    def test_updating(self):
        in_dir, out_dir, config_file = (
            "gc_looping_test_events", "gc_looping_test_output",
            "gc_looping_config.yml")
        config = GrowClustConfig(
            sleep_interval=5, tt_zmax=50.0, ttxmax=500.0,
            in_dir=in_dir, out_dir=out_dir, wavebank_dir=self.wavedir,
            station_file=f"{self.eventdir}/stations.xml",
            rmin=0.2)

        self.clean_up.extend([in_dir, out_dir, config_file])

        config.write(config_file)

        detection_writer = Process(
            target=_write_detections_for_sim,
            args=(self.cat, in_dir, out_dir, 50.0),
            name="DetectionWriter")

        # Run the process in the background
        Logger.info("Starting detection writer")
        detection_writer.start()

        Logger.info("Starting GrowClust runner")
        failed = False
        try:
            gc_runner = GrowClust(config_file=config_file)
            gc_runner.run()
        except Exception as e:
            Logger.exception(f"Failed due to {e}")
            failed = True
        finally:
            detection_writer.kill()

        self.assertFalse(failed)

        cat_back = read_events(f"{out_dir}/*.xml")
        # We don't write out all the events, only the relocated, so they may
        # not all relocate
        self.assertGreater(len(cat_back), 150)
        # self.assertEqual(len(cat_back), len(self.cat))
        for ev in cat_back:
            method_id = ev.origins[-1].method_id.id.split('/')[-1]
            if method_id != "GrowClust":
                print(f"Event: {ev.resource_id.id} not relocated")
                cat_back.write("Growclustcat_looping_failed.xml",
                               format="QUAKEML")
            self.assertTrue(method_id == "GrowClust")

    @classmethod
    def tearDownClass(cls, clean=True):
        if clean:
            for thing in cls.clean_up:
                if os.path.isdir(thing):
                    try:
                        shutil.rmtree(thing)
                    except Exception:
                        pass
                else:
                    try:
                        os.remove(thing)
                    except Exception:
                        pass
            _cleanup()


if __name__ == "__main__":
    unittest.main()
