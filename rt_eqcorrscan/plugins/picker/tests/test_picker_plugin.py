"""
Tests for the lag-calc plugin for RTEQcorrscan
"""

import os
import shutil
import unittest
import pickle
import time
import traceback
import logging

from multiprocessing import Process

from obspy import Catalog, read_events, Stream
from obspy.clients.fdsn import Client

from obsplus.bank import WaveBank

from eqcorrscan import Party

from rt_eqcorrscan.plugins.picker import (
    events_to_party, get_stream, PickerConfig)
from rt_eqcorrscan.plugins.picker import Picker


Logger = logging.getLogger(__name__)


def _write_detections_for_sim(
    catalog: Catalog,
    outdir: str,
    poisondir: str,
    sleep_step: float = 40.0,
):
    slices = [slice(0, 5), slice(5, 12), slice(12, None)]
    for _slice in slices:
        events = catalog[_slice]
        Logger.info(events)
        for event in events:
            Logger.debug(f"Writing event {event.resource_id.id}")
            event.write(f"{outdir}/{event.resource_id.id}.xml",
                        format="QUAKEML")
        time.sleep(sleep_step)
    Logger.info("Writing poison file")
    with open(f"{poisondir}/poison", "w") as f:
        f.write("Poisoned at end of detection writing")
    return


class TestLagCalcPlugin(unittest.TestCase):
    clean_up = []
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "test_data")
        cls.party = Party().read(f"{cls.test_dir}/lag_calc_test_party.tgz")

    def test_party_construction(self):
        party = self.party.copy()

        # Get the events out of the party
        events = party.get_catalog()

        # Write out into expected structure
        template_dir = ".templates"
        os.makedirs(template_dir, exist_ok=True)
        self.clean_up.append(template_dir)
        for f in party:
            t = f.template
            with open(f"{template_dir}/{t.name}.pkl", "wb") as fp:
                pickle.dump(t, fp)

        party_back = events_to_party(events=events, template_dir=template_dir)
        # self.assertEqual(party, party_back)
        # Some elements (threshold_input, threshold_type, threshold)
        # lost in translation
        for fam in party:
            fam_back = party_back.select(fam.template.name)
            if len(fam) == 0:
                self.assertIsNone(fam_back)  # We don't read in empty families
                continue
            self.assertEqual(len(fam), len(fam_back))
            self.assertEqual(fam.template, fam_back.template)
            dets = sorted(fam.detections, key=lambda d: d.detect_time)
            dets_back = sorted(fam_back.detections, key=lambda d: d.detect_time)
            for d, db in zip(dets, dets_back):
                for key, value in d.__dict__.items():
                    if key in ["threshold_type", "threshold_input", "threshold"]:
                        continue
                    value_back = db.__dict__[key]
                    self.assertEqual(value, value_back)

    def test_lag_calc_runner(self):
        config = PickerConfig()
        config_file = "test_lagcalc_config.yml"
        detect_dir = ".lag_calc_test_detections"
        template_dir = ".lag_calc_test_templates"
        wavebank_dir = ".lag_calc_test_wavebank"
        outdir = ".lag_calc_test_outdir"
        config.sleep_interval = 2.0
        config.in_dir, config.template_dir, config.wavebank_dir = (
            detect_dir, template_dir, wavebank_dir)
        config.station_file = os.path.join(self.test_dir, "stations.xml")
        config.min_cc = 0.1
        config.out_dir = outdir
        config.write(config_file)
        self.clean_up.extend(
            [config_file, detect_dir, template_dir, wavebank_dir, outdir])
        for _dir in [detect_dir, template_dir, wavebank_dir, outdir]:
            os.makedirs(_dir, exist_ok=True)

        # Hack process lengths for this test
        party = self.party.copy()
        for f in party:
            f.template.process_length = 300.0

        # Populate directories
        for f in party:
            with open(f"{template_dir}/{f.template.name}.pkl", "wb") as fp:
                pickle.dump(f.template, fp)

        # Get a useful stream
        bulk = set()
        starttime = min(p.time for f in party for d in f
                        for p in d.event.picks) - 350
        endtime = max(p.time for f in party for d in f
                      for p in d.event.picks) + 480
        for family in party:
            for detection in family:
                for pick in detection.event.picks:
                    bulk.update({(
                        pick.waveform_id.network_code or "*",
                        pick.waveform_id.station_code or "*",
                        pick.waveform_id.location_code or "*",
                        pick.waveform_id.channel_code or "*",
                        starttime.datetime, endtime.datetime)})
        # Convert to UTCDateTime
        bulk = [(b[0], b[1], b[2], b[3], starttime, endtime) for b in bulk]
        client = Client("GEONET")
        Logger.debug(f"Downloading streams for bulk:\n{bulk}")
        stream = Stream()
        for _bulk in bulk:
            stream += client.get_waveforms(*_bulk)
        bank = WaveBank(wavebank_dir)
        bank.put_waveforms(stream)

        # We need to set up a process to periodically write detections to
        # the detect_dir
        catalog = party.get_catalog()
        assert len(catalog)
        detection_writer = Process(
            target=_write_detections_for_sim,
            args=(catalog, detect_dir, outdir, 20.),
            name="DetectionWriter")

        # Run the process in the background
        Logger.info("Starting detection writer")
        detection_writer.start()

        Logger.info("Starting lag-calc runner")
        failed = False
        try:
            picker = Picker(config_file=config_file)
            picker.run()
        except Exception as e:
            Logger.error(f"Failed due to {e}")
            trace = traceback.format_exc()
            Logger.exception(trace)
            failed = True
        finally:
            detection_writer.kill()

        self.assertFalse(failed)
        # Get the detections back
        cat_back = read_events(f"{outdir}/*.xml")
        picks = [p for ev in cat_back for p in ev.picks
                 if p.phase_hint in "PS"]
        self.assertGreater(len(picks), 0)
        amp_picks = [p for ev in cat_back for p in ev.picks
                     if p.phase_hint.endswith("AML")]
        self.assertGreater(len(amp_picks), 0)
        self.assertEqual(len(cat_back), len(catalog))

    @classmethod
    def tearDownClass(cls, clean=True):
        if clean:
            for thing in cls.clean_up:
                if os.path.isdir(thing):
                    shutil.rmtree(thing)
                else:
                    os.remove(thing)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level="DEBUG")
    unittest.main()
