""" Test the detection manager. """

import logging
import os
import random
import unittest
import shutil
import subprocess
import pickle
import time

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.event import Event, Origin, Pick, WaveformStreamID
from obsplus import WaveBank

from eqcorrscan import Detection

from rt_eqcorrscan.plugins import detection_manager
from rt_eqcorrscan.plugins.detection_manager import _detection_filename

Logger = logging.getLogger(__name__)

MINTIME = UTCDateTime(2020, 1, 1)
MAXTIME = UTCDateTime(2020, 1, 2)


def _rando_det(offset):
    d = Detection(
        template_name="bob", detect_time=MINTIME + offset, no_chans=2,
        detect_val=1.3, threshold=8, typeofdet="corr", threshold_type="abs",
        threshold_input="1.3", chans=[("WVZ", "HHZ"), ("RPZ", "HHZ")])
    d.event = Event(
        origins=[Origin(time=d.detect_time)],
        picks=[
            Pick(time=d.detect_time + 1,
                 waveform_id=WaveformStreamID(
                     network_code="NZ", station_code="WVZ", location_code="10",
                     channel_code="HHZ")),
            Pick(time=d.detect_time + 1.3,
                 waveform_id=WaveformStreamID(
                     network_code="NZ", station_code="RPZ", location_code="10",
                     channel_code="HHZ"))])
    return d


def _rando_detections(n_dets: int):
    s_range = int((MAXTIME - MINTIME) - 60)
    return [_rando_det(random.randrange(s_range)) for _ in range(n_dets)]


class DetectionManagerTest(unittest.TestCase):
    bank_path = "test_bank"
    detect_directory = "test_detections"

    @classmethod
    def setUpClass(cls) -> None:
        """ Download waveforms. """
        client = Client("GEONET")
        bulk = [
            ("NZ", "WVZ", "10", "HHZ", MINTIME, MAXTIME),
            ("NZ", "RPZ", "10", "HHZ", MINTIME, MAXTIME),
        ]
        st = client.get_waveforms_bulk(bulk=bulk)
        Logger.info(f"Downloaded {len(st)} traces")
        bank = WaveBank(cls.bank_path)
        bank.put_waveforms(st)

    def test_subproc_run(self):
        script_path = detection_manager.__file__
        _call = [
            "python", script_path,
            "-d", self.detect_directory,
            "-w", self.bank_path, "-s", "-p", "-L"]

        Logger.info("Running `{call}`".format(call=" ".join(_call)))
        _detection_manager = subprocess.Popen(_call)

        # Put some detections in the directory
        detections = _rando_detections(5)
        for d in detections:
            detect_file_base = _detection_filename(
                detection=d,
                detect_directory=self.detect_directory)
            detect_filename = f"{detect_file_base}.pkl"
            if os.path.isfile(detect_filename):
                continue
            with open(detect_filename, "wb") as f:
                pickle.dump(d, f)
            Logger.info(f"Written to {detect_filename}, sleeping")
            time.sleep(2)

        Logger.info("Waiting to let the manager keep up")
        time.sleep(5)

        # Stop the manager
        Logger.info("Stopping the manager")
        _detection_manager.kill()
        time.sleep(2)

        # Check for results
        for d in detections:
            detect_file_base = _detection_filename(
                detection=d,
                detect_directory=self.detect_directory)
            self.assertTrue(os.path.isfile(f"{detect_file_base}.ms"))
            self.assertTrue(os.path.isfile(f"{detect_file_base}.xml"))

    def tearDown(self) -> None:
        # Clean up
        shutil.rmtree(self.detect_directory)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.bank_path)


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")

    unittest.main()
