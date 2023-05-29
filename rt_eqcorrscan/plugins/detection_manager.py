"""
Manage detections for a given real-time run.

RTTribe writes out pickle files for each detection. This code will watch
for these detections and convert them to xml files with associated waveform
files.

TODO: This script should also handle cleaning removed detections lost in
   duplicate removal?
"""

import os
import glob
import shutil
import logging
import pickle
import time

from obsplus import WaveBank
from obspy import Stream
from obspy.core.event import Event
from matplotlib.pyplot import Figure

from eqcorrscan import Detection

from rt_eqcorrscan.config import Config


Logger = logging.getLogger(__name__)


def _detection_filename(
        detection: Detection,
        detect_directory: str,
) -> str:
    _path = os.path.join(
        detect_directory, detection.detect_time.strftime("%Y"),
        detection.detect_time.strftime("%j"))
    if not os.path.isdir(_path):
        os.makedirs(_path)
    _filename = os.path.join(_path, detection.id)
    return _filename


def _write_detection(
    detection: Detection,
    detect_file_base: str,
    save_waveform: bool,
    plot_detection: bool,
    stream: Stream,
    fig=None,
    backfill_dir: str = None,
    detect_dir: str = None
) -> Figure:
    """
    Handle detection writing including writing streams and figures.

    Parameters
    ----------
    detection
        The Detection to write
    detect_file_base
        File to write to (without extension)
    save_waveform
        Whether to save the waveform for the detected event or not
    plot_detection
        Whether to plot the detection waveform or not
    stream
        The stream the detection was made in - required for save_waveform and
        plot_detection.
    fig
        A figure object to reuse.
    backfill_dir:
        Backfill directory - set if the detections have already been written
        to this dir and just need to be copied.
    detect_dir
        Detection directory - only used to manipulate backfillers.

    Returns
    -------
    An empty figure object to be reused if a figure was created, or the figure
    passed to it.
    """
    from rt_eqcorrscan.rt_match_filter import _check_stream_is_int
    from rt_eqcorrscan.plotting.plot_event import plot_event

    if backfill_dir:
        backfill_file_base = (
            f"{backfill_dir}/detections/{detect_file_base.split(detect_dir)[-1]}")
        Logger.info(f"Looking for detections in {backfill_file_base}.*")
        backfill_dets = glob.glob(f"{backfill_file_base}.*")
        Logger.info(f"Copying {len(backfill_dets)} to main detections")
        for f in backfill_dets:
            ext = os.path.splitext(f)[-1]
            shutil.copyfile(f, f"{detect_file_base}{ext}")
            Logger.info(f"Copied {f} to {detect_file_base}{ext}")
        return fig

    try:
        detection.event.write(f"{detect_file_base}.xml", format="QUAKEML")
    except Exception as e:
        Logger.error(f"Could not write event file due to {e}")
    detection.event.picks.sort(key=lambda p: p.time)
    st = stream.slice(
        detection.event.picks[0].time - 10,
        detection.event.picks[-1].time + 20).copy()
    if plot_detection:
        # Make plot
        fig = plot_event(fig=fig, event=detection.event, st=st,
                         length=90, show=False)
        try:
            fig.savefig(f"{detect_file_base}.png")
        except Exception as e:
            Logger.error(f"Could not write plot due to {e}")
        fig.clf()
    if save_waveform:
        st = _check_stream_is_int(st)
        try:
            st.write(f"{detect_file_base}.ms", format="MSEED")
        except Exception as e:
            Logger.error(f"Could not write stream due to {e}")
    return fig


def _get_bulk_for_event(event: Event, prepad: float, postpad: float):
    ori_time = (event.preferred_origin() or event.origins[-1]).time
    final_pick_time = max(p.time for p in event.picks)
    bulk = [
        (p.waveform_id.network_code,
         p.waveform_id.station_code,
         p.waveform_id.location_code,
         p.waveform_id.channel_code,
         ori_time - prepad,
         final_pick_time + postpad)
        for p in event.picks]
    return bulk


class DetectionManager:
    sleep_step = 2
    def __init__(self, detect_directory: str, wavebank_directory: str):
        self.detect_directory = detect_directory
        self.handled_detection_files = []
        self._fig = Figure()
        if wavebank_directory:
            self.wavebank = WaveBank(wavebank_directory)

    def check_for_new_detections(self):
        det_files = [f for f in glob.glob(f"{self.detect_directory}/*/*/*.pkl")
                     if f not in self.handled_detection_files]
        return det_files

    def main(self, save_waveforms: bool, plot_detections: bool):
        while True:
            try:
                Logger.info(f"Checking for new detections in {self.detect_directory}")
                det_files = self.check_for_new_detections()
                if len(det_files) == 0:
                    time.sleep(self.sleep_step)
                    continue
                for det_file in det_files:
                    Logger.info(f"Processing {det_file}")
                    # Cope with files being removed while we work on them, this
                    # can happen due to concurrent declustering
                    if not os.path.isfile(det_file):
                        continue
                    with open(det_file, "rb") as f:
                        detection = pickle.load(f)
                    detect_file_base = _detection_filename(
                        detection=detection,
                        detect_directory=self.detect_directory)
                    _filename = f"{detect_file_base}.xml"
                    if os.path.isfile(f"{detect_file_base}.xml"):
                        Logger.info(f"{_filename} exists, skipping")
                        continue
                    Logger.debug(f"Writing detection: {detection.detect_time}")

                    if save_waveforms or plot_detections:
                        bulk = _get_bulk_for_event(
                            event=detection.event, prepad=5, postpad=15)

                        st = self.wavebank.get_waveforms_bulk(bulk)
                    else:
                        st = None

                    self._fig = _write_detection(
                        detection=detection,
                        detect_file_base=detect_file_base,
                        save_waveform=save_waveforms,
                        plot_detection=plot_detections, stream=st,
                        fig=self._fig, detect_dir=self.detect_directory)
                    self.handled_detection_files.append(det_file)
                    Logger.info(f"Finished handling {det_file}")
            except Exception as e:
                Logger.error(f"Exception {e}")
                break
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor and post-process detections")

    parser.add_argument(
        "-d", "--detect-directory", type=str, required=True,
        help="Location of detection to monitor and write to.")
    parser.add_argument(
        "-w", "--wavebank-directory", type=str, default=None,
        help="Location of wavebank to read waveforms from.")
    parser.add_argument(
        "-p", "--plot", action="store_true",
        help="Flag to enable plotting of waveforms")
    parser.add_argument(
        "-s", "--save-waveforms", action="store_true",
        help="Flag to enable saving of waveforms")
    parser.add_argument(
        "-l", "--logfile", type=str, default="detection_manager.log",
        help="Log file for detection manager")
    parser.add_argument(
        "-L", action="store_true", help="Log to screen", dest="log_to_screen")

    args = parser.parse_args()

    config = Config(log_level="INFO")
    config.setup_logging(
        screen=args.log_to_screen, file=True,
        filename=args.logfile)

    det_manager = DetectionManager(
        detect_directory=args.detect_directory,
        wavebank_directory=args.wavebank_directory)

    det_manager.main(save_waveforms=args.save_waveforms,
                     plot_detections=args.plot)
