#!/usr/bin/env python
"""
Functions for spinning up and running a backfiller to detect with new templates in old data.
"""

import os
import logging
import gc
import pickle
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Memory tracking for debugging
import psutil
# from pympler import summary, muppy

from obspy import read, UTCDateTime, Stream

from eqcorrscan import Tribe, Party, Template
from eqcorrscan.core.match_filter.helpers import _moveout

from rt_eqcorrscan import Config
from rt_eqcorrscan.database.client_emulation import LocalClient
from rt_eqcorrscan.rt_match_filter import (
    squash_duplicates, reshape_templates, _write_detection, _detection_filename)


Logger = logging.getLogger(__name__)


def backfill(
    working_dir: str,
    expected_seed_ids: list,
    minimum_data_for_detection: float,
    threshold: float,
    threshold_type: str,
    trig_int: float,
    peak_cores: int = 1,
    cores: int = None,
    parallel_processing: bool = True,
    process_cores: int = None,
    log_to_screen: bool = False,
    starttime: UTCDateTime = None,
    endtime: UTCDateTime = None,
    plot_detections: bool = False,
    save_waveforms: bool = False,
    **kwargs
) -> None:
    """ Background backfill method designed to work in a subprocess. """
    # Get going!
    os.chdir(working_dir)
    config = Config(log_level="INFO")
    config.setup_logging(
        screen=log_to_screen, file=True,
        filename="{0}/rt_eqcorrscan_{1}.log".format(
            working_dir,
            os.path.split(working_dir)[-1]))
    Logger.debug("Set up default logging")

    # Report arguments parsed
    Logger.info(f"Initialising backfiller in {working_dir}")
    Logger.info(f"Using seed ids {expected_seed_ids}")
    Logger.info(f"Detection parameters: threshold={threshold}, threshold_type={threshold_type}, "
                f"trig_int={trig_int}, peak_cores={peak_cores}, cores={cores}, "
                f"parallel_processing={parallel_processing}, process_cores={process_cores}, "
                f"log_to_screen={log_to_screen}, starttime={starttime}, endtime={endtime}")

    # Read in tribe
    with open("tribe.pkl", "rb") as f:
        new_tribe = pickle.load(f)
    # new_tribe = Tribe().read("tribe.tgz")
    # Set up LocalClient in streams folder
    st_client = LocalClient("streams")
    # st_filename = "stream.ms"  # Avoid reading in whole stream - expensive in memory
    # Remove nan-channels that might have been added if the templates are already in use
    new_tribe.templates = [_rm_nan(t) for t in new_tribe.templates]
    # Squash duplicate channels to avoid excessive channels
    new_tribe.templates = [squash_duplicates(t) for t in new_tribe.templates]
    # Reshape
    new_tribe.templates = reshape_templates(
        templates=new_tribe.templates, used_seed_ids=expected_seed_ids)
    Logger.info(f"Backfilling with {len(new_tribe)} templates")
    Logger.debug("Additional templates to be run: \n{0} "
                 "templates".format(len(new_tribe)))

    Logger.info("Starting backfill detection run using:")
    Logger.info(st_client._waveform_db.keys())
    # Break into chunks so that detections can be handled as they happen
    # Can't actually start before the data start
    starttime = UTCDateTime(st_client.starttime)
    # Can't actually end after the data end!
    endtime = UTCDateTime(st_client.endtime)
    if endtime - starttime < minimum_data_for_detection:
        Logger.warning(f"Insufficient data between {starttime} and {endtime}. "
                       f"Need {minimum_data_for_detection}s of data")
        return
    # Try to send off twice the minimum data to allow for overlaps.
    # EQcorrscan overlaps data by the maximum moveout in the templates
    overlap = max(_moveout(template.st) for template in new_tribe)

    # We need to make sure our data are a multiple of the
    # minimum_data_for_detection, otherwise data at the end are dropped. This
    # often leads to missing self-detections.
    total_seconds = endtime - starttime
    chunk_len = (2 * minimum_data_for_detection) - overlap

    # Work out start and end times for chunks that maximize our data at the end
    _starttime, _endtime = endtime - chunk_len, endtime
    chunk_times = []
    while _starttime >= starttime:
        chunk_times.append((_starttime, _endtime))
        _starttime -= minimum_data_for_detection
        _endtime -= minimum_data_for_detection

    # Add in an additional chukn that overlaps with the first chunk if we are
    # missing inital data - we cope by declusetring later
    if chunk_times[-1][0] > starttime:
        chunk_times.append((starttime, starttime + chunk_len))

    # Go forward in time because it makes more sense...?
    chunk_times = chunk_times[::-1]

    new_parties = []
    for _starttime, _endtime in chunk_times:
        Logger.info(f"Running between {_starttime} and {_endtime}")
        st_chunk = st_client.get_waveforms(
            network="*", station="*", location="*", channel="*",
            starttime=_starttime, endtime=_endtime)
        if st_chunk is None or len(st_chunk) == 0:
            Logger.error(f"No data between {_starttime} and {_endtime}")
            _starttime += minimum_data_for_detection
            _endtime += minimum_data_for_detection
            continue
        st_chunk.merge()
        # TODO: Check data integrity here to avoid length errors
        # st_chunk = read(st_filename, starttime=_starttime, endtime=_endtime).merge()
        Logger.info(f"Read in {st_chunk}")
        try:
            _party = new_tribe.detect(
                stream=st_chunk, plot=False, threshold=threshold,
                threshold_type=threshold_type, trig_int=trig_int,
                xcorr_func="numpy",
                concurrency="concurrent",
                peak_cores=peak_cores,
                cores=cores,
                parallel_process=parallel_processing,
                process_cores=process_cores, copy_data=False,
                ignore_bad_data=True,
                overlap=overlap,
                **kwargs)
            # Remove nan channels from templates - there are sometimes issues with
            for family in _party:
                family.template = _rm_nan(family.template)
            Logger.info(f"Backfiller made {len(_party)} detections between {_starttime} and {_endtime}")
            if len(_party):
                new_parties.append(_party)
        except Exception as e:
            Logger.critical(f"Uncaught error: {e}")
            Logger.error(traceback.format_exc())
            continue

        Logger.info(
            f"Backfill detection between {_starttime} and {_endtime} "
            f"completed - handling detections")

        # Clear up un-needed objects
        del st_chunk
        gc.collect()
        # Memory output for debugging memory leaks
        # sum1 = summary.summarize(muppy.get_objects())
        # for line in summary.format_(sum1):
        #     Logger.info(line)
        total_memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        Logger.info(f"Total memory used by {os.getpid()}: {total_memory_mb:.2f} MB")
    # Make a single party from all these parties
    new_party = Party()
    for _party in new_parties:
        new_party += _party
    # Because of overlap we need to decluster
    new_party = new_party.decluster(trig_int=trig_int)
    new_party.families = [f for f in new_party if len(f)]

    Logger.info(f"Handling {len(new_party)} detections")
    os.makedirs("detections")
    fig = plt.Figure()
    for family in new_party:
        for detection in family:
            detection._calculate_event(template=family.template)
            det_starttime = min(p.time for p in detection.event.picks)
            det_endtime = max(p.time for p in detection.event.picks)
            det_starttime -= 20
            det_endtime += 20
            st = st_client.get_waveforms(
                "*", "*", "*", "*", det_starttime, det_endtime)
            fig = _write_detection(
                detection=detection,
                detect_file_base=_detection_filename(
                    detection=detection, detect_directory="detections"),
                save_waveform=save_waveforms, plot_detection=plot_detections,
                stream=st, fig=fig)

    if len(new_party):
        new_party.write(f"{working_dir}/party.tgz")

    Logger.info("Backfiller completed")
    return


def _rm_nan(template: Template) -> Template:
    """
    Remove nan channels in template
    """
    template.st = Stream(
        [t for t in template.st if not np.all(np.isnan(t.data))])
    return template


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to spin-up a backfiller instance")

    parser.add_argument(
        "-w", "--working-dir", type=str,
        help="Working directory containing tribe, stream, config "
             "and location to store temporary files")
    parser.add_argument(
        "-s", "--expected-seed-ids", nargs='+', required=True,
        help="Space separated list of expected seed-ids")
    parser.add_argument(
        "-m", "--minimum-data-for-detection", type=float, required=True,
        help="Minimum length of data for detection in seconds")
    parser.add_argument(
        "-t", "--threshold", type=float, required=True,
        help="Threshold value for detection")
    parser.add_argument(
        "-T", "--threshold-type", type=str, required=True,
        help="Threshold type")
    parser.add_argument(
        "-i", "--trig-int", type=float, required=True,
        help="Minimum trigger interval in seconds")
    parser.add_argument(
        "-p", "--peak-cores", type=int, required=False, default=1,
        help="Number of cores to use for peak finding")
    parser.add_argument(
        "-c", "--cores", type=int, required=False, default=None,
        help="Cores to use for correlation")
    parser.add_argument(
        "-P", "--parallel-processing", action="store_true",
        help="Flag to enable parallel processing of waveform data")
    parser.add_argument(
        "-C", "--process-cores", type=int, required=False, default=None,
        help="Number of cores to use for parallel processing - only enabled with -P")
    parser.add_argument(
        "-l", "--log-to-screen", action="store_true",
        help="Whether to log to screen or not, defaults to False")
    parser.add_argument(
        "--starttime", type=UTCDateTime, required=False,
        help="Starttime as UTCDateTime parsable string"
    )
    parser.add_argument(
        "--save-waveforms", action="store_true",
        help="Falg to save waveforms from detected events")
    parser.add_argument(
        "--endtime", type=UTCDateTime, required=False,
        help="Endtime as UTCDateTime parsable string"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Flag to turn on detections plotting"
    )

    args = parser.parse_args()

    working_dir = os.path.abspath(args.working_dir)

    backfill(working_dir=working_dir, expected_seed_ids=args.expected_seed_ids,
             minimum_data_for_detection=args.minimum_data_for_detection,
             threshold=args.threshold, threshold_type=args.threshold_type,
             trig_int=args.trig_int, peak_cores=args.peak_cores,
             cores=args.cores, parallel_processing=args.parallel_processing,
             process_cores=args.process_cores,
             log_to_screen=args.log_to_screen, starttime=args.starttime,
             endtime=args.endtime, plot_detections=args.plot,
             save_waveforms=args.save_waveforms)
