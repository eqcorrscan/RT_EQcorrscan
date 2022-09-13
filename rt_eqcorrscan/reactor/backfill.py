#!/usr/bin/env python
"""
Functions for spinning up and running a backfiller to detect with new templates in old data.
"""

import os
import logging
import gc

# Memory tracking for debugging
import psutil
from pympler import summary, muppy

from obspy import read

from eqcorrscan import Tribe, Party
from rt_eqcorrscan import Config
from rt_eqcorrscan.rt_match_filter import squash_duplicates, reshape_templates


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
                f"log_to_screen={log_to_screen}")

    # Read in tribe
    new_tribe = Tribe().read("tribe.tgz")
    st_filename = "stream.ms"
    # Squash duplicate channels to avoid excessive channels
    new_tribe.templates = [squash_duplicates(t) for t in new_tribe.templates]
    # Reshape
    new_tribe.templates = reshape_templates(
        templates=new_tribe.templates, used_seed_ids=expected_seed_ids)
    Logger.info(f"Backfilling with {len(new_tribe)} templates")
    Logger.debug("Additional templates to be run: \n{0} "
                 "templates".format(len(new_tribe)))

    # Get header info for stream
    st_head = read(st_filename, headonly=True)

    Logger.info("Starting backfill detection run with:")
    Logger.info(st_head.__str__(extended=True))
    # Break into chunks so that detections can be handled as they happen
    starttime, endtime = (
        min(tr.stats.starttime for tr in st_head),
        max(tr.stats.endtime for tr in st_head))
    del st_head  # Not needed anymore
    if endtime - starttime < minimum_data_for_detection:
        Logger.warning(f"Insufficient data between {starttime} and {endtime}. "
                       f"Need {minimum_data_for_detection}s of data")
        return
    # Try to send off twice the minimum data to allow for overlaps.
    _starttime, _endtime = (
        starttime, min(endtime, starttime + 2 * minimum_data_for_detection))
    new_party = Party()
    if _endtime >= (endtime + minimum_data_for_detection):
        Logger.warning("Insufficient data for backfill, not running")
        Logger.warning(f"{_endtime} >= {endtime + minimum_data_for_detection}")
    while _endtime < (endtime + minimum_data_for_detection):
        st_chunk = read(st_filename, starttime=_starttime, endtime=_endtime)
        try:
            new_party += new_tribe.detect(
                stream=st_chunk, plot=False, threshold=threshold,
                threshold_type=threshold_type, trig_int=trig_int,
                xcorr_func="numpy",
                concurrency="concurrent",
                peak_cores=peak_cores,
                cores=cores,
                parallel_process=parallel_processing,
                process_cores=process_cores, copy_data=False,
                **kwargs)
            Logger.info(f"Backfiller made {len(new_party)} detections between {_starttime} and {_endtime}")
        except Exception as e:
            Logger.error(e)
            _starttime += minimum_data_for_detection
            _endtime += minimum_data_for_detection
            continue

        Logger.info(
            f"Backfill detection between {_starttime} and {_endtime} "
            f"completed - handling detections")
        _starttime += minimum_data_for_detection
        _endtime += minimum_data_for_detection

        # Clear up un-needed objects
        del st_chunk
        gc.collect()
        # Memory output for debugging memory leaks
        sum1 = summary.summarize(muppy.get_objects())
        for line in summary.format_(sum1):
            Logger.info(line)
        total_memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        Logger.info(f"Total memory used by {os.getpid()}: {total_memory_mb:.2f} MB")
    new_party.write(f"{working_dir}/party.tgz")
    Logger.info("Backfiller completed")
    return


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

    args = parser.parse_args()

    backfill(working_dir=args.working_dir, expected_seed_ids=args.expected_seed_ids,
             minimum_data_for_detection=args.minimum_data_for_detection,
             threshold=args.threshold, threshold_type=args.threshold_type,
             trig_int=args.trig_int, peak_cores=args.peak_cores,
             cores=args.cores, parallel_processing=args.parallel_processing,
             process_cores=args.process_cores, log_to_screen=args.log_to_screen)
