"""
Relocation workflow.

Steps:
1. Watch directory for detections
2. If new detections, read in to memory (event and waveform)
3. Run lag-calc against all old detections in memory that meet distance criteria
4. Append to dt.cc file
5. [Optional] Run initial location program (hyp by default)
6. Append to event.dat file
7. Run relocation code (growclust by default)
8. Output resulting catalogue to csv and json formatted Catalog
   (faster IO than quakeml)

Designed to be run as continuously running subprocess managed by rt_eqcorrscan
"""

import logging
import subprocess
import glob

from eqcorrscan.utils.catalog_to_dd import (
    _hypodd_event_str, write_station, _compute_dt_correlations)


def run_growclust():
    """ Run growclust """
    return


def run_hyp():
    """ Run hypocentre """
    return


def relocator(
    detect_dir: str,
):
    old_detect_files = []
    while True:
        detect_files = glob.glob(detect_dir)
        new_detect_files = [
            df for df in detect_files if df not in old_detect_files]
        if len(new_detect_files) == 0:
            continue  # Carry on waiting!

        # Read in detections

        # Run location code (hyp)

        # Compute correlations using _compute_dt_correlations
        # Note: maintain event id mapper in memory

        # Write event.dat (append to already existing file)

        # Write dt.cc (append to already existing file)

        # Run relocation code (growclust)

        # Write out current state of catalogue - try to not maintain too much in memory!

        # Update old events
        old_detect_files.extend(new_detect_files)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RT_EQcorrscan relocation plugin.")

    parser.add_argument(
        "-d", "--detect-dir", type=str, required=True,
        help="Detection directory to watch for new detections.")
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="Configuration file path.")

    args = parser.parse_args()

    relocator()