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

from obspy import read_events, Catalog, read_inventory

from eqcorrscan.utils.catalog_to_dd import (
    _hypodd_event_str, write_station, _compute_dt_correlations)

from rt_eqcorrscan.config import read_config
from rt_eqcorrscan.plugins.plugin import Watcher
from rt_eqcorrscan.plugins.relocation.hyp_runner import (
    seisan_hyp, VelocityModel)


def run_growclust():
    """ Run growclust """
    return


def run_hyp():
    """ Run hypocentre """
    return


def relocator(
    detect_dir: str,
    config_file: str,
    inventory_file: str,
    velocity_file: str = "vmodel.txt"
):
    config = read_config(config_file=config_file)
    config.setup_logging(filename="rt_eqcorrscan_relocator.log")

    inv = read_inventory(inventory_file)
    vmodel = VelocityModel.read(velocity_file)

    watcher = Watcher(watch_pattern=f"{detect_dir}/*.xml")

    _remodel = True
    event_id_mapper = dict()
    while True:
        watcher.check_for_updates()

        if len(watcher.new) == 0:
            continue  # Carry on waiting!

        # Read in detections
        cat = Catalog()
        for ev_file in watcher.new:
            cat += read_events(ev_file)

        # Run location code (hyp)
        cat_located = Catalog()
        for event in cat:
            event_located = seisan_hyp(
                event=event, inventory=inv, velocities=vmodel.velocities,
                vpvs=vmodel.vpvs, remodel=_remodel, clean=False)
            # Only remodel the traveltime tables the first run.
            _remodel = False
            cat_located += event_located

        # Read in relevant waveforms? Maintain dict of paths to waveforms and event ids?

        # Compute correlations using _compute_dt_correlations
        # TODO: To do this efficiently we should have prepped waveforms in memory?


        # Write event.dat (append to already existing file)

        # Write dt.cc (append to already existing file)

        # Run relocation code (growclust)

        # Write out current state of catalogue - try to not maintain too much in memory!

        # Update old events
        watcher.processed(watcher.new)
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
    parser.add_argument(
        "-i", "--inventory", type=str, required=True,
        help="Inventory file path.")

    args = parser.parse_args()

    relocator(detect_dir=args.detect_dir, config_file=args.config,
              inventory_file=args.inventory)
