#!/usr/bin/env python3
"""
Build a template database based on configuration parameters.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import os
import logging
import faulthandler

faulthandler.enable()

from obspy import UTCDateTime
from concurrent.futures import ProcessPoolExecutor

from rt_eqcorrscan.config import read_config
from rt_eqcorrscan.database import TemplateBank


Logger = logging.getLogger(__name__)


def run(
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    max_workers: int = None,
    **kwargs
):
    config = read_config(config_file=kwargs.get("config_file", None))
    debug = kwargs.get("debug", False)
    working_dir = kwargs.get("working_dir", None)
    if debug:
        config.log_level = "DEBUG"
        print(f"Using the following configuration:\n{config}")
    config.setup_logging()
    Logger.debug("Running in debug mode - expect lots of output!")

    if working_dir:
        Logger.info(f"Changing to working directory: {working_dir}")
        os.chdir(working_dir)

    template_bank = TemplateBank(
        config.database_manager.event_path,
        name_structure=config.database_manager.name_structure,
        event_format=config.database_manager.event_format,
        path_structure=config.database_manager.path_structure,
        event_ext=config.database_manager.event_ext,
        executor=ProcessPoolExecutor(max_workers=max_workers))

    template_bank.pickle_templates(starttime=starttime, endtime=endtime)


def main():
    import argparse

    kwargs = {}
    parser = argparse.ArgumentParser(
        description="Pickle a TemplateBank")
    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file",
        required=False)
    parser.add_argument(
        "--debug", action="store_true", help="Flag to set log level to debug")
    parser.add_argument(
        "-s", "--starttime", type=str, default="1970-01-01",
        help="Starttime parsable by obspy's UTCDateTime to begin database from")
    parser.add_argument(
        "-e", "--endtime", type=str, default=None,
        help="Endtime parsable by obspy's UTCDateTime to end database at. "
             "Defaults to now")
    parser.add_argument(
        "-w", "--working-dir", type=str,
        help="Working directory - will change to this directory after reading "
             "config file. All paths must be correct for this working dir.")
    parser.add_argument(
        "-n", "--max-workers", type=int, default=None,
        help="Maximum workers for ProcessPoolExecutor, defaults to the number "
             "of cores on the machine")

    args = parser.parse_args()

    try:
        starttime = UTCDateTime(args.starttime)
    except Exception as e:
        Logger.error(e)
        raise NotImplementedError(
            f"Could not parse {args.starttime} to UTCDateTime")

    if args.endtime:
        try:
            endtime = UTCDateTime(args.endtime)
        except Exception as e:
            Logger.error(e)
            raise NotImplementedError(
                f"Could not parse {args.endtime} to UTCDateTime")
    else:
        endtime = UTCDateTime()

    kwargs.update({"debug": args.debug, "config_file": args.config, 
                   "working_dir": args.working_dir})
    run(starttime=starttime, endtime=endtime, 
        max_workers=args.max_workers, **kwargs)


if __name__ == "__main__":
    main()
