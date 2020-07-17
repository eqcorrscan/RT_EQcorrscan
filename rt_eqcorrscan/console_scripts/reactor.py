#!/usr/bin/env python3
"""
Script to listen to an event service and start real-time matched-filter
detection when triggered.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import logging
import faulthandler

faulthandler.enable()

from functools import partial
from concurrent.futures import ProcessPoolExecutor

from obspy import Catalog, UTCDateTime

from rt_eqcorrscan.config import read_config
from rt_eqcorrscan.event_trigger import (
    magnitude_rate_trigger_func, CatalogListener)
from rt_eqcorrscan.reactor import Reactor
from rt_eqcorrscan.database import TemplateBank


Logger = logging.getLogger(__name__)


def run(**kwargs):
    config = read_config(config_file=kwargs.get("config_file", None))
    debug = kwargs.get("debug", False)
    update_bank = kwargs.get("update_bank", True)
    listener_starttime = kwargs.get("listener_starttime", None)
    if debug:
        config.log_level = "DEBUG"
        print("Using the following configuration:\n{0}".format(config))
    config.setup_logging()
    Logger.debug("Running in debug mode - expect lots of output!")

    client = config.rt_match_filter.get_client()

    trigger_func = partial(
        magnitude_rate_trigger_func,
        magnitude_threshold=config.reactor.magnitude_threshold,
        rate_threshold=config.reactor.rate_threshold,
        rate_bin=config.reactor.rate_radius,
        minimum_events_in_bin=config.reactor.minimum_events_in_bin)

    template_bank = TemplateBank(
        config.database_manager.event_path,
        name_structure=config.database_manager.name_structure,
        event_format=config.database_manager.event_format,
        path_structure=config.database_manager.path_structure,
        event_ext=config.database_manager.event_ext,
        executor=ProcessPoolExecutor())

    if update_bank:
        Logger.info("Updating bank before running")
        template_bank.update_index()

    listener = CatalogListener(
        client=client,
        catalog_lookup_kwargs=config.reactor.catalog_lookup_kwargs,
        template_bank=template_bank, interval=600, keep=86400.,
        catalog=Catalog(),
        waveform_client=config.rt_match_filter.get_waveform_client())

    reactor = Reactor(
        client=client,
        listener=listener, trigger_func=trigger_func,
        template_database=template_bank, config=config,
        listener_starttime=listener_starttime)
    reactor.run(max_run_length=config.reactor.max_run_length)
    return


def main():
    import argparse

    kwargs = {}
    parser = argparse.ArgumentParser(
        description="Run the RT_EQcorrscan Reactor")
    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file",
        required=False)
    parser.add_argument(
        "--debug", action="store_true", help="Flag to set log level to debug")
    parser.add_argument(
        "-u", "--update-bank", action="store_true",
        help="Flag to update template bank index before running, use if events"
             " have been manually added")
    parser.add_argument(
        "-s", "--listener-starttime", type=UTCDateTime, 
        help="UTCDateTime parsable starttime for the listener - will collect "
             "events from this date to now and react to them.")

    args = parser.parse_args()

    kwargs.update({"debug": args.debug, "config_file": args.config,
                   "update_bank": args.update_bank, 
                   "listener_starttime": args.listener_starttime})
    run(**kwargs)


if __name__ == "__main__":
    main()
