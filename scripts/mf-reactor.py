#!/usr/bin/env python3
"""
Script to listen to an event service and start real-time matched-filter
detection when triggered.

"""

import logging
import faulthandler

faulthandler.enable()

from functools import partial
from obspy import Catalog

from rt_eqcorrscan.config import read_config
from rt_eqcorrscan.event_trigger import (
    magnitude_rate_trigger_func, CatalogListener)
from rt_eqcorrscan.reactor import Reactor
from rt_eqcorrscan.database import TemplateBank
from rt_eqcorrscan.streaming import RealTimeClient


Logger = logging.getLogger(__name__)


def run(**kwargs):
    config = read_config(config_file=kwargs.get("config_file", None))
    debug = kwargs.get("debug", False)
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
        event_name_structure=config.database_manager.event_name_structure,
        event_format=config.database_manager.event_format,
        path_structure=config.database_manager.path_structure,
        event_ext=config.database_manager.event_ext)

    listener = CatalogListener(
        client=client,
        catalog_lookup_kwargs=config.reactor.catalog_lookup_kwargs,
        template_bank=template_bank, interval=600, keep=86400.,
        catalog=Catalog(),
        waveform_client=config.rt_match_filter.get_waveform_client())
    rt_client = RealTimeClient(
        server_url=config.rt_match_filter.seedlink_server_url,
        buffer_capacity=config.rt_match_filter.buffer_capacity)

    reactor = Reactor(
        client=client, rt_client=rt_client,
        listener=listener, trigger_func=trigger_func,
        template_database=template_bank,
        template_lookup_kwargs=dict(
            starttime=config.database_manager.lookup_starttime),
        real_time_tribe_kwargs=config.rt_match_filter,
        plot_kwargs=config.plot,
        listener_kwargs=dict(
            min_stations=config.database_manager.min_stations,
            template_kwargs=config.template))
    reactor.run()
    return


if __name__ == "__main__":
    import argparse

    kwargs = {}
    parser = argparse.ArgumentParser(
        description="Run the RT_EQcorrscan Reactor")
    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file",
        required=False)
    parser.add_argument(
        "--debug", action="store_true", help="Flag to set log level to debug")

    args = parser.parse_args()

    kwargs.update({"debug": args.debug, "config_file": args.config})
    run(**kwargs)
