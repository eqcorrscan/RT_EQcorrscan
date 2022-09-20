#!/usr/bin/env python
"""
Functions to assist in synthesising real-time detection.

This script has 3 main steps:
    1. Generate a TemplateBank (in a real-situation this would just be updated);
    2. Read in an appropriate Tribe
    3. Start the RealTimeTribe
"""
import logging
import os

from obspy import UTCDateTime
from obspy.core.event import Event
from obspy.clients.fdsn import Client

from obsplus import WaveBank

from collections import namedtuple
from functools import partial

from rt_eqcorrscan import TemplateBank, Reactor, read_config
from rt_eqcorrscan.reactor import estimate_region
from rt_eqcorrscan.reactor.spin_up import get_inventory
from rt_eqcorrscan.config.config import Config
from rt_eqcorrscan.event_trigger import (
    CatalogListener, magnitude_rate_trigger_func)

Logger = logging.getLogger(__name__)

KNOWN_QUAKES = {
    "eketahuna": "2014p051675",
    "cook-strait": "2013p543824",  # M 6.5 preceeded by two 5.7 and 5.8
}

Region = namedtuple("Region", ["latitude", "longitude", "radius"])


def synthesise_real_time(
    triggering_event: Event,
    database_duration: float,
    config: Config,
    detection_runtime: float = 3600.0,
    make_templates: bool = True,
    speed_up: float = 1,
    debug: bool = False,
    query_interval: float = 60,
):
    """
    Synthesise a real-time matched-filter process for old data.

    Parameters
    ----------
    triggering_event:
        The Event that should trigger the system (must have happened in the
        past)
    database_duration:
        The duration to create the template database for in days prior to the
        triggering event
    config:
        Configuration for this synthesis
    detection_runtime:
        Maximum run-time for the detector in seconds
    make_templates:
        Whether templates need to be made or not.
    speed_up:
        Speed-up factor for detector - stream data faster than "real-time".
    debug:
        Whether to run logging in debug or not
    query_interval:
        How often to query the waveform server in seconds.  Smaller numbers
        will query more often, but this is limited by disk read speeds - make
        sure you don't go too small and make your system stall!
    """
    if debug:
        config.log_level = "DEBUG"
        print("Using the following configuration:\n{0}".format(config))
    config.setup_logging()

    client = config.rt_match_filter.get_client()

    trigger_origin = (
            triggering_event.preferred_origin() or triggering_event.origins[0])
    region = estimate_region(triggering_event)
    database_starttime = trigger_origin.time - (database_duration * 86400)
    database_endtime = trigger_origin.time

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
        executor=None)

    if make_templates:
        Logger.info("Downloading template events")
        catalog = client.get_events(
            starttime=database_starttime, endtime=database_endtime,
            **region)
        Logger.info(f"Downloaded {len(catalog)} events")
        Logger.info("Building template database")
        template_bank.make_templates(
            catalog=catalog, client=client, **config.template)
    else:
        template_bank.update_index()
    tribe = template_bank.get_templates(
        starttime=database_starttime, endtime=database_endtime, **region)
    inventory = get_inventory(
        client, tribe, triggering_event=triggering_event,
        max_distance=config.rt_match_filter.max_distance,
        n_stations=config.rt_match_filter.n_stations)

    config.plot.update({"offline": True})  # Use to use data time-stamps

    Logger.info("Downloading data")
    wavebank = WaveBank("simulation_wavebank")
    download_chunk_size = min(86400, detection_runtime)
    for network in inventory:
        for station in network:
            for channel in station:
                _starttime = trigger_origin.time - 60
                _endtime = _starttime + detection_runtime
                while _starttime <= _endtime:
                    Logger.info(
                        f"Downloading for {network.code}.{station.code}.{channel.location_code}.{channel.code} "
                        f"between {_starttime} and {_starttime + download_chunk_size}")
                    try:
                        st = client.get_waveforms(
                            network=network.code,
                            station=station.code,
                            channel=channel.code,
                            location=channel.location_code,
                            starttime=_starttime,
                            endtime=_starttime + download_chunk_size)
                    except Exception as e:
                        Logger.error(
                            "Could not download data for "
                            f"{network.code}.{station.code}."
                            f"{channel.location_code}.{channel.code}")
                        Logger.error(e)
                        _starttime += download_chunk_size
                        continue
                    _starttime += download_chunk_size
                    wavebank.put_waveforms(st)

    # Set up config to use the wavebank rather than FDSN.
    config.streaming.update(
        {"rt_client_url": str(wavebank.bank_path),
         "rt_client_type": "obsplus",
         "starttime": trigger_origin.time - 60,
         "speed_up": speed_up,
         "query_interval": 1.0})

    listener = CatalogListener(
        client=client, catalog_lookup_kwargs=region,
        template_bank=template_bank, interval=query_interval, keep=86400,
        catalog=None, waveform_client=client)
    listener._speed_up = speed_up
    listener._test_start_step = UTCDateTime.now() - trigger_origin.time
    listener._test_start_step += 60  # Start up 1 minute before the event

    reactor = Reactor(
        client=client,
        listener=listener, trigger_func=trigger_func,
        template_database=template_bank, config=config)
    reactor._speed_up = speed_up
    reactor._test_start_step = listener._test_start_step
    Logger.info("Starting reactor")
    # reactor.run(max_run_length=config.reactor.max_run_length)
    reactor.run(max_run_length=detection_runtime)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate running a real-time matched-filter process for old data")
    parser.add_argument(
        "--quake", type=str, required=True,
        help="Earthquake to synthesise real-time, either the event-id or a "
             f"known key.\n Known events are: \n{KNOWN_QUAKES}")
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to configuration file", required=False)
    parser.add_argument(
        "--db-duration", type=int, default=365,
        help="Number of days to generate the database for prior to the chosen event")
    parser.add_argument(
        "--runtime", type=float, default=3600,
        help="Duration in seconds to run the detector for.")
    parser.add_argument(
        "--radius", type=float, default=0.5,
        help="Radius in degrees to build database for")
    parser.add_argument(
        "--client", type=str, required=True,
        help="Client to get data from, must have an FDSN waveform and event service")
    parser.add_argument(
        "--templates-made", action="store_false",
        help="Flag to not make new templates - use if re-running an old DB")
    parser.add_argument(
        "--debug", action="store_true",
        help="Flag to run in debug mode, with lots of output to screen")
    parser.add_argument(
        "--speed-up", type=float, default=1,
        help="Multiplier to speed-up RT match-filter by.")
    parser.add_argument(
        "--working-dir", type=str, default=".",
        help="Directory to work in - will move to this directory and put all output here."
    )
    
    args = parser.parse_args()

    try:
        client = Client(args.client.lower())
    except Exception as e:
        Logger.warning(e)
        client = Client(args.client)

    config = read_config(args.config)
    quake_id = KNOWN_QUAKES.get(args.quake, args.quake)
    trigger_event = client.get_events(eventid=quake_id)[0]

    # Move to working directory
    if not os.path.isdir(args.working_dir):
        Logger.info(f"Making directory {args.working_dir}")
        os.makedirs(args.working_dir)
    os.chdir(args.working_dir)

    synthesise_real_time(
        triggering_event=trigger_event, database_duration=args.db_duration,
        config=config, make_templates=args.templates_made,
        debug=args.debug, speed_up=args.speed_up,
        detection_runtime=args.runtime)


if __name__ == "__main__":
    main()
