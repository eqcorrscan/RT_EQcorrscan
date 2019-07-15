#!/usr/bin/env python3
"""
Script to run the real-time matched filter for a given region or earthquake.

"""

import logging

from obspy import UTCDateTime

from rt_eqcorrscan.config import read_config
from rt_eqcorrscan.reactor import estimate_region, get_inventory
from rt_eqcorrscan.database import TemplateBank, check_tribe_quality
from rt_eqcorrscan.rt_match_filter import RealTimeTribe


Logger = logging.getLogger("real-time-mf")


def run_real_time_matched_filter(**kwargs):
    config = read_config(config_file=kwargs.get("config_file", None))
    debug = kwargs.get("debug", False)
    if debug:
        config.log_level = "DEBUG"
        print("Using the following configuration:\n{0}".format(config))
    config.setup_logging()
    Logger.debug("Running in debug mode - expect lots of output!")

    client = config.rt_match_filter.get_client()

    triggering_eventid = kwargs.get("eventid", None)

    if triggering_eventid:
        triggering_event = client.get_events(
            eventid=triggering_eventid)[0]
        region = estimate_region(triggering_event)
    else:
        triggering_event = None
        region = {
            "latitude": kwargs.get("latitude", None),
            "longitude": kwargs.get("longitude", None),
            "maxradius": kwargs.get("maxradius", None)}
    starttime = kwargs.get("starttime", None)
    endtime = kwargs.get("endtime", None)
    if starttime is not None:
        region.update({"starttime": starttime})
    if endtime is not None:
        region.update({"endtime": endtime})
    bank = TemplateBank(
        config.database_manager.event_path,
        event_name_structure=config.database_manager.event_name_structure,
        event_format=config.database_manager.event_format,
        path_structure=config.database_manager.path_structure,
        event_ext=config.database_manager.event_ext)
    df = bank.get_event_summary(**region)
    Logger.info("{0} events within region".format(len(df)))
    Logger.debug("Region: {0}".format(region))
    Logger.info("Reading in Tribe")

    tribe = bank.get_templates(**region)

    Logger.info("Read in tribe of {0} templates".format(len(tribe)))

    _detection_starttime = UTCDateTime.now()
    inventory = get_inventory(
        client, tribe, triggering_event=triggering_event,
        location=region, starttime=_detection_starttime,
        max_distance=1000, n_stations=10)

    used_channels = {
        "{net}.{sta}.{loc}.{chan}".format(
            net=net.code, sta=sta.code, loc=chan.location_code, chan=chan.code)
        for net in inventory for sta in net for chan in sta}

    tribe = check_tribe_quality(
        tribe=tribe, seed_ids=used_channels,
        min_stations=config.database_manager.min_stations)

    Logger.info("After some QC there are {0} templates in the Tribe".format(
        len(tribe)))

    for t in tribe:
        t.process_length = config.rt_match_filter.buffer_capacity

    real_time_tribe = RealTimeTribe(
        tribe=tribe, inventory=inventory,
        server_url=config.rt_match_filter.seedlink_server_url,
        buffer_capacity=config.rt_match_filter.buffer_capacity,
        detect_interval=config.rt_match_filter.detect_interval,
        plot=config.rt_match_filter.plot,
        plot_options=config.plot)

    party = None
    try:
        party = real_time_tribe.run(
            threshold=config.rt_match_filter.threshold,
            threshold_type=config.rt_match_filter.threshold_type,
            trig_int=config.rt_match_filter.trig_int)
    except KeyboardInterrupt as e:
        Logger.error(e)
    finally:
        real_time_tribe.stop()
    return party


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real Time Matched Filter")
    parser.add_argument(
        "--eventid", "-e", type=str, help="Triggering event ID",
        required=False)
    parser.add_argument(
        "--latitude", type=float, help="Latitude for template-search",
        required=False)
    parser.add_argument(
        "--longitude", type=float, help="Longitude for template-search",
        required=False)
    parser.add_argument(
        "--radius", type=float, help="Radius (in degrees) for template-search",
        required=False)
    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file",
        required=False)
    parser.add_argument(
        "--starttime", type=str, required=False,
        help="Start-time as UTCDateTime parable string to collect templates "
             "from")
    parser.add_argument(
        "--endtime", type=str, required=False,
        help="End-time as UTCDateTime parable string to collect templates "
             "up to.")
    parser.add_argument(
        "--debug", action="store_true", help="Flag to set log level to debug")

    args = parser.parse_args()
    if args.eventid is not None:
        kwargs = {"eventid": args.eventid}
    elif args.latitude is not None:
        assert args.longitude is not None, "Latitude, longitude and radius must all be specified"
        assert args.radius is not None, "Latitude, longitude and radius must all be specified"
        kwargs = {"latitude": args.latitude, "longitude": args.longitude,
                  "maxradius": args.radius}
    else:
        raise NotImplementedError(
            "Needs either an event id or a geographic search")
    if args.starttime is not None:
        kwargs.update({"starttime": UTCDateTime(args.starttime)})
    if args.endtime is not None:
        kwargs.update({"endtime": UTCDateTime(args.endtime)})

    kwargs.update({"config_file": args.config, "debug": args.debug})

    run_real_time_matched_filter(**kwargs)
