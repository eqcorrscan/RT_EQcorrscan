#!/usr/bin/env python3
"""
Script to run the real-time matched filter for a given region or earthquake.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import logging

from obspy import UTCDateTime
from obsplus import WaveBank
from eqcorrscan import Party

from rt_eqcorrscan.config import read_config
from rt_eqcorrscan.reactor import estimate_region, get_inventory
from rt_eqcorrscan.database import TemplateBank, check_tribe_quality
from rt_eqcorrscan.database.client_emulation import ClientBank
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
    if kwargs.get("local_archive"):
        local_wave_bank = WaveBank(
            base_path=config.rt_match_filter.local_wave_bank)
        client = ClientBank(
            wave_bank=local_wave_bank, event_bank=client,
            station_bank=client)

    triggering_eventid = kwargs.get("eventid", None)

    if triggering_eventid:
        triggering_event = client.get_events(
            eventid=triggering_eventid)[0]
        region = estimate_region(triggering_event)
        tribe_name = triggering_eventid
    else:
        triggering_event = None
        region = {
            "latitude": kwargs.get("latitude", None),
            "longitude": kwargs.get("longitude", None),
            "maxradius": kwargs.get("maxradius", None)}
        tribe_name = "{0}_{1}_{2}".format(
            kwargs.get("latitude", "lat"),
            kwargs.get("longitude", "long"),
            kwargs.get("maxradius", "rad"))
    starttime = kwargs.get("starttime", None)
    endtime = kwargs.get("endtime", None)
    rt_client_starttime = kwargs.get("rt_client_starttime", None)
    if starttime is not None:
        region.update({"starttime": starttime})
    if endtime is not None:
        region.update({"endtime": endtime})
    elif rt_client_starttime is not None:
        region.update({"endtime": rt_client_starttime})
    bank = TemplateBank(
        base_path=config.database_manager.event_path,
        name_structure=config.database_manager.name_structure,
        event_format=config.database_manager.event_format,
        path_structure=config.database_manager.path_structure,
        event_ext=config.database_manager.event_ext)
    Logger.info("Region: {0}".format(region))
    df = bank.get_event_summary(**region)
    Logger.info("{0} events within region".format(len(df)))
    if len(df) == 0:
        return Party()
    Logger.info("Reading in Tribe")

    tribe = bank.get_templates(**region)

    Logger.info("Read in tribe of {0} templates".format(len(tribe)))

    _detection_starttime = rt_client_starttime or UTCDateTime.now()
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
        min_stations=config.database_manager.min_stations,
        **config.template)

    Logger.info("After some QC there are {0} templates in the Tribe".format(
        len(tribe)))

    if rt_client_starttime is not None:
        config.streaming.starttime = rt_client_starttime
        config.streaming.speed_up = kwargs.get("speed_up", 1.0)
        config.plot.offline = True
    rt_client = config.streaming.get_streaming_client()
    real_time_tribe = RealTimeTribe(
        tribe=tribe, inventory=inventory, rt_client=rt_client,
        detect_interval=config.rt_match_filter.detect_interval,
        plot=config.rt_match_filter.plot, name=tribe_name,
        plot_options=config.plot)
    try:
        real_time_tribe._speed_up = config.streaming.speed_up
    except AttributeError:
        real_time_tribe._speed_up = 1

    party = None
    try:
        party = real_time_tribe.run(**config.rt_match_filter)
    except KeyboardInterrupt as e:
        Logger.error(e)
    finally:
        real_time_tribe.stop()
    return party


def main():
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
        "--template-starttime", type=str, required=False,
        help="Start-time as UTCDateTime parsable string to collect templates "
             "from")
    parser.add_argument(
        "--template-endtime", type=str, required=False,
        help="End-time as UTCDateTime parsable string to collect templates "
             "up to.")
    parser.add_argument(
        "--starttime", type=str, required=False,
        help="Start-time for real-time simulation for past data")
    parser.add_argument(
        "--speed-up", type=float, required=False, default=1.0,
        help="Speed-up factor for past data - unused for real-time")
    parser.add_argument(
        "--debug", action="store_true", help="Flag to set log level to debug")
    parser.add_argument(
        "--local-archive", action="store_true",
        help="Flag to use a local archive for waveform data, defined in "
             "config file ")

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
    if args.template_starttime is not None:
        kwargs.update({"starttime": UTCDateTime(args.template_starttime)})
    if args.template_endtime is not None:
        kwargs.update({"endtime": UTCDateTime(args.template_endtime)})
    if args.starttime is not None:
        kwargs.update({"rt_client_starttime": UTCDateTime(args.starttime)})

    kwargs.update({"config_file": args.config, "debug": args.debug,
                   "local_archive": args.local_archive,
                   "speed_up": args.speed_up})

    run_real_time_matched_filter(**kwargs)


if __name__ == "__main__":
    main()
