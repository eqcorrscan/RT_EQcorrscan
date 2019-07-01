#!/usr/bin/env python3
"""
Script to run the real-time matched filter for a given region or earthquake.

    This file is part of rt_eqcorrscan.

    rt_eqcorrscan is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    rt_eqcorrscan is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with rt_eqcorrscan.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging

from concurrent.futures import ProcessPoolExecutor

from obspy import Stream

from rt_eqcorrscan.config.config import read_config
from rt_eqcorrscan.core.reactor import estimate_region, get_inventory
from rt_eqcorrscan.core.database_manager import TemplateBank
from rt_eqcorrscan.core.rt_match_filter import RealTimeTribe


Logger = logging.getLogger("real-time-mf")


def run_real_time_matched_filter(**kwargs):
    config = read_config(config_file=kwargs.get("config_file", None))
    config.setup_logging()

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
    bank = TemplateBank(
        config.database_manager.event_path,
        event_name_structure=config.database_manager.event_name_structure,
        event_format=config.database_manager.event_format,
        path_structure=config.database_manager.path_structure,
        event_ext=config.database_manager.event_ext)
    Logger.info("Reading in tribe")

    with ProcessPoolExecutor(max_workers=8) as executor:
        tribe = bank.get_templates(executor=executor, **region)

    Logger.info("Read in tribe of {0} templates".format(len(tribe)))

    inventory = get_inventory(
        client, tribe, triggering_event=triggering_event,
        max_distance=1000, n_stations=10)

    used_channels = {
        "{net}.{sta}.{loc}.{chan}".format(
            net=net.code, sta=sta.code, loc=chan.location_code, chan=chan.code)
        for net in inventory for sta in net for chan in sta}

    _templates = []
    for template in tribe:
        _st = Stream()
        for tr in template.st:
            if tr.id in used_channels:
                _st += tr
        template.st = _st
        t_stations = {tr.stats.station for tr in template.st}
        if len(t_stations) >= 5:
            _templates.append(template)
    tribe.templates = _templates

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

    kwargs.update({"config_file": args.config})

    run_real_time_matched_filter(**kwargs)
