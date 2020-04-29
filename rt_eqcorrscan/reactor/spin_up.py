#!/usr/bin/env python
"""
Functions for spinning up and running a real-time tribe based on trigger-events
"""

import os
import logging

from collections import Counter
from typing import List, Union

from obspy import read_events, Inventory, UTCDateTime
from obspy.core.event import Event
from obspy.clients.fdsn.client import FDSNNoDataException
from obspy.geodetics import locations2degrees, kilometer2degrees
from eqcorrscan import Tribe

from rt_eqcorrscan import read_config, RealTimeTribe
from rt_eqcorrscan.database.database_manager import check_tribe_quality
from rt_eqcorrscan.event_trigger.listener import event_time


Logger = logging.getLogger(__file__)


def _read_event_list(fname: str) -> List[str]:
    with open(fname, "r") as f:
        event_ids = [line.strip() for line in f]
    return event_ids


def run(working_dir: str, cores: int = 1, log_to_screen: bool = False):
    os.chdir(working_dir)
    config = read_config('rt_eqcorrscan_config.yml')
    config.setup_logging(
        screen=log_to_screen, file=True,
        filename="{0}/rt_eqcorrscan_{1}.log".format(
            working_dir,
            os.path.split(working_dir)[-1]))
    triggering_event = read_events('triggering_event.xml')[0]
    min_stations = config.rt_match_filter.get("min_stations", None)
    tribe = Tribe().read("tribe.tgz")
    # Remove file to avoid re-reading it
    os.remove("tribe.tgz")

    Logger.info("Read in {0} templates".format(len(tribe)))
    if len(tribe) == 0:
        Logger.warning("No appropriate templates found")
        return
    Logger.info("Checking tribe quality: removing templates with "
                "fewer than {0} stations".format(min_stations))
    tribe = check_tribe_quality(
        tribe, min_stations=min_stations, **config.template)
    Logger.info("Tribe now contains {0} templates".format(len(tribe)))
    if len(tribe) == 0:
        return None, None

    client = config.rt_match_filter.get_client()
    rt_client = config.streaming.get_streaming_client()

    inventory = get_inventory(
        client, tribe, triggering_event=triggering_event,
        max_distance=config.rt_match_filter.max_distance,
        n_stations=config.rt_match_filter.n_stations)
    detect_interval = config.rt_match_filter.get(
        "detect_interval", 60)
    plot = config.rt_match_filter.get("plot", False)
    real_time_tribe = RealTimeTribe(
        tribe=tribe, inventory=inventory, rt_client=rt_client,
        detect_interval=detect_interval, plot=plot,
        plot_options=config.plot,
        name=triggering_event.resource_id.id.split('/')[-1])
    real_time_tribe._parallel_processing = False
    # Disable parallel processing for subprocess
    Logger.info("Created real-time tribe with inventory:\n{0}".format(
        inventory))

    # TODO: How will this work? Currently notifiers are not implemented
    # real_time_tribe.notifier = None

    real_time_tribe_kwargs = {
        "backfill_to": event_time(triggering_event) - 180,
        "backfill_client": config.rt_match_filter.get_waveform_client(),
        "cores": cores}

    real_time_tribe.run(
        threshold=config.rt_match_filter.threshold,
        threshold_type=config.rt_match_filter.threshold_type,
        trig_int=config.rt_match_filter.trig_int,
        min_stations=min_stations,
        keep_detections=86400,
        detect_directory="{name}/detections",
        plot_detections=config.rt_match_filter.plot_detections,
        save_waveforms=config.rt_match_filter.save_waveforms,
        max_run_length=config.rt_match_filter.max_run_length,
        minimum_rate=config.rt_match_filter.minimum_rate,
        **real_time_tribe_kwargs)


def get_inventory(
        client,
        tribe: Union[RealTimeTribe, Tribe],
        triggering_event: Event = None,
        location: dict = None,
        starttime: UTCDateTime = None,
        max_distance: float = 1000.,
        n_stations: int = 10,
        duration: float = 10,
        level: str = "channel",
        channel_list: Union[list, tuple] = ("EH?", "HH?"),
) -> Inventory:
    """
    Get a suitable inventory for a tribe - selects the most used, closest
    stations.


    Parameters
    ----------
    client:
        Obspy client with a get_stations service.
    tribe:
        Tribe or RealTimeTribe of templates to query for stations.
    triggering_event:
        Event with at least an origin to calculate distances from - if not
        specified will use `location`
    location:
        Dictionary with "latitude" and "longitude" keys - only used if
        `triggering event` is not specified.
    starttime:
        Start-time for station search - only used if `triggering_event` is
        not specified.
    max_distance:
        Maximum distance from `triggering_event.preferred_origin` or
        `location` to find stations. Units: km
    n_stations:
        Maximum number of stations to return
    duration:
        Duration stations must be active for. Units: days
    level:
        Level for inventory parsable by `client.get_stations`.
    channel_list
        List of channel-codes to be acquired.  If `None` then all channels
        will be searched.

    Returns
    -------
    Inventory of the most used, closest stations.
    """
    inv = Inventory(networks=[], source=None)
    if triggering_event is not None:
        try:
            origin = (
                triggering_event.preferred_origin() or
                triggering_event.origins[0])
        except IndexError:
            Logger.error("Triggering event has no origin")
            return inv
        lat = origin.latitude
        lon = origin.longitude
        _starttime = origin.time
    else:
        lat = location["latitude"]
        lon = location["longitude"]
        _starttime = starttime

    for channel_str in channel_list or ["*"]:
        try:
            inv += client.get_stations(
                startbefore=_starttime,
                endafter=_starttime + (duration * 86400),
                channel=channel_str, latitude=lat,
                longitude=lon,
                maxradius=kilometer2degrees(max_distance),
                level=level)
        except FDSNNoDataException:
            continue
    if len(inv) == 0:
        return inv
    # Calculate distances
    station_count = Counter(
        [pick.waveform_id.station_code for template in tribe
         for pick in template.event.picks])

    sta_dist = []
    for net in inv:
        for sta in net:
            dist = locations2degrees(
                lat1=lat, long1=lon, lat2=sta.latitude, long2=sta.longitude)
            sta_dist.append((sta.code, dist, station_count[sta.code]))
    sta_dist.sort(key=lambda _: (-_[2], _[1]))
    inv_out = inv.select(station=sta_dist[0][0])
    for sta in sta_dist[1:n_stations]:
        inv_out += inv.select(station=sta[0])
    return inv_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to spin-up a real-time matched-filter instance")

    parser.add_argument(
        "-w", "--working-dir", type=str,
        help="Working directory containing configuration file, list of "
             "templates and place to store temporary files")
    parser.add_argument(
        "-n", "--n-processors", type=int, default=1,
        help="Number of processors to use for detection")
    parser.add_argument(
        "-l", "--log-to-screen", action="store_true",
        help="Whether to log to screen or not, defaults to False")

    args = parser.parse_args()

    run(working_dir=args.working_dir, cores=args.n_processors,
        log_to_screen=args.log_to_screen)
