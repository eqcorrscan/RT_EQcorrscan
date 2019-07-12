"""
Overarching tool for listening to and triggering from FDSN earthquakes.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging
import threading
import time

from collections import Counter
from typing import Union, Callable

from obspy import Inventory, UTCDateTime
from obspy.core.event import Event
from obspy.clients.fdsn.client import FDSNNoDataException
from obspy.geodetics import locations2degrees, kilometer2degrees

from eqcorrscan import Tribe, Party

from rt_eqcorrscan.database.database_manager import TemplateBank
from rt_eqcorrscan.rt_match_filter import RealTimeTribe
from rt_eqcorrscan.event_trigger.catalog_listener import CatalogListener


Logger = logging.getLogger(__name__)


class Reactor(object):
    """
    A class to listen to a client and start-up a real-time instance.

    The real-time instance will be triggered by the listener when set
    conditions are met. Appropriate templates will be extracted from the
    template database on disk and used a Tribe for the real-time detection.

    Once triggered, the listener will keep listening, but will not trigger in
    the same region again while the real-time detector is running. The real-time
    detector has to be stopped manually.

    Parameters
    ----------
    client
        An obspy or obsplus client that supports event and station queries.
    seedlink_server_url
        The url to a seedlink server that will be used to real-time
        matched-filtering
    listener
        Listener for checking current earthquake activity
    trigger_func:
        A function that returns a list of events that exceed some trigger
        parameters given a catalog. See note below.
    template_database
        A template database to be used to generate tribes for real-time
        matched-filter detection.
    listener_kwargs
        Dictionary of keyword arguments to be passed to listener.run
    real_time_tribe_kwargs
        Dictionary of keyword arguments for the real-time tribe. Any keys not
        included will be set to default values.
    plot_kwargs
        Dictionary of plotting keyword arguments - only required if `plot=True`
        in `real_time_tribe_kwargs`.

    Notes
    -----
    `trigger_func` must only take one argument, a catalog of events. To
    achieve this you should generate a partial function from your trigger
    function. For example, using the provided rate-and-magnitude triggering
    function:
    ```
        from rt_eqcorrscan.event_trigger import magnitude_rate_trigger_func
        from functools import partial

        trigger_func = partial(
            magnitude_rate_trigger_func, magnitude_threshold=4,
            rate_threshold=20, rate_bin=0.5)
    ```
    """
    triggered_events = []
    running_templates_ids = []  # A list of currently running templates
    max_station_distance = 1000
    n_stations = 10
    sleep_step = 10

    # The threads that are detecting away!
    detecting_threads = []

    def __init__(
        self,
        client,
        seedlink_server_url: str,
        listener: CatalogListener,
        trigger_func: Callable,
        template_database: TemplateBank,
        listener_kwargs: dict,
        real_time_tribe_kwargs: dict,
        plot_kwargs: dict,
    ):
        self.client = client
        self.seedlink_server_url = seedlink_server_url
        self.listener = listener
        self.trigger_func = trigger_func
        self.template_database = template_database
        self.real_time_tribe_kwargs = real_time_tribe_kwargs
        self.plot_kwargs = plot_kwargs
        self.listener_kwargs = listener_kwargs
        # Time-keepers
        self._run_start = None
        self.up_time = 0

    def get_up_time(self):
        return self._up_time

    def set_up_time(self, now):
        if self._run_start is not None:
            self._up_time = now - self._run_start
        else:
            self._up_time = 0

    up_time = property(get_up_time, set_up_time)

    def run(self, max_run_length: float = None) -> None:
        """Run all the processes."""
        self.listener.background_run(**self.listener_kwargs)
        self._run_start = UTCDateTime.now()
        # Query the catalog in the listener every so often and check
        while True:
            if len(self.listener.old_events) > 0:
                working_ids = list(zip(*self.listener.old_events))[0]
                working_cat = self.template_database.get_events(
                    eventid=working_ids)
            else:
                working_cat = []
            Logger.debug("Currently analysing a catalog of {0} events".format(
                len(working_cat)))

            trigger_events = self.trigger_func(working_cat)
            for trigger_event in trigger_events:
                if trigger_event not in self.triggered_events:
                    Logger.warning(
                        "Listener triggered by event {0}".format(
                            trigger_event))
                    self.triggered_events.append(trigger_event)
                    self.background_spin_up(trigger_event)
            self.set_up_time(UTCDateTime.now())
            if max_run_length and self.up_time >= max_run_length:
                Logger.info("Times up: Stopping")
                self.stop()
                break
            time.sleep(self.sleep_step)

    def background_spin_up(self, triggering_event: Event) -> None:
        """
        Spin up a detection run in a background (daemon) thread.

        Parameters
        ----------
        triggering_event
            Event that triggered this run - needs to have at-least an origin.
        """
        detecting_thread = threading.Thread(
            target=self.spin_up,
            args=(triggering_event, ), name="DetectingThread")
        detecting_thread.daemon = True
        detecting_thread.start()
        self.detecting_threads.append(detecting_thread)
        Logger.info("Started detecting")

    def stop(self) -> None:
        """Stop all the processes."""
        for detecting_thread in self.detecting_threads:
            detecting_thread.join()
        self.listener.background_stop()

    def spin_up(self, triggering_event: Event) -> Party:
        """
        Run the reactors response function.

        Parameters
        ----------
        triggering_event
            Event that triggered this run - needs to have at-least an origin.
        """
        region = estimate_region(triggering_event)
        if region is None:
            return
        tribe = self.template_database.get_templates(**region)
        tribe.templates = [t for t in tribe
                           if t.name not in self.running_templates_ids]
        inventory = get_inventory(
            self.client, tribe, triggering_event=triggering_event,
            max_distance=self.max_station_distance,
            n_stations=self.n_stations)
        buffer_capacity = self.real_time_tribe_kwargs.get(
            "buffer_capacity", 600)
        detect_interval = self.real_time_tribe_kwargs.get(
            "detect_interval", 60)
        plot = self.real_time_tribe_kwargs.get("plot", False)
        plot_length = self.real_time_tribe_kwargs.get(
            "plot_length", 300)

        real_time_tribe = RealTimeTribe(
            tribe=tribe, inventory=inventory,
            server_url=self.seedlink_server_url,
            buffer_capacity=buffer_capacity,
            detect_interval=detect_interval, plot=plot,
            plot_length=plot_length, **self.plot_kwargs)

        self.running_templates_ids.append(
            [t.name for t in real_time_tribe.templates])
        return real_time_tribe.run(**self.real_time_tribe_kwargs)


def get_inventory(
        client,
        tribe: Union[RealTimeTribe, Tribe],
        triggering_event: Event = None,
        location: dict = None,
        starttime: UTCDateTime = None,
        max_distance: float = 1000.,
        n_stations: int = 10,
        duration: float = 10,
        level: str = "channel"
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

    Returns
    -------
    Inventory of the most used, closest stations.
    """
    inv = Inventory(networks=[], source=None)
    if triggering_event is not None:
        try:
            origin = (
                triggering_event.preferred_origin() or
                triggering_event.origins[0]
            )
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

    for channel_str in ["EH?", "HH?"]:
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
    inv_len = 0
    for net in inv:
        inv_len += len(net)
    if inv_len <= n_stations:
        return [sta.code for net in inv for sta in net]
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


def estimate_region(event: Event, min_length: float = 50.) -> dict:
    """
    Estimate the region to find templates within given a triggering event.

    Parameters
    ----------
    event
        The event that triggered this function
    min_length
        Minimum length in km for diameter of event circle around the
        triggering event

    Returns
    -------
    Dictionary keyed by "latitude", "longitude" and "maxradius"

    Notes
    -----
    Uses a basic Wells and Coppersmith relation, scaled by 1.25 times.
    """
    from obspy.geodetics import kilometer2degrees
    try:
        origin = event.preferred_origin() or event.origins[0]
    except IndexError:
        Logger.error("Triggering event has no origin, not using.")
        return None

    try:
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
    except IndexError:
        Logger.warning("Triggering event has no magnitude, using minimum "
                       "length or {0}".format(min_length))
        magnitude = None
    if magnitude:
        length = 10 ** ((magnitude.mag - 5.08) / 1.16)  # Wells and Coppersmith
        # Scale up a bit - for Darfield this gave 0.6 deg, but the aftershock
        # region is more like 1.2 deg radius
        length *= 1.25
    else:
        length = min_length

    if length <= min_length:
        length = min_length
    length = kilometer2degrees(length)
    length /= 2.
    return {
        "latitude": origin.latitude, "longitude": origin.longitude,
        "maxradius": length}


if __name__ == "__main__":
    import doctest

    doctest.testmod()
