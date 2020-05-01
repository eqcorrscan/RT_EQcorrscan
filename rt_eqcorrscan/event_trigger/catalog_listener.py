"""
Listener to an event stream.
"""
import time
import logging
import numpy as np
import copy

from typing import Union, Callable
from obspy import UTCDateTime, Catalog
from obspy.core.event import Event

from rt_eqcorrscan.event_trigger.listener import _Listener, event_time
from rt_eqcorrscan.database.database_manager import (
    TemplateBank, remove_unreferenced)
from rt_eqcorrscan.event_trigger.triggers import magnitude_rate_trigger_func

Logger = logging.getLogger(__name__)


def filter_events(
    events: Union[list, Catalog],
    min_stations: int,
    auto_picks: bool,
    auto_event: bool,
    event_type: Union[list, str],
    **kwargs,
) -> list:
    """
    Filter events from a catalog based on some quality attributes.

    Parameters
    ----------
    events:
        Events to apply filter to.
    min_stations:
        Minimum number of stations for event to be kept.
    auto_picks:
        Whether to keep automatic picks or not.
    auto_event:
        Whether to keep automatic events or not.
    event_type:
        Event types to keep.

    Returns
    -------
    Events that pass the criteria
    """
    if isinstance(events, Catalog):
        events_out = copy.deepcopy(events.events)
    else:
        events_out = copy.deepcopy(events)

    _events_out = []
    for i, ev in enumerate(events_out):
        ev, keep = _qc_event(
            ev, min_stations=min_stations, auto_picks=auto_picks,
            auto_event=auto_event, event_type=event_type
        )
        if keep:
            _events_out.append(ev)
    return _events_out


def _qc_event(
    event: Event,
    min_stations: int = None,
    auto_picks: bool = True,
    auto_event: bool = True,
    event_type: Union[list, str] = None,
) -> tuple:
    """
    QC an individual event - removes picks in place.

    Returns
    -------
    tuple of (event: Event, keep: bool)
    """
    if event_type is not None and isinstance(event_type, str):
        event_type = [event_type]
    if event_type is not None and event.event_type not in event_type:
        return event, False
    elif not auto_event:
        if "manual" not in [ori.evaluation_mode for ori in event.origins]:
            return event, False
    if not auto_picks:
        pick_ids_to_remove = [
            p.resource_id for p in event.picks
            if p.evaluation_mode == "automatic"
        ]
        # remove arrivals and amplitudes and station_magnitudes
        event.picks = [
            p for p in event.picks if p.resource_id not in pick_ids_to_remove
        ]
        event = remove_unreferenced(event)[0]
    stations = {p.waveform_id.station_code for p in event.picks}
    if len(stations) < min_stations:
        return event, False
    return event, True


# TODO: back-fill method - cope with updates to events.
class CatalogListener(_Listener):
    """
    Client query class for obspy clients with a `get_events` service.

    Parameters
    ----------
    client:
        Client to query - must have at least a `get_events` method.
    catalog:
        Catalog of past events - can be empty. Any new events will be compared
        to this catalog and only added to the template bank if they are not
        in the original catalog.
    catalog_lookup_kwargs:
        Dictionary of keyword arguments for `client.get_events`.
    template_bank:
        A Template database - new events will be added to this database.
    interval:
        Interval for querying the client in seconds. Note that rapid queries
        may not be more efficient, and will almost certainly piss off your
        provider.
    keep:
        Time in seconds to keep events for in the catalog in memory. Will not
        remove old events on disk. Use to reduce memory consumption.
    waveform_client
        Client with at least a `get_waveforms` and `get_waveforms_bulk` method.
        If this is None (default) then the `client` will be used.
    """
    busy = False
    _test_start_step = 0  # Number of seconds prior to `now` used for testing.
    _speed_up = 1  # Multiplier for query intervals, used for synthesising
                   # previous sequences, not general purpose.

    def __init__(
        self,
        client,
        template_bank: TemplateBank,
        catalog: Catalog = None,
        catalog_lookup_kwargs: dict = None,
        interval: float = 10,
        keep: float = 86400,
        waveform_client=None,
    ):
        self.client = client
        self.waveform_client = waveform_client or client
        if catalog is None:
            catalog = Catalog()
        self.old_events = [
            (ev.resource_id.id, event_time(ev)) for ev in catalog]
        self.template_bank = template_bank
        self.catalog_lookup_kwargs = catalog_lookup_kwargs or dict()
        self.interval = interval
        self.keep = keep
        self.threads = []
        self.triggered_events = Catalog()
        self.busy = False
        self.previous_time = UTCDateTime.now()

    def __repr__(self):
        """
        ..rubric:: Example
        >>> from obspy.clients.fdsn import Client
        >>> listener = CatalogListener(
        ...     client=Client("GEONET"), catalog=Catalog(),
        ...     catalog_lookup_kwargs=dict(
        ...         latitude=-45, longitude=175, maxradius=2),
        ...     template_bank=TemplateBank('.'))
        >>> print(listener) # doctest: +NORMALIZE_WHITESPACE
        CatalogListener(client=Client(http://service.geonet.org.nz),\
        catalog=Catalog(0 events), interval=10, **kwargs)
        """
        print_str = (
            "CatalogListener(client=Client({0}), catalog=Catalog({1} events), "
            "interval={2}, **kwargs)".format(
                self.client.base_url, len(self.old_events), self.interval))
        return print_str

    @property
    def sleep_interval(self):
        return self.interval / self._speed_up

    def run(
        self,
        make_templates: bool = True,
        template_kwargs: dict = None,
        min_stations: int = 0,
        auto_event: bool = True,
        auto_picks: bool = True,
        event_type: Union[list, str] = None,
        filter_func: Callable = None,
        **filter_kwargs,
    ) -> None:
        """
        Run the listener. New events will be added to the template_bank.

        Parameters
        ----------
        make_templates:
            Whether to add new templates to the database (True) or not.
        template_kwargs:
            Dictionary of keyword arguments for making templates, requires
            at-least: lowcut, highcut, samp_rate, filt_order, prepick, length,
            swin.
        min_stations:
            Minimum number of stations for an event to be added to the
            TemplateBank
        auto_event:
            If True, both automatic and manually reviewed events will be
            included. If False, only manually reviewed events will be included
        auto_picks:
            If True, both automatic and manual picks will be included. If False
            only manually reviewed picks will be included. Note that this is
            done **before** counting the number of stations.
        event_type
            List of event types to keep.
        filter_func
            Function used for filtering. If left as none, this will use the
            `catalog_listener.filter_events` function.
        filter_kwargs:
            If the `filter_func` has changed then this should be the
            additional kwargs for the user-defined filter_func.
        """
        self.busy = True
        self.previous_time -= self._test_start_step
        template_kwargs = template_kwargs or dict()
        loop_duration = 0  # Timer for loop, used in synthesising speed-ups
        while self.busy:
            tic = time.time()  # Timer for loop, used in synthesising speed-ups
            if self._test_start_step > 0:
                # Still processing past data
                self._test_start_step -= loop_duration * self._speed_up
                self._test_start_step += loop_duration
                # Account for UTCDateTime.now() already including loop_
                # duration once.
            elif self._test_start_step < 0:
                # We have gone into the future!
                raise NotImplementedError(
                    "Trying to access future data: spoilers not allowed")
            now = UTCDateTime.now() - self._test_start_step
            # Remove old events from dict
            self._remove_old_events(now)
            Logger.debug("Checking for new events between {0} and {1}".format(
                self.previous_time, now))
            try:
                # Check for new events - add in a pad of five times the
                # checking interval to ensure that slow-to-update events are
                # included.
                new_events = self.client.get_events(
                    starttime=self.previous_time - (5 * self.interval),
                    endtime=now, **self.catalog_lookup_kwargs)
            except Exception as e:
                if "No data available for request" in e.args[0]:
                    Logger.debug("No new data")
                else:  # pragma: no cover
                    Logger.error(
                        "Could not download data between {0} and {1}".format(
                            self.previous_time, now))
                    Logger.error(e)
                time.sleep(self.sleep_interval)
                toc = time.time()  # Timer for loop, used in synthesising speed-ups
                loop_duration = toc - tic
                continue
            if new_events is not None:
                if filter_func is not None:
                    new_events = filter_func(
                        new_events, min_stations=min_stations,
                        auto_picks=auto_picks, auto_event=auto_event,
                        event_type=event_type, **filter_kwargs)
                old_event_ids = [tup[0] for tup in self.old_events]
                new_events = Catalog(
                    [ev for ev in new_events if ev.resource_id
                     not in old_event_ids])
                Logger.info("{0} new events between {1} and {2}".format(
                    len(new_events), self.previous_time, now))
                if len(new_events) > 0:
                    Logger.info("Adding {0} new events to the database".format(
                        len(new_events)))
                    for event in new_events:
                        try:
                            origin = (
                                event.preferred_origin() or event.origins[0])
                        except IndexError:
                            continue
                        try:
                            magnitude = (
                                event.preferred_magnitude() or
                                event.magnitudes[0])
                        except IndexError:
                            continue
                        Logger.info(
                            "Event {0}: M {1:.1f}, lat: {2:.2f}, "
                            "long: {3:.2f}, depth: {4:.2f}km".format(
                                event.resource_id.id, magnitude.mag,
                                origin.latitude, origin.longitude,
                                origin.depth / 1000.))
                    event_info = [
                        (ev.resource_id.id, event_time(ev))
                        for ev in new_events]
                    if make_templates:
                        self.template_bank.make_templates(
                            new_events, client=self.waveform_client,
                            **template_kwargs)
                    else:
                        self.template_bank.put_events(new_events)
                    # Putting the events in the bank clears the catalog.
                    self.old_events.extend(event_info)
                    Logger.debug("Old events current state: {0}".format(
                        self.old_events))
            self.previous_time = now
            time.sleep(self.sleep_interval)
            toc = time.time()  # Timer for loop, used in synthesising speed-ups
            loop_duration = toc - tic


if __name__ == "__main__":
    import doctest

    doctest.testmod()
