"""
Listener to an event stream.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import time
import logging
import numpy as np
import copy

from typing import Union
from obspy import UTCDateTime, Catalog
from obspy.core.event import Event

from rt_eqcorrscan.event_trigger.listener import _Listener, event_time
from rt_eqcorrscan.database.database_manager import (
    TemplateBank, remove_unreferenced)

Logger = logging.getLogger(__name__)


def filter_events(
    events: Union[list, Catalog],
    min_stations: int = None,
    auto_picks: bool = True,
    auto_event: bool = True,
    event_type: Union[list, str] = None,
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
    filter_func:
        Function for filtering out events based on user-defined criteria.
    """
    busy = False
    filter_func = filter_events

    def __init__(
        self,
        client,
        catalog: Catalog,
        catalog_lookup_kwargs: dict,
        template_bank: TemplateBank,
        interval: float = 600,
        keep: float = 86400,
    ):
        self.client = client
        self.old_events = [
            (ev.resource_id.id, event_time(ev)) for ev in catalog]
        self.template_bank = template_bank
        self.catalog_lookup_kwargs = catalog_lookup_kwargs
        self.interval = interval
        self.keep = keep
        self.threads = []
        self.triggered_events = Catalog()
        self.busy = False
        self.previous_time = UTCDateTime.now()

    def __repr__(self):
        print_str = (
            "CatalogListener(client=Client({0}), catalog=Catalog({1} events), "
            "interval={2}, **kwargs)".format(
                self.client.base_url, len(self.catalog), self.interval))
        return print_str

    def _remove_old_events(self, endtime: UTCDateTime) -> None:
        """ Expire old events from the cache. """
        if len(self.old_events) == 0:
            return
        time_diffs = np.array([endtime - tup[1] for tup in self.old_events])
        filt = time_diffs <= self.keep
        self.old_events = self.old_events[filt]

    def run(
        self,
        min_stations: int = 0,
        auto_event: bool = True,
        auto_picks: bool = True,
        **filter_kwargs,
    ) -> None:
        """
        Run the listener. New events will be added to the template_bank.

        Parameters
        ----------
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
        filter_kwargs:
            If the `filter_func` has changed then this should be the
            additional kwargs for the user-defined filter_func.
        """
        if not self.busy:
            self.busy = True
        while self.busy:
            now = UTCDateTime.now()
            # Remove old events from dict
            self._remove_old_events(now)
            new_events = None
            try:
                new_events = self.client.get_events(
                    starttime=self.previous_time, endtime=now,
                    **self.catalog_lookup_kwargs)
            except Exception as e:
                Logger.error(
                    "Could not download data between {0} and {1}".format(
                        self.previous_time, now))
                Logger.error(e)
            if new_events is not None:
                self.filter_func(
                    new_events, min_stations=min_stations,
                    auto_picks=auto_picks, auto_event=auto_event,
                    **filter_kwargs)
                old_event_ids = [tup[0] for tup in self.old_events]
                new_events = Catalog(
                    [ev for ev in new_events if ev.resource_id
                     not in old_event_ids])
                if len(new_events) > 0:
                    self.template_bank.put_events(new_events)
                    self.template_bank.make_templates(new_events)
                    self.old_events.extend([
                        (ev.resource_id.id, event_time(ev))
                        for ev in new_events])
            self.previous_time = now
            time.sleep(self.interval)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
