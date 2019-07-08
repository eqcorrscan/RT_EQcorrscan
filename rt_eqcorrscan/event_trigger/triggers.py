"""
Standardised trigger functions for use with a Reactor.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

from obspy.core.event import Catalog, Event
from obspy.geodetics import locations2degrees

from rt_eqcorrscan.event_trigger.listener import event_time


def get_nearby_events(
        event: Event,
        catalog: Catalog,
        radius: float
) -> Catalog:
    """
    Get a catalog of events close to another event.

    Parameters
    ----------
    event:
        Central event to calculate distance relative to
    catalog:
        Catalog to extract events from
    radius:
        Radius around `event` in km

    Returns
    -------
    Catalog of events close to `event`
    """
    sub_catalog = Catalog(
        [e for e in catalog.events
         if inter_event_distance(event, e) <= radius])
    return sub_catalog


def magnitude_rate_trigger_func(
        catalog: Catalog,
        magnitude_threshold: float = 5.5,
        rate_threshold: float = 20.,
        rate_bin: float = .2,
) -> Catalog:
    """
    Function to turn triggered response on based on magnitude and rate.

    Parameters
    ----------
    catalog:
        Catalog to look in
    magnitude_threshold:
        magnitude threshold for triggering a response
    rate_threshold:
        rate in events per day for triggering a response
    rate_bin:
        radius in degrees to calculate rate for.

    Returns
    -------
    The events that forced the trigger.
    """
    trigger_events = Catalog()
    for event in catalog:
        try:
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
        except IndexError:
            continue
        if magnitude.mag >= magnitude_threshold:
            trigger_events.events.append(event)
    for event in catalog:
        sub_catalog = get_nearby_events(event, catalog, radius=rate_bin)
        rate = average_rate(sub_catalog)
        if rate >= rate_threshold:
            for _event in sub_catalog:
                if _event not in trigger_events:
                    trigger_events.events.append(_event)
    if len(trigger_events) > 0:
        return trigger_events
    return []


def inter_event_distance(
        event1: Event,
        event2: Event
) -> float:
    """
    Calculate the distance (in degrees) between two events

    returns
    -------
        distance in degrees between events
    """
    try:
        origin_1 = event1.preferred_origin() or event1.origins[0]
        origin_2 = event2.preferred_origin() or event2.origins[0]
    except IndexError:
        return 180.
    return locations2degrees(
        lat1=origin_1.latitude, long1=origin_1.longitude,
        lat2=origin_2.latitude, long2=origin_2.longitude)


def average_rate(catalog: Catalog) -> float:
    """
    Compute mean rate of occurrence of events in catalog.

    Parameters
    ----------
    catalog:
        Catalog of events

    Returns
    -------
    Average rate over duration of catalog.
    """
    if len(catalog) <= 1:
        return 0.
    event_times = [event_time(e) for e in catalog]
    rates = [event_times[i] - event_times[i - 1]
             for i in range(len(event_times) - 1)]
    return sum(rates) / len(rates)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
