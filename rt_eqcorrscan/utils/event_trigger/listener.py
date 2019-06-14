"""
Listener to an event stream.
"""
import threading
import time
import logging

from obspy import UTCDateTime, Catalog
from obspy.geodetics import locations2degrees

Logger = logging.getLogger(__name__)


#TODO: The listener should add to the template DB!


def trigger_func(catalog, magnitude_threshold=5.5, rate_threshold=20.,
                 rate_bin=.2):
    """
    Function to turn triggered response on.

    :type catalog: `obspy.core.event.Catalog`
    :param catalog: Catalog to look in
    :type magnitude_threshold: float
    :param magnitude_threshold: magnitude threshold for triggering a response
    :type rate_threshold: float
    :param rate_threshold: rate in events per day for triggering a response
    :type rate_bin: float
    :param rate_bin: radius in degrees to calculate rate for.

    :rtype: `obspy.core.event.Event`
    :returns: The event that forced the trigger.
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


class CatalogListener(object):
    busy = False

    def __init__(self, client, catalog, callback_func, interval=600,
                 keep=86400, trig_func=trigger_func, **kwargs):
        self.client = client
        self.catalog = catalog
        self.kwargs = kwargs
        self.interval = interval
        self.keep = keep
        self.trigger_func = trig_func
        self.threads = []
        self.triggered_events = Catalog()
        self.callback_func = callback_func
        self.busy = False
        self.previous_time = UTCDateTime.now()

    def __repr__(self):
        print_str = (
            "CatalogListener(client=Client({0}), catalog=Catalog({1} events), "
            "interval={2}, **kwargs)".format(
                self.client.base_url, len(self.catalog), self.interval))
        return print_str

    def run(self):
        if not self.busy:
            self.busy = True
        while self.busy:
            now = UTCDateTime.now()
            self.catalog.events = [e for e in self.catalog
                                   if event_time(e) >= now - self.keep]
            try:
                self.catalog += self.client.get_events(
                    starttime=self.previous_time, endtime=now, **self.kwargs)
            except Exception as e:
                Logger.error(
                    "Could not download data between {0} and {1}".format(
                        self.previous_time, now))
                Logger.error(e)
                time.sleep(self.interval)
                continue
            Logger.debug("Currently analysing a catalog of {0} events".format(
                len(self.catalog)))

            trigger_events = self.trigger_func(self.catalog)
            for trigger_event in trigger_events:
                if trigger_event not in self.triggered_events:
                    Logger.warning(
                        "Listener triggered by event {0}".format(
                            trigger_event))
                    self.triggered_events.events.append(trigger_event)
                    self.callback_func(event=trigger_event)
            self.previous_time = now
            time.sleep(self.interval)

    def background_run(self):
        self.busy = True
        listening_thread = threading.Thread(
            target=self.run, name="ListeningThread")
        listening_thread.daemon = True
        listening_thread.start()
        self.threads.append(listening_thread)
        Logger.info("Started listening to {0}".format(self.client.base_url))

    def background_stop(self):
        self.busy = False
        for thread in self.threads:
            thread.join()


def event_time(event):
    """
    Get the origin or first pick time of an event.

    :param event:
    :return:
    """
    try:
        origin = event.preferred_origin() or event.origins[0]
    except IndexError:
        origin = None
    if origin is not None:
        return origin.time
    if len(event.picks) == 0:
        return UTCDateTime(0)
    return min([p.time for p in event.picks])


def get_nearby_events(event, catalog, radius):
    """
    Get a catalog of events close to another event.

    :type event: `obspy.core.event.Event`
    :param event: Central event to calculate distance relative to
    :type catalog: `obspy.core.event.Catalog`
    :param catalog: Catalog to extract events from
    :type radius: float
    :param radius: Radius around `event` in km

    :rtype: `obspy.core.event.Catalog`
    :return: Catalog of events close to `event`
    """
    sub_catalog = Catalog(
        [e for e in catalog.events
         if inter_event_distance(event, e) <= radius])
    return sub_catalog


def inter_event_distance(event1, event2):
    """
    Calculate the distance (in degrees) between two events

    :rtype: float
    :return: distance in degrees between events
    """
    try:
        origin_1 = event1.preferred_origin() or event1.origins[0]
        origin_2 = event2.preferred_origin() or event2.origins[0]
    except IndexError:
        return 180.
    return locations2degrees(
        lat1=origin_1.latitude, long1=origin_1.longitude,
        lat2=origin_2.latitude, long2=origin_2.longitude)


def average_rate(catalog):
    """
    Compute mean rate of occurrence of events in catalog.

    :type catalog: `obspy.core.event.Catalog`
    :param catalog: Catalog of events

    :rtype: float
    :return: rate
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
