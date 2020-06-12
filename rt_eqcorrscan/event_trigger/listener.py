"""
Listener ABC.
"""

import threading
import logging

import numpy as np
from obspy import UTCDateTime
from obspy.core.event import Event
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Union, Tuple

Logger = logging.getLogger(__name__)


EventInfo = namedtuple("EventInfo", ("event_id", "event_time"))


class _Listener(ABC):
    """
    Abstract base class for listener objects - anything to be used by the
    Reactor should fit in this scope.
    """
    busy = False

    threads = []
    client = None
    # TODO: old_events should be a property with getter and setter methods that are threadsafe!
    _old_events = []  # List of tuples of (event_id, event_time)
    keep = 86400  # Time in seconds to keep old events
    lock = threading.Lock()  # Lock for access to old_events

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Run the listener """

    def get_old_events(self) -> List[EventInfo]:
        """ Threadsafe access to underlying list of tuples of old-events. """
        with self.lock:
            old_events = self._old_events
        return old_events

    def set_old_events(self, events: List[EventInfo]):
        with self.lock:
            self._old_events = events

    old_events = property(fget=get_old_events, fset=set_old_events)

    def remove_old_event(self, event: EventInfo):
        with self.lock:  # Make threadsafe
            self._old_events.remove(event)

    def extend(self, events: Union[EventInfo, List[EventInfo]]):
        """ Threadsafe way to add events to the cache """
        if isinstance(events, EventInfo):
            events = [events]
        with self.lock:
            self._old_events.extend(events)

    def _remove_old_events(self, endtime: UTCDateTime) -> None:
        """
        Expire old events from the cache.

        Parameters
        ----------
        endtime
            The time to calculate time-difference relative to. Any events
            older than endtime - self.keep will be removed.
        """
        if len(self.old_events) == 0:
            return
        time_diffs = np.array([endtime - tup[1] for tup in self.old_events])
        filt = time_diffs <= self.keep
        # Need to remove in-place, without creating a new list
        for i, old_event in enumerate(list(self.old_events)):
            if not filt[i]:
                self.remove_old_event(old_event)

    def background_run(self, *args, **kwargs):
        self.busy = True
        listening_thread = threading.Thread(
            target=self.run, args=args, kwargs=kwargs,
            name="ListeningThread")
        listening_thread.daemon = True
        listening_thread.start()
        self.threads.append(listening_thread)
        Logger.info("Started listening to {0}".format(self.client.base_url))

    def background_stop(self):
        self.busy = False
        for thread in self.threads:
            thread.join(timeout=10)
            if thread.is_alive():
                # Didn't join within timeout...
                thread.join()


def event_time(event: Event) -> UTCDateTime:
    """
    Get the origin or first pick time of an event.

    Parameters
    ----------
    event:
        Event to get a time for

    Returns
    -------
    Reference time for event.
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
