"""
Listener ABC.
"""

import threading
import logging

from obspy import Catalog, UTCDateTime
from abc import ABC, abstractmethod


Logger = logging.getLogger(__name__)


class _Listener(ABC):
    """
    Abstract base class for listener objects - anything to be used by the
    Reactor should fit in this scope.
    """
    busy = False

    threads = []
    client = None
    catalog = Catalog()

    @abstractmethod
    def run(self, **kwargs):
        """ Run the listener """

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
