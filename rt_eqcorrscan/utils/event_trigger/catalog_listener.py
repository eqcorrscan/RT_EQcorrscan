"""
Listener to an event stream.
"""
import time
import logging

from obspy import UTCDateTime, Catalog

from rt_eqcorrscan.utils.event_trigger.listener import _Listener, event_time

Logger = logging.getLogger(__name__)


#TODO: The listener should add to the template DB!


class CatalogListener(_Listener):
    busy = False

    def __init__(self, client, catalog, catalog_lookup_kwargs,
                 interval=600, keep=86400):
        self.client = client
        self.catalog = catalog
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

    def run(self):
        if not self.busy:
            self.busy = True
        while self.busy:
            now = UTCDateTime.now()
            self.catalog.events = [
                e for e in self.catalog if event_time(e) >= now - self.keep]
            try:
                self.catalog += self.client.get_events(
                    starttime=self.previous_time, endtime=now,
                    **self.catalog_lookup_kwargs)
            except Exception as e:
                Logger.error(
                    "Could not download data between {0} and {1}".format(
                        self.previous_time, now))
                Logger.error(e)
                time.sleep(self.interval)
                continue

            self.previous_time = now
            time.sleep(self.interval)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
