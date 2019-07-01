"""
Listener to an event stream.

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
import time
import logging

from obspy import UTCDateTime, Catalog

from rt_eqcorrscan.utils.event_trigger.listener import _Listener, event_time
from rt_eqcorrscan.core.database_manager import TemplateBank

Logger = logging.getLogger(__name__)


class CatalogListener(_Listener):
    busy = False

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
        self.catalog = catalog
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

    def run(self):
        if not self.busy:
            self.busy = True
        while self.busy:
            now = UTCDateTime.now()
            self.catalog.events = [
                e for e in self.catalog if event_time(e) >= now - self.keep]
            new_events = None
            try:
                new_events += self.client.get_events(
                    starttime=self.previous_time, endtime=now,
                    **self.catalog_lookup_kwargs)
            except Exception as e:
                Logger.error(
                    "Could not download data between {0} and {1}".format(
                        self.previous_time, now))
                Logger.error(e)
            if new_events is not None:
                self.template_bank.put_events(new_events)
                self.template_bank.make_templates(new_events)
                self.catalog += new_events
            self.previous_time = now
            time.sleep(self.interval)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
