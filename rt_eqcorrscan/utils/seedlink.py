"""
Data handling for seedlink clients for real-time matched-filter detection.

:copyright:
    Calum Chamberlain

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import threading

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy import Stream

from rt_eqcorrscan.utils.debug_log import verbose_print


class RealTimeClient(EasySeedLinkClient):
    """
    SeedLink client link for Real-Time Matched-Filtering.
    """
    busy = False

    def __init__(self, server_url, autoconnect=True, buffer=Stream(),
                 buffer_capacity=600, verbosity=0):
        super().__init__(server_url=server_url, autoconnect=autoconnect)
        self.buffer = buffer
        self.buffer_capacity = buffer_capacity
        self.verbosity = verbosity
        verbose_print("Instantiated RealTime client: {0}".format(self),
                      1, self.verbosity)

    def __repr__(self):
        """
        Print information about the client.

        .. rubric:: Example

        >>> client = RealTimeClient(server_url="geofon.gfz-potsdam.de")
        >>> print(client) # doctest: +NORMALIZE_WHITESPACE
        Seed-link client at geofon.gfz-potsdam.de, buffer capacity: 600s
            Current Buffer:
        0 Trace(s) in Stream:
        <BLANKLINE>
        """
        print_str = (
            "Seed-link client at {0}, buffer capacity: {1}s\n"
            "\tCurrent Buffer:\n{2}".format(
                self.server_hostname, self.buffer_capacity, self.buffer))
        return print_str

    def buffer_full(self):
        if len(self.buffer) == 0:
            return False
        for tr in self.buffer:
            if tr.stats.endtime - tr.stats.starttime < self.buffer_capacity:
                return False
        return True

    def buffer_length(self):
        """
        Return the maximum length of the buffer
        """
        return (max([tr.stats.endtime for tr in self.buffer]) -
                min([tr.stats.starttime for tr in self.buffer]))

    def _bg_run(self):
        while self.busy:
            self.run()

    def background_run(self):
        """Run the seedlink client in the background."""
        self.busy = True
        thread = threading.Thread(target=self._bg_run)
        thread.daemon = True
        thread.start()

    def background_stop(self):
        """Stop the background thread."""
        self.busy = False
        self.conn.terminate()
        self.close()

    def on_data(self, trace):
        """
        Handle incoming data
        :param trace:
        :return: buffer if full.
        """
        verbose_print("Packet: {0}".format(trace), 4, self.verbosity)
        self.buffer += trace
        self.buffer.merge()
        _tr = self.buffer.select(id=trace.id)[0]
        if _tr.stats.npts * _tr.stats.delta > self.buffer_capacity:
            verbose_print(
                "Trimming trace to {0}-{1}".format(
                    _tr.stats.endtime - self.buffer_capacity,
                    _tr.stats.endtime), 3, self.verbosity)
            _tr.trim(_tr.stats.endtime - self.buffer_capacity)
        else:
            verbose_print("Buffer contains {0}".format(self.buffer), 3,
                          self.verbosity)

    def on_terminate(self):  # pragma: no cover
        """
        Handle termination gracefully
        """
        verbose_print("Termination of {0}".format(self.__repr__()),
                      1, self.verbosity)
        return self.buffer

    def on_error(self):  # pragma: no cover
        """
        Handle errors gracefully.
        """
        verbose_print("SeedLink error", 1, self.verbosity)
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
