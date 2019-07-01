"""
Data handling for seedlink clients for real-time matched-filter detection.

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
import threading
import logging

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy import Stream


Logger = logging.getLogger(__name__)

LOGGING_MAP = {
    'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING,
    'error': logging.ERROR, 'critical': logging.CRITICAL}


class RealTimeClient(EasySeedLinkClient):
    """
    SeedLink client link for Real-Time Matched-Filtering.
    """
    busy = False

    def __init__(self, server_url, autoconnect=True, buffer=Stream(),
                 buffer_capacity=600, log_level='warning'):
        try:
            logging.basicConfig(
                level=LOGGING_MAP[log_level],
                format="%(asctime)s   %(threadName)s\t%(levelname)s \t"
                       "%(message)s")
        except KeyError:
            print("log_level must be in {0}".format(LOGGING_MAP.keys()))
        super().__init__(server_url=server_url, autoconnect=autoconnect)
        self.buffer = buffer
        self.buffer_capacity = buffer_capacity
        self.threads = []
        Logger.info("Instantiated RealTime client: {0}".format(self))

    def __repr__(self):
        """
        Print information about the client.

        .. rubric:: Example

        >>> client = RealTimeClient(server_url="geofon.gfz-potsdam.de")
        >>> print(client) # doctest: +NORMALIZE_WHITESPACE
        Seed-link client at geofon.gfz-potsdam.de, status: Stopped, buffer \
        capacity: 600s
            Current Buffer:
        0 Trace(s) in Stream:
        <BLANKLINE>
        """
        status_map = {True: "Running", False: "Stopped"}
        print_str = (
            "Seed-link client at {0}, status: {1}, buffer capacity: {2}s\n"
            "\tCurrent Buffer:\n{3}".format(
                self.server_hostname, status_map[self.busy],
                self.buffer_capacity, self.buffer))
        return print_str

    @property
    def buffer_full(self):
        if len(self.buffer) == 0:
            return False
        for tr in self.buffer:
            if tr.stats.endtime - tr.stats.starttime < self.buffer_capacity:
                return False
        return True

    @property
    def buffer_length(self):
        """
        Return the maximum length of the buffer
        """
        return (max([tr.stats.endtime for tr in self.buffer]) -
                min([tr.stats.starttime for tr in self.buffer]))

    def get_stream(self):
        return self.buffer.copy()

    def _bg_run(self):
        while self.busy:
            self.run()

    def background_run(self):
        """Run the seedlink client in the background."""
        self.busy = True
        streaming_thread = threading.Thread(
            target=self._bg_run, name="StreamThread")
        streaming_thread.daemon = True
        streaming_thread.start()
        self.threads.append(streaming_thread)
        Logger.info("Started streaming")

    def background_stop(self):
        """Stop the background thread."""
        self.busy = False
        self.conn.terminate()
        self.close()
        for thread in self.threads:
            thread.join()

    def on_data(self, trace):
        """
        Handle incoming data
        :param trace:
        """
        logging.debug("Packet of {0} samples for {1}".format(
            trace.stats.npts, trace.id))
        self.buffer += trace
        self.buffer.merge()
        _tr = self.buffer.select(id=trace.id)[0]
        if _tr.stats.npts * _tr.stats.delta > self.buffer_capacity:
            Logger.debug(
                "Trimming trace to {0}-{1}".format(
                    _tr.stats.endtime - self.buffer_capacity,
                    _tr.stats.endtime))
            _tr.trim(_tr.stats.endtime - self.buffer_capacity)
        else:
            Logger.debug("Buffer contains {0}".format(self.buffer))

    def on_terminate(self):  # pragma: no cover
        """
        Handle termination gracefully
        """
        Logger.info("Termination of {0}".format(self.__repr__()))
        return self.buffer

    def on_error(self):  # pragma: no cover
        """
        Handle errors gracefully.
        """
        Logger.error("SeedLink error")
        pass

    def on_seedlink_error(self):
        self.on_error()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
