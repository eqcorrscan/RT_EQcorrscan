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
import time
import logging

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy import Stream, UTCDateTime

from rt_eqcorrscan.plotting.plot_buffer import PlotBuffer

SLEEP_STEP = 1.0
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
        logging.info("Instantiated RealTime client: {0}".format(self))

    def __repr__(self):
        """
        Print information about the client.

        .. rubric:: Example

        >>> client = RealTimeClient(server_url="geofon.gfz-potsdam.de")
        >>> print(client) # doctest: +NORMALIZE_WHITESPACE
        Seed-link client at geofon.gfz-potsdam.de, status: Stopped, buffer capacity: 600s
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

    def _bg_run(self):
        while self.busy:
            self.run()

    def background_run(self, plot=False, plot_length=600, ylimits=(-2, 2),
                       size=(6, 6)):
        """Run the seedlink client in the background."""
        self.busy = True
        streaming_thread = threading.Thread(
            target=self._bg_run, name="StreamThread")
        streaming_thread.daemon = True
        streaming_thread.start()
        self.threads.append(streaming_thread)
        logging.info("Started streaming")
        if plot:
            plotting_thread = threading.Thread(
                target=self._plot, name="PlotThread", kwargs={
                    'plot_length': plot_length, 'ylimits': ylimits,
                    'size': size})
            plotting_thread.daemon = True
            plotting_thread.start()
            self.threads.append(plotting_thread)
            logging.info("Plotting thread started")

    def background_stop(self):
        """Stop the background thread."""
        self.busy = False
        self.conn.terminate()
        self.close()
        for thread in self.threads:
            thread.join()

    def _plot(self, plot_length, ylimits, size):
        """Plot the data as it comes in."""
        while len(self.buffer) == 0:
            # Wait until we have some data
            time.sleep(SLEEP_STEP)
        plotter = PlotBuffer(buffer=self.buffer, plot_length=plot_length,
                             ylimits=ylimits, size=size)
        old_buffer_limits = {tr.id: (tr.stats.starttime, tr.stats.endtime)
                             for tr in self.buffer}
        while self.busy:
            buffer_limits = {tr.id: (tr.stats.starttime, tr.stats.endtime)
                             for tr in self.buffer}
            if buffer_limits != old_buffer_limits:
                logging.info("New data to plot")
                if len(self.buffer) != len(plotter.data):
                    # Create a new plotter
                    plotter.stop()
                    # Keep the old data
                    new_buffer = self.buffer + plotter.data
                    new_buffer.merge(method=1, fill_value=0,
                                     interpolation_samples=0)
                    plotter = PlotBuffer(
                        buffer=new_buffer, plot_length=plotter.plot_length,
                        ylimits=ylimits, size=size)
                else:
                    plotter.update(new_data=self.buffer,
                                   plot_end=UTCDateTime.now())
                old_buffer_limits = buffer_limits
            time.sleep(SLEEP_STEP)
        logging.info("Stopping plotting")
        plotter.stop()
        logging.info("Plotting stopped")

    def on_data(self, trace):
        """
        Handle incoming data
        :param trace:
        :return: buffer if full.
        """
        logging.info("Packet of {0} samples for {1}".format(
            trace.stats.npts, trace.id))
        self.buffer += trace
        self.buffer.merge()
        _tr = self.buffer.select(id=trace.id)[0]
        if _tr.stats.npts * _tr.stats.delta > self.buffer_capacity:
            logging.info(
                "Trimming trace to {0}-{1}".format(
                    _tr.stats.endtime - self.buffer_capacity,
                    _tr.stats.endtime))
            _tr.trim(_tr.stats.endtime - self.buffer_capacity)
        else:
            logging.debug("Buffer contains {0}".format(self.buffer))

    def on_terminate(self):  # pragma: no cover
        """
        Handle termination gracefully
        """
        logging.info("Termination of {0}".format(self.__repr__()))
        return self.buffer

    def on_error(self):  # pragma: no cover
        """
        Handle errors gracefully.
        """
        logging.error("SeedLink error")
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
