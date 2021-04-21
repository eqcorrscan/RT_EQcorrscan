"""
Data handling for seedlink clients for real-time matched-filter detection.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging

from queue import Empty

from obspy.clients.seedlink.easyseedlink import (
    EasySeedLinkClient, EasySeedLinkClientException)
from obspy.clients.seedlink.slpacket import SLPacket
from obspy import Stream, UTCDateTime

from obsplus import WaveBank

from rt_eqcorrscan.streaming.streaming import _StreamingClient


Logger = logging.getLogger(__name__)


class RealTimeClient(_StreamingClient, EasySeedLinkClient):
    """
    SeedLink client link for Real-Time Matched-Filtering.

    Parameters
    ----------
    server_url
        URL for seedlink server
    buffer
        Stream to buffer data into
    buffer_capacity
        Length of buffer in seconds. Old data are removed in a LIFO style.
    """
    def __init__(
        self,
        server_url: str,
        buffer: Stream = None,
        buffer_capacity: float = 600.,
    ) -> None:
        EasySeedLinkClient.__init__(
            self, server_url=server_url, autoconnect=False)
        _StreamingClient.__init__(
            self, server_url=server_url, buffer=buffer,
            buffer_capacity=buffer_capacity)
        self.conn.keepalive = 0
        self.conn.netdly = 30
        Logger.debug("Instantiated RealTime client: {0}".format(self))

    def __repr__(self):
        """
        Print information about the client.

        .. rubric:: Example

        >>> client = RealTimeClient(server_url="geofon.gfz-potsdam.de")
        >>> print(client) # doctest: +NORMALIZE_WHITESPACE
        Seed-link client at geofon.gfz-potsdam.de, status: Stopped, buffer \
        capacity: 600.0s
            Current Buffer:
        Buffer(0 traces, maxlen=600.0)
        """
        status_map = {True: "Running", False: "Stopped"}
        with self.lock:
            print_str = (
                "Seed-link client at {0}, status: {1}, buffer capacity: "
                "{2:.1f}s\n\tCurrent Buffer:\n{3}".format(
                    self.server_hostname, status_map[self.streaming],
                    self.buffer_capacity, self.buffer))
        return print_str

    def run(self):
        """
        Start streaming data from the SeedLink server.

        Streams need to be selected using
        :meth:`~.EasySeedLinkClient.select_stream` before this is called.

        This method enters an infinite loop, calling the client's callbacks
        when events occur.

        This is an edited version of EasySeedlinkClient from ObsPy allowing
        for connection closure from another process.
        """
        self.can_add_streams = False
        # Note: This somewhat resembles the run() method in SLClient.

        # Check if any streams have been specified (otherwise this will result
        # in an infinite reconnect loop in the SeedLinkConnection)
        if not len(self.conn.streams):
            msg = 'No streams specified. Use select_stream() to select ' + \
                  'a stream.'
            raise EasySeedLinkClientException(msg)

        self._EasySeedLinkClient__streaming_started = True

        # Start the collection loop
        self._stop_called = False  # Reset this - if someone called run,
        # they probably want us to run!

        # This only works for local running, not background
        while not self._stop_called:
            data = self.conn.collect()
            try:
                kill = self._killer_queue.get(block=False)
            except Empty:
                kill = False
            if kill:
                Logger.warning(
                    "Run termination called - poison received.")
                self.on_terminate()
                self._stop_called = True
                break

            if data == SLPacket.SLTERMINATE:
                Logger.warning("Received Terminate request from host")
                self.on_terminate()
                return
            elif data == SLPacket.SLERROR:
                self.on_seedlink_error()
                continue

            # At this point the received data should be a SeedLink packet
            # XXX In SLClient there is a check for data == None, but I think
            #     there is no way that self.conn.collect() can ever return None
            assert(isinstance(data, SLPacket))

            packet_type = data.get_type()

            # Ignore in-stream INFO packets (not supported)
            if packet_type not in (SLPacket.TYPE_SLINF, SLPacket.TYPE_SLINFT):
                # The packet should be a data packet
                trace = data.get_trace()
                # Pass the trace to the on_data callback
                self.on_data(trace)

            # Check the incoming queue for data and add it to the buffer
            # Doing this outside of "on_data" should allow external Processes
            # to add data to the buffer
            Logger.debug("Checking the incoming queue")
            self._add_data_from_queue()

        # If we get to here, stop has been called so we can terminate
        self.on_terminate()
        self.streaming = False
        return

    def copy(self, empty_buffer: bool = True):
        """
        Generate a new, unconnected copy of the client.

        Parameters
        ----------
        empty_buffer
            Whether to start the new client with an empty buffer or not.
        """
        if empty_buffer:
            buffer = Stream()
        else:
            buffer = self.stream
        return RealTimeClient(
            server_url=self.server_hostname, buffer=buffer,
            buffer_capacity=self.buffer_capacity)

    def start(self) -> None:
        """ Start the connection. """
        try:
            self.connect()
        except Exception as e:
            Logger.warning(f"Could not connect due to {e}, has a "
                           "connection already started?")
        self.started = True
        self.last_data = UTCDateTime.now()

    def restart(self) -> None:
        """ Restart the streamer. """
        Logger.warning("RESTART: Stopping the streamer")
        self.stop()
        Logger.warning("RESTART: Starting the streamer")
        self.start()
        Logger.warning("RESTART: Completed restart")

    def select_stream(self, net: str, station: str, selector: str = None):
        """
        Select a stream for data transfer.

        Adapted from Obspy EasySeedLinkClient

        Parameters:
        net:
            Network ID
        station:
            Station ID
        selector:
            Valid Seedlink Channel selector
        """
        if not self.has_capability('multistation'):
            msg = 'SeedLink server does not support multi-station mode'
            raise EasySeedLinkClientException(msg)

        if not self.can_add_streams:
            msg = 'Adding streams is not supported after the SeedLink ' + \
                  'connection has entered streaming mode.'
            raise EasySeedLinkClientException(msg)

        self.conn.add_stream(net, station, selector, seqnum=-1, timestamp=None)

    def stop(self) -> None:
        self._stop_called = True
        Logger.info("Terminating connection")
        self.conn.terminate()
        self.conn.do_terminate()
        Logger.info("Closing connection")
        self.close()
        Logger.info("Stopped Streamer")

    def on_seedlink_error(self):  # pragma: no cover
        """ Cope with seedlink errors."""
        self.on_error()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
