"""
Data handling for seedlink clients for real-time matched-filter detection.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging
import threading

from obspy.clients.seedlink.easyseedlink import (
    EasySeedLinkClient, EasySeedLinkClientException)
from obspy.clients.seedlink.slpacket import SLPacket
from obspy import Stream, UTCDateTime

from obsplus import WaveBank

from rt_eqcorrscan.streaming.streaming import _StreamingClient


Logger = logging.getLogger(__name__)

EXIT_EVENT = threading.Event()


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
        wavebank: WaveBank = None,
    ) -> None:
        EasySeedLinkClient.__init__(
            self, server_url=server_url, autoconnect=False)
        _StreamingClient.__init__(
            self, server_url=server_url, buffer=buffer,
            buffer_capacity=buffer_capacity, wavebank=wavebank)
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
                    self.server_hostname, status_map[self.busy],
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
        for connection closure from another thread.
        """
        # Note: This somewhat resembles the run() method in SLClient.
        EXIT_EVENT.clear()  # Clear the exit event

        # Check if any streams have been specified (otherwise this will result
        # in an infinite reconnect loop in the SeedLinkConnection)
        if not len(self.conn.streams):
            msg = 'No streams specified. Use select_stream() to select ' + \
                  'a stream.'
            raise EasySeedLinkClientException(msg)

        self._EasySeedLinkClient__streaming_started = True

        # Start the collection loop
        while True:
            data = self.conn.collect()

            if data == SLPacket.SLTERMINATE:
                Logger.warning("Recieved Terminate request from host")
                self.on_terminate()
                break
            if EXIT_EVENT.is_set():
                Logger.warning("Run termination called - EXIT_EVENT is set.")
                self.on_terminate()
                break
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
            buffer_capacity=self.buffer_capacity, wavebank=self.wavebank)

    def start(self) -> None:
        """ Start the connection. """
        if not self.started:
            self.connect()
            self.started = True
            self._last_data = UTCDateTime.now()
        else:
            Logger.warning("Attempted to start connection, but "
                           "connection already started.")

    def background_stop(self):
        """Stop the background thread."""
        self._stop_called = True
        EXIT_EVENT.set()
        self.stop()
        for thread in self.threads:
            Logger.info("Joining thread")
            thread.join()
            Logger.info("Thread joined")
        self.threads = []
        EXIT_EVENT.clear()

    def restart(self) -> None:
        """ Restart the streamer. """
        Logger.warning("RESTART: Stopping the streamer")
        self.stop()
        Logger.warning("RESTART: Starting the streamer")
        self.start()
        Logger.warning("RESTART: Completed restart")

    @property
    def can_add_streams(self) -> bool:
        return not self._EasySeedLinkClient__streaming_started

    def stop(self) -> None:
        self._stop_called = True
        self.busy = False
        Logger.info("Terminating connection")
        self.conn.terminate()
        self.conn.do_terminate()
        Logger.info("Closing connection")
        self.close()
        self.started = False
        Logger.info("Stopped Streamer")

    def on_terminate(self) -> Stream:  # pragma: no cover
        """
        Handle termination gracefully
        """
        Logger.info("Termination of {0}".format(self.__repr__()))
        if not self._stop_called:  # Make sure we don't double-call stop methods
            if len(self.threads):
                self.background_stop()
            else:
                self.stop()
        else:
            Logger.info("Stop already called - not duplicating")
        EXIT_EVENT.clear()
        return self.stream


    def on_seedlink_error(self):  # pragma: no cover
        """ Cope with seedlink errors."""
        self.on_error()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
