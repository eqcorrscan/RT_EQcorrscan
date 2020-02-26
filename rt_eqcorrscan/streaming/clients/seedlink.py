"""
Data handling for seedlink clients for real-time matched-filter detection.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy import Stream

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
        wavebank: WaveBank = None,
    ) -> None:
        EasySeedLinkClient.__init__(
            self, server_url=server_url, autoconnect=False)
        _StreamingClient.__init__(
            self, server_url=server_url, buffer=buffer,
            buffer_capacity=buffer_capacity, wavebank=wavebank)
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
        print_str = (
            "Seed-link client at {0}, status: {1}, buffer capacity: {2:.1f}s\n"
            "\tCurrent Buffer:\n{3}".format(
                self.server_hostname, status_map[self.busy],
                self.buffer_capacity, self.buffer))
        return print_str

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
            buffer = self.buffer.copy()
        return RealTimeClient(
            server_url=self.server_hostname, buffer=buffer,
            buffer_capacity=self.buffer_capacity, wavebank=self.wavebank)

    def start(self) -> None:
        """ Start the connection. """
        if not self.started:
            self.connect()
            self.started = True
        else:
            Logger.warning("Attempted to start connection, but "
                           "connection already started.")

    @property
    def can_add_streams(self) -> bool:
        return not self._EasySeedLinkClient__streaming_started

    def stop(self) -> None:
        self.busy = False
        self.conn.terminate()
        self.close()
        self.started = False

    def on_seedlink_error(self):  # pragma: no cover
        """ Cope with seedlink errors."""
        self.on_error()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
