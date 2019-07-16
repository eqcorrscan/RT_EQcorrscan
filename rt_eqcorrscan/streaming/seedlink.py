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

from rt_eqcorrscan.streaming.streaming import _StreamingClient


Logger = logging.getLogger(__name__)


class RealTimeClient(_StreamingClient, EasySeedLinkClient):
    """
    SeedLink client link for Real-Time Matched-Filtering.

    Parameters
    ----------
    server_url
        URL for seedlink server
    autoconnect
        Whether to start connection automatically or wait
    buffer
        Stream to buffer data into
    buffer_capacity
        Length of buffer in seconds. Old data are removed in a LIFO style.
    """
    busy = False

    def __init__(
        self,
        server_url: str,
        autoconnect: bool = True,
        buffer: Stream = None,
        buffer_capacity: float = 600.
    ) -> None:
        EasySeedLinkClient.__init__(
            self, server_url=server_url, autoconnect=autoconnect)
        _StreamingClient.__init__(
            self, client_name=server_url, buffer=buffer,
            buffer_capacity=buffer_capacity)

        Logger.info("Instantiated RealTime client: {0}".format(self))

    def __repr__(self):
        """
        Print information about the client.

        .. rubric:: Example

        >>> client = RealTimeClient(server_url="geofon.gfz-potsdam.de")
        >>> print(client) # doctest: +NORMALIZE_WHITESPACE
        Seed-link client at geofon.gfz-potsdam.de, status: Stopped, buffer \
        capacity: 600.0s
            Current Buffer:
        0 Trace(s) in Stream:
        <BLANKLINE>
        """
        status_map = {True: "Running", False: "Stopped"}
        print_str = (
            "Seed-link client at {0}, status: {1}, buffer capacity: {2:.1f}s\n"
            "\tCurrent Buffer:\n{3}".format(
                self.server_hostname, status_map[self.busy],
                self.buffer_capacity, self.buffer))
        return print_str

    @property
    def can_add_streams(self) -> bool:
        return not self._EasySeedLinkClient__streaming_started

    def stop(self) -> None:
        self.busy = False
        self.conn.terminate()
        self.close()

    def on_seedlink_error(self):
        """ Cope with seedlink errors."""
        self.on_error()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
