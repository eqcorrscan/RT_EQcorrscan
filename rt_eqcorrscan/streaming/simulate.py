"""
Data handling to simulate a real-time client from old data via FDSN
for testing of real-time matched-filter detection.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging
import time
from numpy.random import randint

from obspy import Stream, UTCDateTime
from rt_eqcorrscan.streaming.streaming import _StreamingClient


Logger = logging.getLogger(__name__)


class SimulateRealTimeClient(_StreamingClient):
    """
    Simulation of a real-time client for past data. Used for testing

    Parameters
    ----------
    client
        Any client that supports streaming data.
    starttime
        Starttime for client (in the past)
    query_interval
        Interval in seconds to query the client for new data
    speed_up
        Multiplier to run faster than real-time (real-time is 1.0).
    buffer
        Stream to buffer data into
    buffer_capacity
        Length of buffer in seconds. Old data are removed in a LIFO style.
    """
    def __init__(
        self,
        client,
        starttime: UTCDateTime,
        query_interval: float = 20.,
        speed_up: float = 1.,
        buffer: Stream = None,
        buffer_capacity: float = 600.
    ) -> None:
        self.client = client
        super().__init__(client_name=self.client.base_url, buffer=buffer,
                         buffer_capacity=buffer_capacity)
        self.starttime = starttime
        self.query_interval = query_interval
        self.speed_up = speed_up
        self.bulk = []
        self.streaming = False
        Logger.info("Instantiated simulated real-time client: {0}".format(
            self.client))

    @property
    def can_add_streams(self) -> bool:
        return True  # We can always add streams

    def select_stream(self, net: str, station: str, selector: str) -> None:
        """
        Select streams to "stream".

        net
            The network id
        station
            The station id
        selector
            a valid SEED ID channel selector, e.g. ``EHZ`` or ``EH?``
        """
        _bulk = {
            "network": net, "station": station, "location": "*",
            "channel": selector, "starttime": None, "endtime": None}
        if _bulk not in self.bulk:
            Logger.debug("Added {0} to streaming selection".format(_bulk))
            self.bulk.append(_bulk)

    def run(self) -> None:
        self.streaming = True
        now = self.starttime
        while self.streaming:
            for _bulk in self.bulk:
                jitter = randint(int(self.query_interval))
                _bulk.update({
                    "starttime": now - self.query_interval,
                    "endtime": now - jitter})
                Logger.debug("Querying client for {0}".format(_bulk))
                try:
                    st = self.client.get_waveforms(**_bulk)
                except Exception as e:
                    Logger.error(e)
                    continue
                for tr in st:
                    self.on_data(tr)
            time.sleep(self.query_interval / self.speed_up)
            now += self.query_interval

    def stop(self) -> None:
        self.busy = False
        self.streaming = False


if __name__ == "__main__":
    import doctest

    logging.basicConfig(level="DEBUG")
    doctest.testmod()
