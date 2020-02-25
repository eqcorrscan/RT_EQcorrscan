"""
Data handling to simulate a real-time client from old data via ObsPlus WaveBank
for testing of real-time matched-filter detection.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging

from obspy import Stream, UTCDateTime

from obsplus import WaveBank

from rt_eqcorrscan.database.client_emulation import ClientBank
from rt_eqcorrscan.streaming.clients.obspy import RealTimeClient as OBSRTCli


Logger = logging.getLogger(__name__)


class RealTimeClient(OBSRTCli):
    """
    Simulation of a real-time client for past data. Used for testing

    Parameters
    ----------
    server_url
        The base-path for the ObsPlus wavebank.
    client
        Any client or that supports waveform data queries.
    starttime
        Starttime for client (in the past)
    query_interval
        Interval in seconds to query the client for new data
    speed_up
        Multiplier to run faster than real-time (real-time is 1.0).
    buffer
        Stream to buffer data into
    buffer_capacity
        Length of buffer in seconds. Old data are removed in a FIFO style.
    """
    def __init__(
        self,
        server_url: str,
        starttime: UTCDateTime,
        client=None,
        query_interval: float = 10.,
        speed_up: float = 1.,
        buffer: Stream = None,
        buffer_capacity: float = 600.,
        **kwargs
    ) -> None:
        if client is None:
            try:
                client = ClientBank(wave_bank=WaveBank(server_url))
            except Exception as e:
                Logger.error("Could not instantiate simulated client")
                raise e
        super().__init__(
            server_url=server_url, starttime=starttime, client=client,
            query_interval=query_interval, speed_up=speed_up, buffer=buffer,
            buffer_capacity=buffer_capacity, wavebank=None)


if __name__ == "__main__":
    import doctest

    logging.basicConfig(level="DEBUG")
    doctest.testmod()
