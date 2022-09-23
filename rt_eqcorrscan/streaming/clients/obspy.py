"""
Data handling to simulate a real-time client from old data via ObsPy clients
for testing of real-time matched-filter detection.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging
import time
import copy
from numpy import random
import importlib

from typing import Iterable
from obspy import Stream, UTCDateTime
from queue import Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

from rt_eqcorrscan.streaming.streaming import _StreamingClient


Logger = logging.getLogger(__name__)


# TODO: This should maintain a buffer in-memory and have a method to repopulate the buffer, called from within the run loop. That should get data from within a Process. Needs a lock on the stream.
class StreamClient:
    """
    In-memory handling of stream as fake Client. Cache future data in memory and
    provide via client-like get_waveforms requests.

    Parameters
    ----------

    """
    __st = Stream()

    def __init__(
        self,
        client,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        buffer_length: float = 3600,
        min_buffer_length: float = 120,
    ):
        self.client = client
        self.starttime = starttime
        self.endtime = endtime
        self.buffer_length = buffer_length
        self.min_buffer_length = min_buffer_length

    def stream(self):
        pass

    def get_waveforms(
            self,
            network: str,
            station: str,
            location: str,
            channel: str,
            starttime: UTCDateTime,
            endtime: UTCDateTime
    ):
        pass

    def get_wavforms_bulk(self, bulk: Iterable):
        st = Stream()
        for _bulk in bulk:
            st += self.get_waveforms(*_bulk)
        return st


class RealTimeClient(_StreamingClient):
    """
    Simulation of a real-time client for past data. Used for testing

    Parameters
    ----------
    server_url
        URL or mappabale name of the client, if not providing a Client, then
        this should be the argument to set-up a client of `client_type`
    client
        Any client or that supports waveform data queries.
    starttime
        Starttime for client (in the past)
    client_type
        Obspy client type to start-up, only used if `client=None`.
    query_interval
        Interval in seconds to query the client for new data
    speed_up
        Multiplier to run faster than real-time (real-time is 1.0).
    buffer
        Stream to buffer data into
    buffer_capacity
        Length of buffer in seconds. Old data are removed in a FIFO style.
    """
    client_base = "obspy.clients"
    can_add_streams = True
    max_threads = None

    def __init__(
        self,
        server_url: str,
        starttime: UTCDateTime,
        client=None,
        client_type: str = "FDSN",
        query_interval: float = 10.,
        speed_up: float = 1.,
        buffer: Stream = None,
        buffer_capacity: float = 600.,
    ) -> None:
        if client is None:
            try:
                _client_module = importlib.import_module(
                    f"{self.client_base}.{client_type.lower()}")
                client = _client_module.Client(server_url)
            except Exception as e:
                Logger.error("Could not instantiate simulated client")
                raise e
        self.client = client
        super().__init__(
            server_url=self.client.base_url, buffer=buffer,
            buffer_capacity=buffer_capacity)
        self.starttime = starttime
        self.query_interval = query_interval
        self.speed_up = speed_up
        self.bulk = []
        self.streaming = False
        Logger.info(
            "Instantiated simulated real-time client "
            "(starttime = {0}): {1}".format(self.starttime, self))

    def start(self) -> None:
        """ Dummy - client is always started. """
        self.started = True
        return

    def restart(self) -> None:
        """ Restart the streamer. """
        self.stop()
        self.start()

    def copy(self, empty_buffer: bool = True):
        if empty_buffer:
            buffer = Stream()
        else:
            buffer = self.stream
        return RealTimeClient(
            server_url=self.client.base_url,
            client=self.client, starttime=self.starttime,
            query_interval=self.query_interval, speed_up=self.speed_up,
            buffer=buffer, buffer_capacity=self.buffer_capacity)

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

    def _collect_bulk(self, last_query_start, now, executor):
        query_passed, st = True, Stream()
        for _bulk in self.bulk:
            jitter = random.randint(int(self.query_interval))
            _bulk.update({
                "starttime": last_query_start,
                "endtime": now - jitter})
        futures = {executor.submit(self.client.get_waveforms, **_bulk):
                   _bulk for _bulk in self.bulk}
        for future in as_completed(futures):
            _bulk = futures[future]
            try:
                _st = future.result()
            except Exception as e:
                Logger.error("Failed (bulk={0})".format(_bulk))
                Logger.error(e)
                query_passed = False
                continue
            st += _st
        return st, query_passed

    def run(self) -> None:
        assert len(self.bulk) > 0, "Select a stream first"
        self.streaming = True
        # start threadpool executor
        executor = ThreadPoolExecutor(max_workers=min(len(self.bulk), self.max_threads))
        now = copy.deepcopy(self.starttime)
        self.last_data = UTCDateTime.now()
        last_query_start = now - self.query_interval
        while not self._stop_called:
            # If this is running in a process then we need to check the queue
            try:
                kill = self._killer_queue.get(block=False)
            except Empty:
                kill = False
            Logger.debug(f"Kill status: {kill}")
            if kill:
                Logger.warning("Termination called, stopping collect loop")
                self.on_terminate()
                break
            _query_start = UTCDateTime.now()
            st, query_passed = self._collect_bulk(
                last_query_start=last_query_start, now=now, executor=executor)
            for tr in st:
                self.on_data(tr)
                time.sleep(0.0001)
            # Put the data in the buffer
            self._add_data_from_queue()
            _query_duration = UTCDateTime.now() - _query_start
            Logger.debug(
                "It took {0:.2f}s to query the database and sort data".format(
                    _query_duration))
            sleep_step = (
                self.query_interval - _query_duration) / self.speed_up
            if sleep_step > 0:
                Logger.info("Waiting {0:.2f}s before next query".format(
                    sleep_step))
                time.sleep(sleep_step)
            else:
                Logger.warning(f"Query ({_query_duration} took longer than query "
                               f"interval {self.query_interval}")
            now += max(self.query_interval, _query_duration)
            if query_passed:
                last_query_start = min(_bulk["endtime"] for _bulk in self.bulk)
        self.streaming = False
        # shut down threadpool, we done.
        executor.shutdown(wait=False, cancel_futures=True)
        return

    def stop(self) -> None:
        Logger.info("STOP!")
        self._stop_called, self.started = True, False


if __name__ == "__main__":
    import doctest

    logging.basicConfig(level="DEBUG")
    doctest.testmod()
