"""
Data handling to simulate a real-time client from old data via ObsPy clients
for testing of real-time matched-filter detection.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging
import os
import time
from numpy import random
import importlib

from copy import copy, deepcopy
from typing import Iterable
from obspy import Stream, UTCDateTime
from queue import Empty, Full
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Lock, Process, Queue

from rt_eqcorrscan.database.client_emulation import LocalClient
from rt_eqcorrscan.streaming.streaming import _StreamingClient


Logger = logging.getLogger(__name__)


class StreamClient:
    """
    In-memory handling of stream as fake Client. Cache future data in memory and
    provide via client-like get_waveforms requests.

    Parameters
    ----------
    client
        Client to get data from - can be anything with a .get_waveforms method
    buffer_length
        Initial length of buffer in seconds
    min_buffer_fraction
        Minimum buffer fraction to trigger a re-fresh of the buffer.
    """
    # Locks on shared objects
    _stream_lock = Lock()

    _maintain = False
    _stop_called = False

    def __init__(
        self,
        client,
        buffer_length: float = 3600,
        min_buffer_fraction: float = 0.25,
        speed_up: float = 1.0,
    ):
        self.__st = Stream()
        self.client = client
        self.buffer_length = buffer_length
        self._min_buffer_length = buffer_length * min_buffer_fraction
        self.speed_up = speed_up

        # Data queue for communication
        self._stream_queue = Queue()

        # Poison!
        self._killer_queue = Queue(maxsize=1)
        self._dead_queue = Queue(maxsize=1)

        self.processes = []
        self.base_url = self.client.base_url

    @property
    def stream(self) -> Stream:
        with self._stream_lock:
            Logger.debug("Stream getter called")
            st, i = Stream(), 0
            while True:
                try:
                    st += self._stream_queue.get(block=True, timeout=0.5)
                    Logger.debug(f"{i}\tGot stream from queue: {st}")
                except Empty:
                    if i == 0:
                        Logger.warning(
                            "Stream queue is empty - have you initialized?")
                    break
                i += 1
            self.__st = st.merge(method=1)
            Logger.debug(f"Putting stream back into queue: {self.__st}")
            self._stream_queue.put(self.__st, block=True, timeout=0.5)
        return self.__st

    @stream.setter
    def stream(self, st: Stream):
        Logger.debug("Stream setter called")
        # Empty stream queue
        Logger.debug("Emptying queue")
        with self._stream_lock:
            while True:
                try:
                    self._stream_queue.get(block=True, timeout=0.5)
                except Empty:
                    Logger.debug("Queue is empty")
                    break
        # Put stream in queue
        Logger.debug("Putting stream in queue")
        with self._stream_lock:
            self._stream_queue.put(st, block=True, timeout=0.5)
        Logger.debug("Synchronizing")
        _ = self.stream  # Synchronize

    @property
    def stats(self) -> dict:
        """ Provide a copy of the stats """
        st = self.stream
        stats = {(tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel):
                 (tr.stats.starttime, tr.stats.endtime) for tr in st}
        return stats

    def get_waveforms(
            self,
            network: str,
            station: str,
            location: str,
            channel: str,
            starttime: UTCDateTime,
            endtime: UTCDateTime,
            rm_selected: bool = True
    ) -> Stream:
        """

        Parameters
        ----------
        network
            Network code to get data for
        station
            Station code to get data for
        location
            Location code to get data for
        channel
            Channel code to get data for
        starttime
            Starttime to select data for
        endtime
            Endtime to select data for
        rm_selected
            Whether to remove the data from the buffer - defaults to True

        Returns
        -------
        Stream of selected data.
        """
        return self.get_waveforms_bulk(
            [network, station, location, channel, starttime, endtime])

    def get_waveforms_bulk(
            self,
            bulk: Iterable,
            rm_selected: bool = True
    ) -> Stream:
        """

        Parameters
        ----------
        bulk
            Bulk of (network, station, location, channel, starttime, endtime)
            to request data for.
        rm_selected
            Whether to remove the selected data from the buffer. Default: True

        Returns
        -------
        Stream of select data from buffer.
        """
        full_st, st = self.stream, Stream()  # Get local version of stream
        for _bulk in bulk:
            n, s, l, c, _s, _e = _bulk
            st += full_st.select(
                network=n, station=s, location=l, channel=c).slice(
                starttime=_s, endtime=_e).copy()
        # Remove all at once to avoid accessing the queue repeatedly
        if rm_selected:
            trimmed_st = Stream()
            for _bulk in bulk:
                network, station, location, channel, starttime, endtime = _bulk
                trimmed = full_st.select(
                    network=network, station=station, location=location,
                    channel=channel)
                if len(trimmed) == 0:
                    continue
                trimmed_st += trimmed.trim(starttime=endtime)
            Logger.debug(f"Putting trimmed stream back in:\n{trimmed_st}")
            # Put back into queue
            self.stream = trimmed_st
        Logger.debug(f"Returning stream from StreamClient for bulk: {bulk}:\n{st}")
        return st

    def initiate_buffer(self, seed_ids: Iterable[str], starttime: UTCDateTime):
        """
        Initial population of the buffer.

        Parameters
        ----------
        seed_ids:
            Iterable of seed ids as network.station.location.channel
        starttime:
            Starttime to initialise buffer from
        """
        st = Stream()
        bulk = [
            tuple(seed_id.split('.') +
            [starttime, starttime + self.buffer_length])
            for seed_id in seed_ids]
        for _bulk in bulk:
            st += self.client.get_waveforms(*_bulk)
        self.stream = st
        Logger.debug(f"Collected hidden buffer:\n{st}")
        return

    def _clear_killer(self):
        """ Clear the killer Queue. """
        while True:
            try:
                self._killer_queue.get(block=False)
            except Empty:
                break
        while True:
            try:
                self._dead_queue.get(block=False)
            except Empty:
                break

    def maintain_buffer(self):
        """
        Maintain buffer length in the background.
        """
        self._maintain = True
        self._clear_killer()
        maintaining_process = Process(
            target=self._bg_run, name="MaintainProcess")
        # streaming_process.daemon = True
        maintaining_process.start()
        self.processes.append(maintaining_process)
        Logger.info("Started streaming")
        Logger.warning(self.processes)

    def _bg_run(self):
        """ Run the run methods in the background and close nicely when done. """
        while self._maintain:
            self.run()
        Logger.info("Running stopped, busy set to False")
        try:
            self._dead_queue.get(block=False)
        except Empty:
            pass
        # Report death
        Logger.info("Reporting death")
        self._dead_queue.put(True)
        Logger.info("Death reported")
        return

    def background_stop(self):
        """Stop the background process."""
        Logger.info("Adding Poison to Kill Queue")

        self._killer_queue.put(True)
        self.stop()
        # Wait until streaming has stopped
        Logger.info(
            f"Waiting for maintaining to stop: status = {self._maintain}")
        while self._maintain:
            try:
                self._maintain = not self._dead_queue.get(block=False)
            except Empty:
                time.sleep(1)
                pass
        Logger.debug("Streaming stopped")
        # Empty queues
        for queue in [self._stream_queue, self._killer_queue, self._dead_queue]:
            while True:
                try:
                    queue.get(block=False)
                except Empty:
                    break
        # join the processes
        Logger.info(f"Stopping maintaining processes: {self.processes}")
        for process in self.processes:
            Logger.info(f"Joining process: {process.name}")
            process.join(5)
            if hasattr(process, 'exitcode') and process.exitcode:
                Logger.info("Process failed to join, terminating")
                process.terminate()
                Logger.info("Terminated")
                process.join()
            Logger.info("Process joined")
        self.processes = []
        self._maintain = False
        Logger.info("Background stop completed")
        return

    def run(self):
        """
        Maintain buffer length
        """
        kill = False
        while not self._stop_called:
            Logger.info(f"Hidden Streamer running for: {self.stats}")
            new_stream = Stream()
            for nslc, (starttime, endtime) in self.stats.items():
                Logger.debug(f"Hidden Streamer: {nslc} length: "
                            f"{endtime - starttime}, min-length: "
                            f"{self._min_buffer_length}")
                if endtime - starttime <= self._min_buffer_length:
                    endtime = starttime + self.buffer_length
                    net, sta, loc, chan = nslc
                    Logger.debug(
                        f"Updating buffer for {net}.{sta}.{loc}.{chan} "
                        f"between {starttime} and {endtime}")
                    new_stream += self.client.get_waveforms(
                        network=net, station=sta, channel=chan, location=loc,
                        starttime=starttime, endtime=endtime)
            if len(new_stream):
                Logger.debug(f"Acquired new data for buffer: {new_stream}")
                new_stream += self.stream
                new_stream.merge(method=1)
                self.stream = new_stream
            # Sleep in small steps
            _sleep, sleep_duration, sleep_step = (
                0, (self._min_buffer_length / 2) / self.speed_up, 0.5)
            Logger.debug(f"Sleeping for {sleep_duration}")
            while _sleep <= sleep_duration:
                # If this is running in a process then we need to check the queue
                try:
                    kill = self._killer_queue.get(block=False)
                except Empty:
                    kill = False
                Logger.debug(f"Kill status: {kill}")
                if kill:
                    # Need to put back into the killer queue to make sure other
                    # processes get killed
                    self._killer_queue.put(True)
                    Logger.warning("Termination called, stopping collect loop")
                    self.on_terminate()
                    break
                time.sleep(sleep_step)
                _sleep += sleep_step
            if kill:
                break
        Logger.debug("Out of run loop, returning")
        self._maintain = False
        return

    def stop(self) -> None:
        Logger.info("STOP! Hidden streamer")
        self._stop_called = True

    def on_terminate(self):  # pragma: no cover
        """
        Handle termination gracefully
        """
        Logger.info("Termination of {0}".format(self.__repr__()))
        if not self._stop_called:  # Make sure we don't double-call stop methods
            if len(self.processes):
                self.background_stop()
            else:
                self.stop()
        else:
            Logger.info("Stop already called - not duplicating")


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
    max_threads = os.cpu_count() + 4

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
        pre_empt_data: bool = True,
        pre_empt_len: float = 6000.,
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
        self.pre_empt_data = pre_empt_data

        # Convert to pre-emptive client.
        if pre_empt_data and not isinstance(self.client, StreamClient):
            self.client = StreamClient(
                self.client, min_buffer_fraction=0.2,
                buffer_length=pre_empt_len or 10 * buffer_capacity,
                speed_up=speed_up)
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
        bulk = deepcopy(self.bulk)
        for _bulk in bulk:
            # jitter = random.randint(int(self.query_interval / 10) or 1)
            jitter = 0
            _bulk.update({
                "starttime": last_query_start,
                "endtime": now - jitter})
        self.bulk = bulk
        Logger.debug(f"Bulk request using: {bulk}")
        if self.pre_empt_data:
            # Use inbuilt bulk method - more efficient
            Logger.debug("Pre-empted data: Using get_waveforms_bulk")
            return self.client.get_waveforms_bulk(
                [(b['network'], b['station'], b['location'], b['channel'],
                  b['starttime'], b['endtime'])
                 for b in bulk]), True
        if executor is None:
            for _bulk in bulk:
                Logger.debug(f"Getting data for {_bulk}")
                try:
                    _st = self.client.get_waveforms(**_bulk)
                except Exception as e:
                    Logger.error(f"Failed (bulk={_bulk})")
                    Logger.error(e)
                    query_passed = False
                    continue
                else:
                    for tr in _st:
                        Logger.debug(f"Got {tr}")
                        st += tr
            st = st.merge(method=1)
        else:
            futures = {executor.submit(self.client.get_waveforms, **_bulk):
                       _bulk for _bulk in bulk}
            for future in as_completed(futures):
                _bulk = futures[future]
                try:
                    _st = future.result()
                except Exception as e:
                    Logger.error("Failed (bulk={0})".format(_bulk))
                    Logger.error(e)
                    query_passed = False
                    continue
                for tr in _st:
                    Logger.debug(f"For bulk: {_bulk}")
                    Logger.debug(f"Got {tr} from future")
                st = (st + _st).merge(method=1)
        return st, query_passed

    def background_run(self):
        """Run the client in the background."""
        self.streaming, self.started, self.can_add_streams = True, True, False
        self._clear_killer()   # Clear the kill queue

        # Start and manage the maintainer here
        if self.pre_empt_data:
            assert len(self.bulk) > 0, "Select a stream first"
            Logger.debug("Collecting pre-emptive data")
            _sids = [
                f"{b['network']}.{b['station']}.{b['location']}.{b['channel']}"
                for b in self.bulk]
            self.client.initiate_buffer(
                seed_ids=_sids, starttime=self.starttime)
            self.client.maintain_buffer()
            self.processes.extend(self.client.processes)
        streaming_process = Process(
            target=self._bg_run, name="StreamProcess",
            kwargs={"do_not_start_maintainer": True})
        # streaming_process.daemon = True
        streaming_process.start()
        self.processes.append(streaming_process)
        Logger.info("Started streaming")
        Logger.info(f"Processes: {', '.join([p.name for p in self.processes])}")

    def background_stop(self):
        """Stop the background process."""
        Logger.info("Adding Poison to Kill Queue")
        # Run communications before termination
        st = self.stream
        self.__buffer_full = self.buffer_full
        self.__last_data = self.last_data

        if self.pre_empt_data:
            Logger.info("Running background stop on maintain client")
            self.client.background_stop()

        Logger.debug(f"Stream on termination: {st}")
        self._killer_queue.put(True)
        self.stop()
        # Local buffer
        for tr in st:
            Logger.info("Adding trace to local buffer")
            self.buffer.add_stream(tr)
        # Wait until streaming has stopped
        Logger.info(
            f"Waiting for streaming to stop: status = {self.streaming}")
        while self.streaming:
            try:
                self.streaming = not self._dead_queue.get(block=False)
            except Empty:
                time.sleep(1)
                pass
        Logger.debug("Streaming stopped")
        # Empty queues
        for queue in [self._incoming_queue, self._stream_queue,
                      self._killer_queue, self._last_data_queue]:
            while True:
                try:
                    queue.get(block=False)
                except Empty:
                    break
        # join the processes
        for process in self.processes:
            Logger.info(f"Joining process {process.name}")
            # Ugly, but we seem to need to terminate processes - probably doing
            # something wrong.
            process.terminate()
            Logger.info(f"Process {process.name} joined")
        self.processes = []
        self.streaming = False
        Logger.info("Streaming background stop completed")
        return

    def run(self, do_not_start_maintainer: bool = False) -> None:
        assert len(self.bulk) > 0, "Select a stream first"
        if self.pre_empt_data and not do_not_start_maintainer:
            Logger.debug("Collecting pre-emptive data")
            _sids = [
                f"{b['network']}.{b['station']}.{b['location']}.{b['channel']}"
                for b in self.bulk]
            self.client.initiate_buffer(
                seed_ids=_sids, starttime=self.starttime)
            self.client.maintain_buffer()

        self.streaming = True
        # start threadpool executor
        # executor = ThreadPoolExecutor(max_workers=min(len(self.bulk), self.max_threads))
        executor = None  # Debugging whether the threadpool is the issue with not getting new data?
        query_starttime = deepcopy(self.starttime)
        self.last_data = UTCDateTime.now()
        last_query_start = query_starttime - self.query_interval
        killed = False
        elapsed = 0.0
        while not self._stop_called:
            tic = time.perf_counter()
            now = query_starttime + (elapsed * self.speed_up)
            Logger.debug(f"After {elapsed * self.speed_up:.1f} s, the time is now {now}")

            Logger.debug(f"Requesting data between {last_query_start} and {now}")
            st, query_passed = self._collect_bulk(
                last_query_start=last_query_start, now=now, executor=executor)
            Logger.debug(f"Received stream from database: \n{st.__str__(extended=True)}")

            Logger.debug(f"Getting data took {(time.perf_counter() - tic) * self.speed_up}s")

            # Trim to what we need - this will also limit the query duration
            Logger.debug(f"Trimming streaming data between "
                         f"{now - (2 * self.buffer_capacity)} and {now}")
            st = st.trim(starttime=now - (2 * self.buffer_capacity), endtime=now)

            for tr in st:
                Logger.debug(f"Putting {tr.id}, {tr.stats.starttime} -- {tr.stats.endtime} into buffer")
                self.on_data(tr)
                time.sleep(0.0001)

            # Put the data in the buffer
            self._add_data_from_queue()

            Logger.debug(
                "It took {0:.2f}s to query the database and sort data".format(
                    time.perf_counter() - tic))

            sleep_step = self.query_interval - ((time.perf_counter() - tic) * self.speed_up)
            if sleep_step > 0:
                Logger.debug("Waiting {0:.2f}s before next query".format(
                    sleep_step))  # Report fake time
                _slept, _sleep_int = 0.0, min(1, (sleep_step / self.speed_up) / 100)
                # Sleep in small steps to make death responsive
                while _slept < sleep_step / self.speed_up:
                    toc = time.perf_counter()
                    time.sleep(_sleep_int)
                    killed = self._kill_check()
                    if killed:
                        break
                    _slept += time.perf_counter() - toc
                Logger.debug("Waking up")
            else:
                Logger.warning(f"Query took longer than query "
                               f"interval {self.query_interval}")
                killed = self._kill_check()
            if killed:
                break
            if query_passed:
                # last_query_start = min(_bulk["endtime"] for _bulk in self.bulk)
                if len(st):
                    last_query_start = min(tr.stats.starttime for tr in st)
                    # Don't update if we didn't get a stream!
            else:
                Logger.warning("Query failed, may end up with gappy data")
                last_query_start += (time.perf_counter() - tic) * self.speed_up
            Logger.debug(f"After checks the stream is {self.stream}")
            elapsed += time.perf_counter() - tic
        self.streaming = False
        # shut down threadpool, we done.
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        if self.pre_empt_data and not do_not_start_maintainer:
            self.client.background_stop()
            # self.client._killer_queue.put(True)
        return

    def stop(self) -> None:
        Logger.warning("STOP! obspy streamer")
        if self.pre_empt_data:
            self.client.background_stop()
            #if self.client._killer_queue.empty():
            #    self.client._killer_queue.put(True)
        self._stop_called, self.started = True, False
        self.streaming = False


if __name__ == "__main__":
    import doctest

    logging.basicConfig(level="DEBUG")
    doctest.testmod()
