"""
Functions and classes for handling streaming of data for real-time
matched-filtering.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import time
import multiprocessing
import logging
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, List
from queue import Empty, Full

from obspy import Stream, Trace, UTCDateTime
from obsplus import WaveBank

from rt_eqcorrscan.streaming.buffers import Buffer

Logger = logging.getLogger(__name__)


class _StreamingClient(ABC):
    """
    Abstract Base Class for streaming clients

    Parameters
    ----------
    server_url
        Client URL. Used for book-keeping.
    buffer
        Stream to buffer data into
    buffer_capacity
        Length of buffer in seconds. Old data are removed in a FIFO style.
    wavebank
        WaveBank to save data to. Used for backfilling by RealTimeTribe.
        Set to `None` to not use a WaveBank.

    Notes
    -----
        Requires a `run` and `select_stream` method, however, these cannot be
        added here as abstract methods because they clash with instantiation of
        `EasySeedLinkClient`.
    """
    busy = False
    started = False
    lock = multiprocessing.Lock()  # Lock for buffer access
    wavebank_lock = multiprocessing.Lock()
    has_wavebank = False
    __last_data = None
    _stop_called = False

    def __init__(
        self,
        server_url: str = None,
        buffer: Union[Stream, Buffer] = None,
        buffer_capacity: float = 600.,
        wavebank: WaveBank = WaveBank("Streaming_WaveBank"),
    ) -> None:
        self.server_url = server_url
        if buffer is None:
            buffer = Buffer(traces=[], maxlen=buffer_capacity)
        elif isinstance(buffer, Stream):
            buffer = Buffer(buffer.traces, maxlen=buffer_capacity)
        self._buffer = buffer
        self.buffer_capacity = buffer_capacity

        # Queues for communication
        self._stream_queue = multiprocessing.Queue(maxsize=1)
        self._last_data_queue = multiprocessing.Queue(maxsize=1)
        self._buffer_full_queue = multiprocessing.Queue(maxsize=1)
        # Poison!
        self._killer_queue = multiprocessing.Queue(maxsize=1)

        # Private attributes for properties
        self.__stream = Stream()
        self.__last_data = None
        self.__buffer_full = False

        # Wavebank status to avoid accessing the underlying, lockable, wavebank
        self.__wavebank = wavebank
        if wavebank:
            self.has_wavebank = True
        self._wavebank_warned = False  # Reduce duplicate warnings
        self.processes = []

    def __repr__(self):
        """
        Print information about the client.
        """
        status_map = {True: "Running", False: "Stopped"}
        with self.lock:
            print_str = (
                "Client at {0}, status: {1}, buffer capacity: {2:.1f}s\n"
                "\tCurrent Buffer:\n{3}".format(
                    self.server_url, status_map[self.busy],
                    self.buffer_capacity, self.buffer))
        return print_str

    @abstractmethod
    def start(self) -> None:
        """ Open the connection to the streaming service. """

    @abstractmethod
    def stop(self) -> None:
        """ Stop the system. """

    @property
    @abstractmethod
    def can_add_streams(self) -> bool:
        """ Whether streams can be added."""

    @property
    def buffer(self) -> Buffer:
        return self._buffer

    def clear_buffer(self):
        """Clear the current buffer. Cannot be accessed outside the process."""
        with self.lock:
            self._buffer = Buffer(traces=[], maxlen=self.buffer_capacity)

    @property
    def buffer_full(self) -> bool:
        try:
            self.__buffer_full = self._buffer_full_queue.get(block=False)
        except Empty:
            pass
        return self.__buffer_full

    @buffer_full.setter
    def buffer_full(self, full: bool):
        try:
            self._buffer_full_queue.put(full, block=False)
        except Full:
            try:
                self._buffer_full_queue.get(False)
            except Empty:
                pass
            try:
                self._buffer_full_queue.put(full, timeout=10)
            except Full:
                Logger.error("Could not update buffer full - queue is full")

    @property
    def last_data(self) -> UTCDateTime:
        try:
            self.__last_data = self._last_data_queue.get(block=False)
        except Empty:
            pass
        return self.__last_data

    @last_data.setter
    def last_data(self, timestamp: UTCDateTime):
        try:
            self._last_data_queue.put(timestamp, block=False)
        except Full:
            # Empty it!
            try:
                self._last_data_queue.get(block=False)
            except Empty:
                # Just in case the state changed...
                pass
            try:
                self._last_data_queue.put(timestamp, timeout=10)
            except Full:
                Logger.error("Could not update the state of last data - queue is full")

    @property
    def stream(self) -> Stream:
        try:
            self.__stream = self._stream_queue.get(block=False)
        except Empty:
            Logger.debug("No stream in queue")
            pass
        # If the queue is empty then return current state - this happens
        # if the queue has not been updated since we last checked.
        return self.__stream

    @stream.setter
    def stream(self, st: Stream):
        try:
            self._stream_queue.put(st, block=False)
            Logger.debug("Put stream into queue")
        except Full:
            # Empty it!
            try:
                self._stream_queue.get(block=False)
            except Empty:
                # Just in case the state changed...
                pass
            try:
                self._stream_queue.put(st, timeout=10)
                Logger.debug("Put stream into queue")
            except Full:
                Logger.error(
                    "Could not update the state of stream - queue is full")

    @property
    def buffer_length(self) -> float:
        """
        Return the maximum length of the buffer in seconds.
        """
        st = self.stream
        if len(st) == 0:
            return 0.0
        return max([tr.stats.npts / tr.stats.sampling_rate for tr in st])

    @property
    def buffer_ids(self) -> set:
        return {tr.id for tr in self.stream}

    @property
    def wavebank(self):
        return self.__wavebank

    @wavebank.setter
    def wavebank(self, wavebank: WaveBank):
        self.__wavebank = wavebank
        if wavebank:
            self.has_wavebank = True
        else:
            self.has_wavebank = False

    @abstractmethod
    def copy(self, empty_buffer: bool = True):
        """
        Generate a new - unconnected client.

        Parameters
        ----------
        empty_buffer
            Whether to start the new client with an empty buffer or not.
        """

    @abstractmethod
    def restart(self):
        """
        Disconnect and reconnect and restart the Streaming Client.
        """

    def _access_wavebank(
        self,
        method: str,
        timeout: float = None,
        *args, **kwargs
    ):
        """
        Thread and process safe access to the wavebank.

        Multiple processes cannot access the underlying HDF5 file at the same
        time.  This method waits until access to the HDF5 file is available and

        Parameters
        ----------
        method
            Method of wavebank to call
        timeout
            Maximum time to try to get access to the file
        args
            Arguments passed to method
        kwargs
            Keyword arguments passed to method

        Returns
        -------
        Whatever should be returned by the method.
        """
        if not self.has_wavebank:
            if not self._wavebank_warned:
                Logger.error("No wavebank attached to streamer")
            return None
        timer, wait_step = 0.0, 0.5
        with self.wavebank_lock:
            try:
                func = self.wavebank.__getattribute__(method)
            except AttributeError:
                Logger.error(f"No wavebank method named {method}")
                return None
            # Attempt to access the underlying wavebank
            out = None
            while timer < timeout:
                tic = time.time()
                try:
                    out = func(*args, **kwargs)
                    break
                except (IOError, OSError) as e:
                    time.sleep(wait_step)
                toc = time.time()
                timer += toc - tic
            else:
                Logger.error(
                    f"Waited {timer} s and could not access the wavebank "
                    f"due to {e}")
        return out

    def get_wavebank_stream(self, bulk: List[tuple]) -> Stream:
        """ processsafe get-waveforms-bulk call """
        st = self._access_wavebank(
            method="get_waveforms_bulk", timeout=120., bulk=bulk)
        return st

    def _bg_run(self):
        while self.busy:
            self.run()
        Logger.info("Running stopped, busy set to False")
        return

    def _clear_killer(self):
        """ Clear the killer Queue. """
        while True:
            try:
                self._killer_queue.get(block=False)
            except Empty:
                break

    def background_run(self):
        """Run the client in the background."""
        self.busy, self.started = True, True
        self._clear_killer()   # Clear the kill queue
        streaming_process = multiprocessing.Process(
            target=self._bg_run, name="StreamProcess")
        # streaming_process.daemon = True
        streaming_process.start()
        self.processes.append(streaming_process)
        Logger.info("Started streaming")

    def background_stop(self):
        """Stop the background process."""
        self._stop_called = True
        self._killer_queue.put(True)
        self.stop()
        for process in self.processes:
            Logger.info("Joining process")
            process.join()
            Logger.info("Thread joined")
        self.processes = []

    def on_data(self, trace: Trace):
        """
        Handle incoming data

        Parameters
        ----------
        trace
            New data.
        """
        self.last_data = UTCDateTime.now()
        Logger.debug("Packet of {0} samples for {1}".format(
            trace.stats.npts, trace.id))
        with self.lock:
            Logger.debug(f"Adding data: Lock status: {self.lock}")
            try:
                self.buffer.add_stream(trace)
            except Exception as e:
                Logger.error(
                    f"Could not add {trace} to buffer due to {e}")
            self.buffer_full = self.buffer.is_full()
        if trace.data.dtype == np.int32 and trace.data.dtype.type != np.int32:
            # Cope with a windows error where data come in as 
            # "int32" not np.int32. See https://github.com/obspy/obspy/issues/2683
            trace.data = trace.data.astype(np.int32)
        if self.has_wavebank:
            self._access_wavebank(
                method="put_waveforms", timeout=120., stream=Stream([trace]))
        Logger.debug("Buffer contains {0}".format(self.buffer))
        Logger.debug(f"Finished adding data: Lock status: {self.lock}")
        Logger.debug(f"Buffer stream: \n{self.buffer.stream}")
        self.stream = self.buffer.stream
        Logger.debug(f"Stream: \n{self.stream}")

    def on_terminate(self) -> Stream:  # pragma: no cover
        """
        Handle termination gracefully
        """
        st = self.stream   # Get stream before termination - cannot communicate
        Logger.debug(f"Stream on termination: \n{st}")
        Logger.info("Termination of {0}".format(self.__repr__()))
        if not self._stop_called:  # Make sure we don't double-call stop methods
            if len(self.processes):
                self.background_stop()
            else:
                self.stop()
        else:
            Logger.info("Stop already called - not duplicating")
        self.stream = st
        return st

    @staticmethod
    def on_error():  # pragma: no cover
        """
        Handle errors gracefully.
        """
        Logger.error("Client error")
        pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()
