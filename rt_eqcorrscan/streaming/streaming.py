"""
Functions and classes for handling streaming of data for real-time
matched-filtering.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import logging
import time
import numpy as np
import warnings

from abc import ABC, abstractmethod
from typing import Union
from queue import Empty, Full

from obspy import Stream, Trace, UTCDateTime

from rt_eqcorrscan.streaming.buffers import Buffer

import platform
if platform.system() != "Linux":
    warnings.warn("Currently Process-based streaming is only supported on "
                  "Linux, defaulting to Thread-based streaming - you may run "
                  "into issues when detecting frequently")
    import threading as multiprocessing
    from queue import Queue
    from threading import Thread as Process
else:
    import multiprocessing
    from multiprocessing import Queue, Process

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

    Notes
    -----
        Requires a `run` and `select_stream` method, however, these cannot be
        added here as abstract methods because they clash with instantiation of
        `EasySeedLinkClient`.
    """
    streaming = False
    started = False
    can_add_streams = True
    lock = multiprocessing.Lock()  # Lock for buffer access
    _stop_called = False

    def __init__(
        self,
        server_url: str = None,
        buffer: Union[Stream, Buffer] = None,
        buffer_capacity: float = 600.,
    ) -> None:
        self.server_url = server_url
        if buffer is None:
            buffer = Buffer(traces=[], maxlen=buffer_capacity)
        elif isinstance(buffer, Stream):
            buffer = Buffer(buffer.traces, maxlen=buffer_capacity)
        self._buffer = buffer
        self.buffer_capacity = buffer_capacity

        # Queues for communication

        # Outgoing data
        self._stream_queue = Queue(maxsize=1)
        # Incoming data - no limit on size, just empty it!
        self._incoming_queue = Queue()

        # Quereyable attributes to get a view of the size of the buffer
        self._last_data_queue = Queue(maxsize=1)
        self._buffer_full_queue = Queue(maxsize=1)

        # Poison!
        self._killer_queue = Queue(maxsize=1)
        self._dead_queue = Queue(maxsize=1)

        # Private attributes for properties
        self.__stream = Stream()
        self.__last_data = None
        self.__buffer_full = False

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
                    self.server_url, status_map[self.streaming],
                    self.buffer_capacity, self.buffer))
        return print_str

    @abstractmethod
    def start(self) -> None:
        """ Open the connection to the streaming service. """

    @abstractmethod
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

    @abstractmethod
    def stop(self) -> None:
        """ Stop the system. """

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
            Logger.debug("_last_data is full")
            # Empty it!
            try:
                self._last_data_queue.get(block=False)
            except Empty:
                # Just in case the state changed...
                Logger.debug("_last_data is empty :(")
                pass
            try:
                self._last_data_queue.put(timestamp, timeout=10)
            except Full:
                Logger.error("Could not update the state of last data - queue is full")

    @property
    def stream(self) -> Stream:
        try:
            self.__stream = self._stream_queue.get(block=False)
            # Need to put it back for future Processes!
            try:
                self._stream_queue.put(self.__stream, block=False)
            except Full:
                # Something else has added while we were not looking! Okay
                pass
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

    def _bg_run(self):
        while self.streaming:
            self.run()
        Logger.info("Running stopped, busy set to False")
        try:
            self._dead_queue.get(block=False)
        except Empty:
            pass
        self._dead_queue.put(True)
        return

    def background_run(self):
        """Run the client in the background."""
        self.streaming, self.started, self.can_add_streams = True, True, False
        self._clear_killer()   # Clear the kill queue
        streaming_process = Process(
            target=self._bg_run, name="StreamProcess")
        # streaming_process.daemon = True
        streaming_process.start()
        self.processes.append(streaming_process)
        Logger.info("Started streaming")

    def background_stop(self):
        """Stop the background process."""
        Logger.info("Adding Poison to Kill Queue")
        # Run communications before termination
        st = self.stream
        self.__buffer_full = self.buffer_full
        self.__last_data = self.last_data

        Logger.debug(f"Stream on termination: {st}")
        self._killer_queue.put(True)
        self.stop()
        # Local buffer
        for tr in st:
            Logger.info("Adding trace to local buffer")
            self.buffer.add_stream(tr)
        # Wait until streaming has stopped
        Logger.debug(
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
            Logger.info("Joining process")
            process.join(5)
            if hasattr(process, 'exitcode') and process.exitcode:
                Logger.info("Process failed to join, terminating")
                process.terminate()
                Logger.info("Terminated")
                process.join()
            Logger.info("Process joined")
        self.processes = []
        self.streaming = False

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
        # Put data into queue - get the run process to handle it!
        if self.streaming:
            self._incoming_queue.put(trace)
            Logger.debug("Added trace to queue")
        else:
            # If the streamer is not running somewhere else, then we have to
            # add data in this Process.
            Logger.debug("Not streaming - will add directly now.")
            self._add_trace_to_buffer(trace)

    def _add_data_from_queue(self):
        """
        Check the incoming data queue for any data and add it to the queue!
        """
        # Empty the queue into Process-local memory
        traces = []
        while True:
            try:
                trace = self._incoming_queue.get(block=False)
                Logger.debug(f"Extracted trace from incoming queue: \n{trace}")
                traces.append(trace)
            except Empty:
                Logger.debug("Incoming data queue is empty")
                break
        if len(traces) == 0:
            Logger.debug("No traces extracted from incoming queue")
            return
        for trace in traces:
            self._add_trace_to_buffer(trace)

    def _add_trace_to_buffer(self, trace: Trace):
        """
        Add a trace to the buffer.

        Parameters
        ----------
        trace
            Trace to add to the internal buffer
        """

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


# def _bg_run(streamer: _StreamingClient):
#     Logger.debug(streamer)
#     streamer.run()
#     Logger.info("Running stopped, streaming set to False")
#     return


if __name__ == "__main__":
    import doctest

    doctest.testmod()
