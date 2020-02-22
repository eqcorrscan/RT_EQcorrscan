"""
Functions and classes for handling streaming of data for real-time
matched-filtering.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import threading
import logging

from abc import ABC, abstractmethod
from typing import Union

from obspy import Stream, Trace
from obsplus import WaveBank

from rt_eqcorrscan.streaming.buffers import Buffer

Logger = logging.getLogger(__name__)


class _StreamingClient(ABC):
    """
    Abstract Base Class for streaming clients

    Parameters
    ----------
    client_name
        Client name, used for book-keeping
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

        self.wavebank = wavebank
        self.threads = []

    def __repr__(self):
        """
        Print information about the client.
        """
        status_map = {True: "Running", False: "Stopped"}
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
        """ Clear the current buffer. """
        self._buffer = Buffer(traces=[], maxlen=self.buffer_capacity)

    @property
    def buffer_full(self) -> bool:
        if len(self.buffer) == 0:
            return False
        return self.buffer.is_full()

    @property
    def buffer_length(self) -> float:
        """
        Return the maximum length of the buffer in seconds.
        """
        if len(self.buffer) == 0:
            return 0.
        return max([tr.data_len for tr in self.buffer])

    @abstractmethod
    def copy(self, empty_buffer: bool = True):
        """
        Generate a new - unconnected client.

        Parameters
        ----------
        empty_buffer
            Whether to start the new client with an empty buffer or not.
        """

    def get_stream(self) -> Stream:
        """ Get a copy of the current data in buffer. """
        return self.buffer.stream

    def _bg_run(self):
        while self.busy:
            self.run()

    def background_run(self):
        """Run the seedlink client in the background."""
        self.busy = True
        streaming_thread = threading.Thread(
            target=self._bg_run, name="StreamThread")
        streaming_thread.daemon = True
        streaming_thread.start()
        self.threads.append(streaming_thread)
        Logger.info("Started streaming")

    def background_stop(self):
        """Stop the background thread."""
        self.stop()
        for thread in self.threads:
            thread.join()

    def on_data(self, trace: Trace):
        """
        Handle incoming data

        Parameters
        ----------
        trace
            New data.
        """
        logging.debug("Packet of {0} samples for {1}".format(
            trace.stats.npts, trace.id))
        self.buffer.add_stream(trace)
        if self.wavebank is not None:
            self.wavebank.put_waveforms(stream=Stream([trace]))
            # Note that this should be undertaken by put_waveforms,
            # but seems to get missed...
            self.wavebank.update_index()
        Logger.debug("Buffer contains {0}".format(self.buffer))

    def on_terminate(self) -> Stream:  # pragma: no cover
        """
        Handle termination gracefully
        """
        Logger.info("Termination of {0}".format(self.__repr__()))
        return self.buffer

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
