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

from obspy import Stream, Trace

from rt_eqcorrscan.streaming.buffers import Buffer

Logger = logging.getLogger(__name__)


#TODO: Save data to a wavebank.
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
        Length of buffer in seconds. Old data are removed in a LIFO style.

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
        client_name: str = None,
        buffer: Stream = None,
        buffer_capacity: float = 600.,
    ) -> None:
        self.client_name = client_name
        if buffer is None:
            # buffer = Buffer()
            biffer = Stream()
        # elif isinstance(buffer, Stream):
        #     buffer = Buffer(buffer.traces)
        self.buffer = buffer
        self.buffer_capacity = buffer_capacity
        self.threads = []

    def __repr__(self):
        """
        Print information about the client.
        """
        status_map = {True: "Running", False: "Stopped"}
        print_str = (
            "Client at {0}, status: {1}, buffer capacity: {2:.1f}s\n"
            "\tCurrent Buffer:\n{3}".format(
                self.client_name, status_map[self.busy],
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
    def buffer_full(self) -> bool:
        if len(self.buffer) == 0:
            return False
        for tr in self.buffer:
            if tr.stats.endtime - tr.stats.starttime < self.buffer_capacity:
                return False
        return True

    @property
    def buffer_length(self) -> float:
        """
        Return the maximum length of the buffer
        """
        if len(self.buffer) == 0:
            return 0
        return (max([tr.stats.endtime for tr in self.buffer]) -
                min([tr.stats.starttime for tr in self.buffer]))

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
        return self.buffer.copy()

    def _bg_run(self):
        e = None
        while self.busy:
            try:
                self.run()
            except Exception as e:
                break
        if e is not None:
            raise e

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

    # TODO: Save to wavebank if required
    def on_data(self, trace: Trace):
        """
        Handle incoming data

        Parameters
        ----------
        trace
            New data.
        """
        """
        TODO: This is not memory efficient - should
        1. Add trace to stream if trace not in stream already
        2. Append data until buffer full for trace
        3. Once buffer is full for trace should shift data and put new data in,
           NOT add - this creates additional objects.
        """
        logging.debug("Packet of {0} samples for {1}".format(
            trace.stats.npts, trace.id))
        self.buffer += trace
        self.buffer.merge()
        _tr = self.buffer.select(id=trace.id)[0]
        if _tr.stats.npts * _tr.stats.delta > self.buffer_capacity:
            Logger.debug(
                "Trimming trace to {0}-{1}".format(
                    _tr.stats.endtime - self.buffer_capacity,
                    _tr.stats.endtime))
            _tr.trim(_tr.stats.endtime - self.buffer_capacity)
        else:
            Logger.debug("Buffer contains {0}".format(self.buffer))

    def on_terminate(self) -> Stream:  # pragma: no cover
        """
        Handle termination gracefully
        """
        Logger.info("Termination of {0}".format(self.__repr__()))
        return self.buffer

    def on_error(self):  # pragma: no cover
        """
        Handle errors gracefully.
        """
        Logger.error("Client error")
        pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()
