"""
Classes for buffering seismic data in memory.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging
import copy
import numpy as np

from typing import Union, List

from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats
from obspy.core.util import AttribDict

Logger = logging.getLogger(__name__)


class Buffer(object):
    """
    Container for TraceBuffers.

    Parameters
    ----------
    traces
        List of TraceBuffers
    maxlen
        Maximum length for TraceBuffers in seconds.
    """
    def __init__(self, traces: List = None, maxlen: float = None):
        if traces is None:
            traces = []
        else:
            for tr in traces:
                assert isinstance(tr, TraceBuffer)
        self.traces = traces
        self.maxlen = maxlen

    def add_stream(self, stream: Union[Trace, Stream]) -> None:
        """
        Add a stream or trace to the buffer.

        Parameters
        ----------
        stream

        Returns
        -------

        """
        if isinstance(stream, Trace):
            stream = Stream([stream])
        for tr in stream:
            traces_in_buffer = self.select(seed_id=tr.id)
            if len(traces_in_buffer) > 0:
                for trace_in_buffer in traces_in_buffer:
                    trace_in_buffer.add_trace(tr)
            else:
                self.traces.append(TraceBuffer(
                    data=tr.data, header=tr.stats,
                    maxlen=int(self.maxlen * tr.stats.sampling_rate)))

    def select(self, seed_id: str) -> List:
        """
        Select traces from the buffer based on seed id

        Parameters
        ----------
        seed_id
            Standard four-part seed id as
            {network}.{station}.{location}.{channel}

        Returns
        -------
        List of matching traces.
        """
        return [tr for tr in self.traces if tr.id == seed_id]

    @property
    def stream(self):
        return Stream([tr.trace for tr in self.traces])


class BufferStats(AttribDict):
    """
    Container for header attributes for TraceBuffer.

    This is very similar to obspy's Stats, but endtime is fixed and starttime
    is calculated from other attributes.
    """
    readonly = ['starttime']
    defaults = {
        'sampling_rate': 1.0,
        'delta': 1.0,
        'starttime': UTCDateTime(0),
        'endtime': UTCDateTime(0),
        'npts': 0,
        'calib': 1.0,
        'network': '',
        'station': '',
        'location': '',
        'channel': '',
    }

    def __init__(self, header: dict = None):
        super().__init__(header)

    def __setitem__(self, key, value):
        # keys which need to refresh derived values
        if key in ['delta', 'sampling_rate', 'endtime', 'npts']:
            # ensure correct data type
            if key == 'delta':
                key = 'sampling_rate'
                try:
                    value = 1.0 / float(value)
                except ZeroDivisionError:
                    value = 0.0
            elif key == 'sampling_rate':
                value = float(value)
            elif key == 'endtime':
                value = UTCDateTime(value)
            elif key == 'npts':
                if not isinstance(value, int):
                    value = int(value)
            # set current key
            super().__setitem__(key, value)
            # set derived value: delta
            try:
                delta = 1.0 / float(self.sampling_rate)
            except ZeroDivisionError:
                delta = 0
            self.__dict__['delta'] = delta
            # set derived value: starttime
            if self.npts == 0:
                timediff = 0
            else:
                timediff = float(self.npts - 1) * delta
            self.__dict__['starttime'] = self.endtime - timediff
            return
        # prevent a calibration factor of 0
        if key == 'calib' and value == 0:
            Logger.warning('Calibration factor set to 0.0!')
        # all other keys
        if isinstance(value, dict):
            super().__setitem__(key, AttribDict(value))
        else:
            super().__setitem__(key, value)

    __setattr__ = __setitem__

    def __str__(self):
        """
        Return better readable string representation of BufferStats object.
        """
        priorized_keys = ['network', 'station', 'location', 'channel',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'calib']
        return self._pretty_str(priorized_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class TraceBuffer(object):
    """
    Container for Trace-like data
    Note: does not include standard Trace methods, use TraceBuffer.trace.`meth`

    Parameters
    ----------
    data
        Iterable of data - will be made into a deque.
    header
        Standard header info for an Obspy Trace.
    maxlen
        Maximum length of trace in sample.

    Examples
    --------
    >>> from obspy import read
    >>> st = read()
    >>> trace_buffer = TraceBuffer(
    ...     data=st[0].data, header=st[0].stats, maxlen=100)
    >>> len(trace_buffer)
    100
    >>> trace_buffer.stats.npts
    100
    >>> trace_buffer.stats.endtime
    UTCDateTime(2009, 8, 24, 0, 20, 32, 990000)
    >>> trace_buffer.stats.starttime
    UTCDateTime(2009, 8, 24, 0, 20, 32)
    >>> assert np.all(trace_buffer.data.data == st[0].data[-100:])
    """
    def __init__(
        self,
        data: np.ndarray,
        header: Union[dict, Stats, BufferStats],
        maxlen: int
    ):
        # Take the right-most samples
        self.data = NumpyDeque(data, maxlen=maxlen)
        header = copy.deepcopy(header)
        # We need to make sure that starttime is correctly set
        self.stats = BufferStats(header)
        self.stats.npts = len(self.data.data)

    def __repr__(self) -> str:
        return "TraceBuffer(data={0}, header={1}, maxlen={2})".format(
            self.data, self.stats, self.data.maxlen)

    def __len__(self) -> int:
        return len(self.data.data)

    def get_id(self) -> str:
        """
        Return a SEED compatible identifier of the trace.

        Returns
        -------
            The SEED ID of the trace.
        """
        seed_formatter = "{network}.{station}.{location}.{channel}"
        return seed_formatter.format(**self.stats)

    id = property(get_id)

    @id.setter
    def id(self, value):
        try:
            net, sta, loc, cha = value.split(".")
        except AttributeError:
            msg = ("Can only set a Trace's SEED ID from a string "
                   "(and not from {})").format(type(value))
            raise TypeError(msg)
        except ValueError:
            msg = "Not a valid SEED ID: '{}'".format(value)
            raise ValueError(msg)
        self.stats.network = net
        self.stats.station = sta
        self.stats.location = loc
        self.stats.channel = cha

    def add_trace(self, trace: Trace) -> None:
        """
        Add one TraceBuffer to another.

        If overlaps occur, new-data (from other) are kept and old-data
        (from self) are replaced.

        Parameters
        ----------
        trace
            New trace to add to the buffer - will be added in place.

        Examples
        --------
        >>> from obspy import Trace, UTCDateTime
        >>> import numpy as np
        >>> trace_buffer = TraceBuffer(
        ...     data=np.arange(10), header=dict(
        ...         station="bob", endtime=UTCDateTime(2018, 1, 1, 0, 0, 1),
        ...         delta=1.),
        ...     maxlen=15)
        >>> print(trace_buffer.data)
        NumpyDeque(data=[-- -- -- -- -- 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0], maxlen=15)
        >>> trace = Trace(
        ...     np.arange(7)[::-1], header=dict(
        ...         station="bob", starttime=UTCDateTime(2018, 1, 1), delta=1.))
        >>> trace_buffer.add_trace(trace)
        >>> print(trace_buffer.stats.endtime)
        2018-01-01T00:00:06.000000Z
        >>> print(trace_buffer.data)
        NumpyDeque(data=[1. 2. 3. 4. 5. 6. 7. 8. 6. 5. 4. 3. 2. 1. 0.], maxlen=15)

        Try adding a trace that is longer than the maxlen
        >>> trace = Trace(
        ...     np.arange(20), header=dict(
        ...         station="bob", starttime=trace_buffer.stats.endtime,
        ...         delta=1.))
        >>> print(trace.stats.endtime)
        2018-01-01T00:00:25.000000Z
        >>> trace_buffer.add_trace(trace)
        >>> print(trace_buffer.stats.endtime)
        2018-01-01T00:00:25.000000Z
        >>> print(trace_buffer.data)
        NumpyDeque(data=[ 5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19.], maxlen=15)

        Add a trace that starts after the current tracebuffer ends
        >>> trace = Trace(
        ...     np.arange(5), header=dict(
        ...         station="bob", starttime=trace_buffer.stats.endtime + 5,
        ...         delta=1.))
        >>> print(trace.stats.endtime)
        2018-01-01T00:00:34.000000Z
        >>> trace_buffer.add_trace(trace)
        >>> print(trace_buffer.stats.endtime)
        2018-01-01T00:00:34.000000Z
        >>> print(trace_buffer.data)
        NumpyDeque(data=[15.0 16.0 17.0 18.0 19.0 -- -- -- -- -- 0.0 1.0 2.0 3.0 4.0], maxlen=15)
        """
        # Check that stats match
        assert self.id == trace.id, "IDs {0} and {1} differ".format(
            self.id, trace.id)
        assert self.stats.sampling_rate == trace.stats.sampling_rate, (
                "Sampling rates {0} and {1} differ".format(
                    self.stats.sampling_rate, trace.stats.sampling_rate))
        assert self.stats.calib == trace.stats.calib, (
                "Calibration factors {0} and {1} differ".format(
                    self.stats.calib, trace.stats.calib))
        # If data are newer in trace than in self.
        if trace.stats.endtime > self.stats.endtime:
            # If there is overlap
            if trace.stats.starttime < self.stats.endtime:
                old_data = trace.slice(
                    endtime=self.stats.endtime - self.stats.delta).data
                if len(old_data) > self.data.maxlen:
                    old_data_start = -self.data.maxlen
                    insert_start = 0
                else:
                    old_data_start = 0
                    insert_start = -len(old_data)
                self.data.insert(
                    old_data[old_data_start:], insert_start, self.data.maxlen)
                new_data = trace.slice(starttime=self.stats.endtime).data
            # If there is a gap.
            elif trace.stats.starttime > self.stats.endtime:
                new_data = np.empty(
                    trace.stats.npts +
                    int(self.stats.sampling_rate *
                        (trace.stats.starttime - self.stats.endtime)),
                    dtype=trace.data.dtype)
                mask = np.ones_like(new_data)
                new_data[-trace.stats.npts:] = trace.data
                mask[-trace.stats.npts:] = 0
                new_data = np.ma.masked_array(new_data, mask=mask)
            # Otherwise just extend with the new data.
            else:
                new_data = trace.data
            self.data.extend(new_data)
            self.stats.endtime = trace.stats.endtime
        else:
            trim_start = self.stats.endtime - (
                    self.data.maxlen * self.stats.delta)
            old_data = trace.slice(starttime=trim_start).data
            insert_start = int(self.stats.sampling_rate * (
                    self.stats.endtime - trim_start))
            insert_stop = insert_start + len(old_data)
            self.data.insert(old_data, insert_start, insert_stop)
        self.stats.npts = len(self.data.data)

    @property
    def trace(self) -> Trace:
        """
        Get a static trace representation of the buffer.

        Returns
        -------
        A trace with the buffer's data and stats. If there are gaps in the
        buffer they will be masked.
        """
        return Trace(header=self.stats.__dict__, data=self.data.data)

    def copy(self):
        """
        Generate a copy of the buffer.

        Returns
        -------
        A deepcopy of the tracebuffer.
        """
        return copy.deepcopy(self)


class NumpyDeque(object):
    """
    Simple implementation of necessary deque methods for 1D numpy arrays.

    Parameters
    ----------
    data
        Data to initialize the deque with
    maxlen
        Maximum length of the deque.
    """
    def __init__(
        self,
        data: Union[list, np.ndarray, np.ma.MaskedArray],
        maxlen: int
    ):
        self._maxlen = maxlen
        self._data = np.empty(maxlen)
        length = min(maxlen, len(data))
        if isinstance(data, np.ma.MaskedArray):
            mask_value = data.mask
            data = data.data
        else:
            mask_value = np.zeros(length)
        self._data[-length:] = data[-length:]
        self._mask = np.zeros_like(self._data, dtype=np.bool)
        self._mask[0:-length] = np.ones(maxlen - length)
        self._mask[-length:] = mask_value

    def __repr__(self):
        return "NumpyDeque(data={0}, maxlen={1})".format(
            self.data, self.maxlen)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def data(self) -> np.ndarray:
        if self._mask.sum() > 0:
            return np.ma.masked_array(self._data, mask=self._mask)
        return self._data

    def extend(
        self,
        other: Union[list, np.ndarray, np.ma.MaskedArray]
    ) -> None:
        """
        Put new data onto the right of the deque.

        If the deque overflows maxlen the leftmost items will be removed.
        Works in place.

        Parameters
        ----------
        other
            The new data to extend the deque with.
        """
        other_length = len(other)
        if isinstance(other, np.ma.MaskedArray):
            mask_value = other.mask
            other = other.data
        else:
            mask_value = np.zeros(other_length)
        if other_length >= self.maxlen:
            self._data[0:] = other[-self.maxlen:]
            self._mask[:] = mask_value[-self.maxlen:]
            return
        self._data[:] = np.roll(self._data[::-1], other_length)[::-1]
        self._mask[:] = np.roll(self._mask[::-1], other_length)[::-1]
        self._data[-other_length:] = other
        self._mask[-other_length:] = mask_value

    def extendleft(
        self,
        other: Union[list, np.ndarray, np.ma.MaskedArray]
    ) -> None:
        """
        Put new data onto the left of the deque.

        If the deque overflows maxlen the rightmost items will be removed.
        Works in place.

        Parameters
        ----------
        other
            The new data to extend the deque with.
        """
        other_length = len(other)
        if isinstance(other, np.ma.MaskedArray):
            mask_value = other.mask
            other = other.data
        else:
            mask_value = np.zeros(other_length)
        if other_length >= self.maxlen:
            self._data[0:] = other[0:self.maxlen]
            self._mask[:] = mask_value[0:self.maxlen]
            return
        self._data[:] = np.roll(self._data, other_length)
        self._mask[:] = np.roll(self._mask, other_length)
        self._data[0:other_length] = other
        self._mask[0:other_length] = mask_value

    def append(self, other) -> None:
        """
        Append an element to the right of the deque.

        If the new deque overflows, the leftmost element will be removed.

        Parameters
        ----------
        other
            Single element to add to the deque.
        """
        self.extend([other])

    def appendleft(self, other) -> None:
        """
        Append an element to the left of the deque.

        If the new deque overflows, the rightmost element will be removed.

        Parameters
        ----------
        other
            Single element to add to the deque.
        """
        self.extendleft([other])

    def insert(
        self,
        other: Union[list, np.ndarray, np.ma.MaskedArray],
        start: int,
        stop: int
    ) -> None:
        """
        Insert elements between a stop and start point of the deque.

        If `other` is a masked array, only unmasked elements will be inserted
        into the deque.

        If `other` is longer than `self.maxlen` then it will be used to extend
        the deque.

        Parameters
        ----------
        other
            Elements to insert.
        start
            Start of the slice in the current deque
        stop
            End of the slice in the current deque

        Examples
        --------
        >>> np_deque = NumpyDeque(data=[0, 1, 2], maxlen=5)
        >>> print(np_deque)
        NumpyDeque(data=[-- -- 0.0 1.0 2.0], maxlen=5)

        Insert a single element list
        >>> np_deque.insert([6], 1, 2)
        >>> print(np_deque)
        NumpyDeque(data=[-- 6.0 0.0 1.0 2.0], maxlen=5)

        Insert a numpy array
        >>> np_deque.insert(np.array([11, 12]), 3, 5)
        >>> print(np_deque)
        NumpyDeque(data=[-- 6.0 0.0 11.0 12.0], maxlen=5)

        Insert a masked array - only the unmasked elements are used.
        >>> np_deque.insert(
        ...     np.ma.masked_array([99, 99], mask=[True, False]), 2, 4)
        >>> print(np_deque)
        NumpyDeque(data=[-- 6.0 0.0 99.0 12.0], maxlen=5)

        Insert an array longer than maxlen
        >>> np_deque.insert(np.arange(10), 0, 10)
        >>> print(np_deque)
        NumpyDeque(data=[5. 6. 7. 8. 9.], maxlen=5)
        """
        if len(other) > self.maxlen:
            Logger.warning("Array longer than max-length, reverting to extend")
            self.extend(other)
        elif isinstance(other, np.ma.MaskedArray):
            # Only take the non-masked bits
            for i in range(len(other)):
                if not other.mask[i]:
                    self._data[start + i] = other.data[i]
                    self._mask[start + i] = 0
        else:
            self._data[start:stop] = other
            self._mask[start:stop] = 0


if __name__ == "__main__":
    import doctest
    doctest.testmod()