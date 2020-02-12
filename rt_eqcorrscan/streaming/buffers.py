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
from collections.abc import Sized

from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats
from obspy.core.util import AttribDict

Logger = logging.getLogger(__name__)


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

    def __repr__(self):  # pragma: no cover
        return self.__str__()

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
            return
        # all other keys
        if isinstance(value, dict):
            super().__setitem__(key, AttribDict(value))
        else:
            super().__setitem__(key, value)

    __setattr__ = __setitem__

    def __str__(self):  # pragma: no cover
        """
        Return better readable string representation of BufferStats object.
        """
        priorized_keys = ['network', 'station', 'location', 'channel',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'calib']
        return self._pretty_str(priorized_keys)

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
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
        Maximum length of trace in samples.

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

    def __repr__(self) -> str:  # pragma: no cover
        return "TraceBuffer(data={0}, header={1}, maxlen={2})".format(
            self.data, self.stats, self.data.maxlen)

    def __len__(self) -> int:
        """ Length of buffer in samples. """
        return len(self.data)

    def __add__(self, other):
        new = self.copy()
        new.add_trace(other)
        return new

    def __iadd__(self, other):
        self.add_trace(other)
        return self

    @property
    def data_len(self) -> float:
        """ Length of buffer in seconds. """
        return self.__len__() * self.stats.delta

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
        >>> print(trace_buffer.data) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[-- -- -- -- -- 0 1 2 3 4 5 6 7 8 9], maxlen=15)
        >>> trace = Trace(
        ...     np.arange(7)[::-1], header=dict(
        ...         station="bob", starttime=UTCDateTime(2018, 1, 1), delta=1.))
        >>> trace_buffer.add_trace(trace)
        >>> print(trace_buffer.stats.endtime)
        2018-01-01T00:00:06.000000Z
        >>> print(trace_buffer.data) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[0 1 2 3 4 5 6 7 6 5 4 3 2 1 0], maxlen=15)

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
        >>> print(trace_buffer.data) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[5 6 7 8 9 10 11 12 13 14 15 16 17 18 19], maxlen=15)

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
        >>> print(trace_buffer.data) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[15 16 17 18 19 -- -- -- -- -- 0 1 2 3 4], maxlen=15)

        Add a trace that starts one sample after the current trace ends

        >>> trace = Trace(
        ...     np.arange(5), header=dict(
        ...         station="bob",
        ...         starttime=trace_buffer.stats.endtime + trace_buffer.stats.delta,
        ...         delta=1.))
        >>> print(trace_buffer.stats.endtime)
        2018-01-01T00:00:34.000000Z
        >>> print(trace.stats.starttime)
        2018-01-01T00:00:35.000000Z
        >>> trace_buffer.add_trace(trace)
        >>> print(trace_buffer.data) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[-- -- -- -- -- 0 1 2 3 4 0 1 2 3 4], maxlen=15)
        """
        if isinstance(trace, TraceBuffer):
            trace = trace.trace
        # Check that stats match
        assert self.id == trace.id, "IDs {0} and {1} differ".format(
            self.id, trace.id)
        assert self.stats.sampling_rate == trace.stats.sampling_rate, (
            "Sampling rates {0} and {1} differ".format(
                self.stats.sampling_rate, trace.stats.sampling_rate))
        assert self.stats.calib == trace.stats.calib, (
            "Calibration factors {0} and {1} differ".format(
                self.stats.calib, trace.stats.calib))
        # Remove older data than our minimum starttime - faster to if this.
        if trace.stats.starttime < self.stats.starttime:
            trace = trace.slice(starttime=self.stats.starttime)
            if len(trace) == 0:
                return
        # If data are newer in trace than in self.
        if trace.stats.endtime > self.stats.endtime:
            # If there is overlap
            if trace.stats.starttime <= self.stats.endtime:
                old_data = trace.slice(endtime=self.stats.endtime).data
                insert_start = -len(old_data)
                self.data.insert(old_data, insert_start)
                new_data = trace.slice(
                    starttime=self.stats.endtime + self.stats.delta).data
            # If there is a gap - defined as more than 1.5 samples. Coping with
            # rounding errors in UTCDateTime.
            elif trace.stats.starttime >= self.stats.endtime + (1.5 * self.stats.delta):
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
            # No new times covered - insert old data into array.
            insert_start = (trace.stats.starttime -
                            self.stats.starttime) * self.stats.sampling_rate
            # Cope with small shifts due to sampling time-stamp rounding
            assert abs(insert_start - round(insert_start)) < .1, \
                "Traces are not sampled at the same base time-stamp, {0} != {1}".format(
                    round(insert_start), insert_start)
            self.data.insert(trace.data, int(round(insert_start)))
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
        return Trace(header=self.stats.__dict__, data=self.data.data.copy())

    def is_full(self, strict=False) -> bool:
        """
        Check if the tracebuffer is full or not.

        If strict=False (default) then only the start and end of the buffer
        are checked.  Otherwise (strict=True) the whole buffer must contain
        real data (e.g. the mask is all False)

        Parameters
        ----------
        strict
            Whether to check the whole deque (True), or just the start and end
            (False: default).
        """
        return self.data.is_full(strict=strict)

    def copy(self):
        """
        Generate a copy of the buffer.

        Returns
        -------
        A deepcopy of the tracebuffer.
        """
        return TraceBuffer(
            data=copy.deepcopy(self.data.data),
            header=copy.deepcopy(self.stats.__dict__),
            maxlen=copy.deepcopy(self.data.maxlen))


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
        self._data = np.empty(maxlen, dtype=type(data[0]))
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
        data_str = self.data.__str__()
        if data_str.startswith("[ "):
            data_str = data_str.replace("[ ", "[")
        return "NumpyDeque(data={0}, maxlen={1})".format(
            data_str, self.maxlen)

    def __len__(self):
        return sum(~self._mask)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def data(self) -> np.ndarray:
        if self._mask.sum() > 0:
            return np.ma.masked_array(self._data, mask=self._mask)
        return self._data

    def is_full(self, strict=False) -> bool:
        """
        Check whether the buffer is full.

        If strict=False (default) then only the start and end of the buffer
        are checked.  Otherwise (strict=True) the whole buffer must contain
        real data (e.g. the mask is all False)

        Parameters
        ----------
        strict
            Whether to check the whole deque (True), or just the start and end
            (False: default).
        """
        if strict:
            return ~np.any(self._mask)
        return ~np.any([self._mask[0], self._mask[-1]])

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
        self._data[0:-other_length] = self._data[other_length:]
        self._mask[0:-other_length] = self._mask[other_length:]
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
        self._data[other_length:] = self._data[0:-other_length]
        self._mask[other_length:] = self._mask[0:-other_length]
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
        if isinstance(other, Sized):
            raise TypeError("other must be a single item, use extend")
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
        if isinstance(other, Sized):
            raise TypeError("other must be a single item, use extendleft")
        self.extendleft([other])

    def insert(
            self,
            other: Union[list, np.ndarray, np.ma.MaskedArray],
            index: int,
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
        index
            Start of the slice in the current deque

        Examples
        --------
        >>> np_deque = NumpyDeque(data=[0, 1, 2], maxlen=5)
        >>> print(np_deque) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[-- -- 0 1 2], maxlen=5)

        Insert a single element list

        >>> np_deque.insert([6], 1)
        >>> print(np_deque) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[-- 6 0 1 2], maxlen=5)

        Insert a numpy array

        >>> np_deque.insert(np.array([11, 12]), 3)
        >>> print(np_deque) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[-- 6 0 11 12], maxlen=5)

        Insert a masked array - only the unmasked elements are used.

        >>> np_deque.insert(
        ...     np.ma.masked_array([99, 99], mask=[True, False]), 2)
        >>> print(np_deque) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[-- 6 0 99 12], maxlen=5)

        Insert an array longer than maxlen

        >>> np_deque.insert(np.arange(10), 0)
        >>> print(np_deque) # doctest: +NORMALIZE_WHITESPACE
        NumpyDeque(data=[5 6 7 8 9], maxlen=5)
        """
        if not isinstance(other, Sized):
            other = [other]
        stop = index + len(other)
        if len(other) > self.maxlen:
            Logger.warning("Array longer than max-length, reverting to extend")
            self.extend(other)
        elif isinstance(other, np.ma.MaskedArray):
            # Only take the non-masked bits
            for i in range(len(other)):
                if not other.mask[i]:
                    self._data[index + i] = other.data[i]
                    self._mask[index + i] = 0
        else:
            if stop == 0:
                self._data[index:] = other
                self._mask[index:] = 0
            else:
                self._data[index:stop] = other
                self._mask[index:stop] = 0


class Buffer(object):
    """
    Container for TraceBuffers.

    Parameters
    ----------
    traces
        Stream or List of TraceBuffers or Traces
    maxlen
        Maximum length for TraceBuffers in seconds.

    Examples
    --------

    >>> from obspy import read
    >>> st = read()
    >>> print(st)
    3 Trace(s) in Stream:
    BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    >>> buffer = Buffer(st, maxlen=10.)
    >>> print(buffer)
    Buffer(3 traces, maxlen=10.0)
    """
    def __init__(
        self,
        traces: Union[Stream, List[Union[Trace, TraceBuffer]]] = None,
        maxlen: float = None
    ):
        assert traces or maxlen, "Requires at least maxlen or traces."
        assert isinstance(traces, (Stream, list)), "Must be either Stream or list"
        for tr in traces:
            assert isinstance(tr, (Trace, TraceBuffer)), "Must be Trace or TraceBuffer"
        self.traces = []
        self._maxlen = maxlen or max(
            [tr.stats.npts * tr.stats.delta for tr in traces])
        for tr in traces:
            if isinstance(tr, Trace):
                self.traces.append(TraceBuffer(
                    data=tr.data, header=tr.stats,
                    maxlen=int(tr.stats.sampling_rate * self._maxlen)))
            elif isinstance(tr, TraceBuffer):
                self.traces.append(tr)
        self.sanitize_traces()

    def __repr__(self):
        return "Buffer({0} traces, maxlen={1})".format(
            self.__len__(), self.maxlen)

    def __iter__(self):
        return self.traces.__iter__()

    def __len__(self):
        return len(self.traces)

    def __add__(self, other: Union[Trace, Stream]):
        new_buffer = self.copy()
        new_buffer.add_stream(other)
        return new_buffer

    def __iadd__(self, other: Union[Trace, Stream]):
        self.add_stream(other)
        return self

    def copy(self):
        return Buffer(traces=[tr.trace for tr in self.traces],
                      maxlen=self.maxlen)

    @property
    def maxlen(self):
        return self._maxlen

    @maxlen.setter
    def maxlen(self, maxlen):
        self._maxlen = maxlen
        self.sanitize_traces()

    def sanitize_traces(self):
        """ Ensure all traces meet that maxlen criteria. """
        _traces = []
        for tr in self.traces:
            _maxlen = int(self.maxlen * tr.stats.sampling_rate)
            if not tr.data.maxlen == _maxlen:
                # Need to make a new tracebuffer with the correct maxlen
                _tr = tr.trace
                _traces.append(TraceBuffer(
                    data=_tr.data, header=_tr.stats, maxlen=_maxlen))
            else:
                _traces.append(tr)
        self.traces = _traces

    def add_stream(self, stream: Union[Trace, Stream]) -> None:
        """
        Add a stream or trace to the buffer.

        Note that if `stream` is a Buffer, the maxlen of the initial Buffer
        will be used.

        Parameters
        ----------
        stream
        """
        if isinstance(stream, Trace):
            stream = Stream([stream])
        elif isinstance(stream, Buffer):
            stream = stream.stream
        for tr in stream:
            traces_in_buffer = self.select(id=tr.id)
            if len(traces_in_buffer) > 0:
                for trace_in_buffer in traces_in_buffer:
                    trace_in_buffer.add_trace(tr)
            else:
                self.traces.append(TraceBuffer(
                    data=tr.data, header=tr.stats,
                    maxlen=int(self.maxlen * tr.stats.sampling_rate)))

    def select(self, id: str) -> List:
        """
        Select traces from the buffer based on seed id

        Parameters
        ----------
        id
            Standard four-part seed id as
            {network}.{station}.{location}.{channel}

        Returns
        -------
        List of matching traces.
        """
        return [tr for tr in self.traces if tr.id == id]

    @property
    def stream(self) -> Stream:
        """
        Get a static Stream view of the buffer

        Returns
        -------
        A stream representing the current state of the Buffer.
        """
        return Stream([tr.trace for tr in self.traces])

    def is_full(self, strict=False) -> bool:
        """
        Check whether the buffer is full or not.

        If strict=False (default) then only the start and end of the buffer
        are checked.  Otherwise (strict=True) the whole buffer must contain
        real data (e.g. the mask is all False)

        Parameters
        ----------
        strict
            Whether to check the whole deque (True), or just the start and end
            (False: default).
        """
        for tr in self.traces:
            if not tr.is_full(strict=strict):
                return False
        return True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
