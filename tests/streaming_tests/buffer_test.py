"""
Test for RT_EQcorrscan's buffer implementation
"""

import unittest
import numpy as np

from obspy import read, UTCDateTime

from rt_eqcorrscan.streaming.buffers import (
    Buffer, BufferStats, TraceBuffer, NumpyDeque)


class TestBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.st = read()
        cls.st1 = cls.st.slice(
            cls.st[0].stats.starttime, cls.st[0].stats.starttime + 10)
        cls.st2 = cls.st.slice(
            cls.st[0].stats.starttime + 10, cls.st[0].stats.endtime)

    def test_buffer_init_with_stream(self) -> None:
        buffer = Buffer(traces=self.st)
        self.assertEqual(buffer.maxlen, 30.)

    def test_buffer_init_with_traces(self) -> None:
        buffer = Buffer(traces=self.st.traces)
        self.assertEqual(buffer.maxlen, 30.)
        for tr in buffer:
            self.assertEqual(len(tr), 30. * tr.stats.sampling_rate)

    def test_buffer_init_with_traces_maxlen(self) -> None:
        buffer = Buffer(traces=self.st.traces, maxlen=20.)
        self.assertEqual(buffer.maxlen, 20.)
        for tr in buffer:
            self.assertEqual(len(tr), 20. * tr.stats.sampling_rate)

    def test_init_with_trace_buffer(self):
        buffer = Buffer(
            traces=[TraceBuffer(
                data=tr.data, header=tr.stats, maxlen=tr.stats.npts)
                for tr in self.st], maxlen=30.)
        self.assertEqual(buffer.stream, self.st)

    def test_failed_init(self):
        with self.assertRaises(AssertionError):
            Buffer(["a"])

    def test_reset_maxlen(self) -> None:
        buffer = Buffer(traces=self.st.traces)
        self.assertEqual(buffer.maxlen, 30.)
        buffer.maxlen = 20.
        self.assertEqual(buffer.maxlen, 20.)
        for tr in buffer:
            self.assertEqual(len(tr), 20. * tr.stats.sampling_rate)

    def test_addition(self) -> None:
        buffer = Buffer(traces=self.st1, maxlen=30.)
        self.assertEqual(buffer.stream.split(), self.st1)
        buffer.add_stream(self.st2)
        buffer_stream = buffer.stream
        for tr in buffer_stream:
            tr.stats.pop("processing")
        self.assertEqual(buffer_stream, self.st)

    def test_add_new_trace(self):
        buffer = Buffer(self.st)
        new_trace = self.st[0].copy()
        new_trace.stats.station = "NOOB"
        buffer.add_stream(new_trace)
        self.assertEqual(len(buffer), 4)
        buffer_stream = buffer.stream
        self.assertEqual(buffer_stream.select(station="NOOB")[0], new_trace)
        for tr in buffer_stream:
            if tr.stats.station == "NOOB":
                continue
            else:
                self.assertEqual(tr, self.st.select(id=tr.id)[0])

    def test_add_buffer(self):
        buffer1 = Buffer(traces=self.st1, maxlen=30.)
        buffer2 = Buffer(traces=self.st2, maxlen=25.)
        buffer3 = buffer1 + buffer2
        self.assertEqual(buffer1.stream.split(), self.st1)
        self.assertEqual(buffer2.stream.split(), self.st2)
        buffer_stream = buffer3.stream
        for tr in buffer_stream:
            tr.stats.pop("processing")
        self.assertEqual(buffer_stream, self.st)

    def test_full(self):
        buffer = Buffer(traces=self.st1, maxlen=30.)
        self.assertFalse(buffer.is_full())
        buffer += self.st2
        self.assertTrue(buffer.is_full())


class TestBufferStats(unittest.TestCase):
    def base_stats(self):
        stats = BufferStats(
            dict(npts=101, endtime=UTCDateTime(2015, 9, 26),
                 station="GCSZ", network="NZ", location="10", channel="EHZ",
                 sampling_rate=100.0))
        return stats

    def test_updating_endtime(self):
        stats = self.base_stats()
        self.assertEqual(stats.endtime - stats.starttime, 1.0)
        stats.endtime += 86400
        self.assertEqual(stats.endtime, UTCDateTime(2015, 9, 27))
        self.assertEqual(stats.endtime - stats.starttime, 1.0)

    def test_changing_npts(self):
        stats = self.base_stats()
        stats.npts = 201.
        self.assertEqual(stats.endtime - stats.starttime, 2.0)
        self.assertEqual(stats.endtime, UTCDateTime(2015, 9, 26))

    def test_set_zero_delta(self):
        stats = self.base_stats()
        stats.delta = 0.0
        self.assertEqual(stats.sampling_rate, 0.0)

    def test_zero_calib(self):
        stats = self.base_stats()
        stats.calib = 0.0
        self.assertEqual(stats, self.base_stats())

    def test_set_with_dict(self):
        stats = self.base_stats()
        stats.misc = dict(albert="walrous")


class TestTraceBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.st = read()

    def trace_buffer(self):
        return TraceBuffer(data=self.st[0].data, header=self.st[0].stats,
                           maxlen=self.st[0].stats.npts)

    def test_data_len(self):
        trace_buffer = self.trace_buffer()
        self.assertEqual(
            trace_buffer.data_len - trace_buffer.stats.delta,
            trace_buffer.stats.endtime - trace_buffer.stats.starttime)

    def test_set_id(self):
        trace_buffer = self.trace_buffer()
        trace_buffer.id = "RE.ALF.10.HHZ"
        self.assertEqual(trace_buffer.stats.station, "ALF")
        self.assertEqual(trace_buffer.stats.network, "RE")
        self.assertEqual(trace_buffer.stats.location, "10")
        self.assertEqual(trace_buffer.stats.channel, "HHZ")

    def test_set_id_with_not_string(self):
        trace_buffer = self.trace_buffer()
        with self.assertRaises(TypeError):
            trace_buffer.id = 124

    def test_set_id_with_invalid_string(self):
        trace_buffer = self.trace_buffer()
        with self.assertRaises(ValueError):
            trace_buffer.id = "alf.bob"

    def test_add_lots_of_old_data(self):
        trace_buffer = self.trace_buffer()
        maxlen = trace_buffer.data.maxlen
        additional_trace = self.trace_buffer().trace
        additional_trace.stats.starttime -= 20
        additional_trace.data = np.concatenate(
            [additional_trace.data, np.random.randn(4000)])
        trace_buffer += additional_trace
        self.assertEqual(len(trace_buffer), maxlen)

    def test_add_without_changing_original(self):
        trace_buffer1 = TraceBuffer(
            data=self.st[0].data, header=self.st[0].stats,
            maxlen=5000)
        trace_buffer2 = TraceBuffer(
            data=self.st[1].data, header=self.st[0].stats,
            maxlen=self.st[0].stats.npts)
        trace_buffer3 = trace_buffer1 + trace_buffer2
        self.assertNotEqual(trace_buffer3, trace_buffer1)


class TestNumpyDeque(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.maxlen = 200

    def test_extend_left(self):
        deque_1 = NumpyDeque(np.arange(20), maxlen=self.maxlen)
        deque_1.extendleft(np.arange(10))
        self.assertEqual(len(deque_1), 20)
        self.assertEqual(len(deque_1.data), 200)
        self.assertTrue(deque_1.is_full())
        self.assertFalse(deque_1.is_full(strict=True))
        self.assertTrue(np.all(deque_1[0:10] == deque_1[-10:]))

    def test_extend_left_with_masked(self):
        deque_1 = NumpyDeque(np.arange(20), maxlen=self.maxlen)
        masked_array = np.ma.MaskedArray(
            data=np.arange(20), mask=np.ones(20))
        masked_array.mask[0:10] = False
        deque_1.extendleft(masked_array)
        # There should now be only the 0-9 in the first 10 slots and nothing
        # else
        self.assertTrue(np.all(deque_1[0:10] == np.arange(10)))
        self.assertTrue(np.all(deque_1.data.compressed() == np.arange(10)))

    def test_extend_left_too_long(self):
        deque_1 = NumpyDeque(np.arange(20), maxlen=30)
        other = np.arange(40)
        deque_1.extendleft(other)
        self.assertTrue(np.all(deque_1.data == other[0:30]))

    def test_append(self):
        deque_1 = NumpyDeque(np.arange(20), maxlen=20)
        deque_1.append(99)
        self.assertEqual(deque_1[-1], 99)

    def test_append_left(self):
        deque_1 = NumpyDeque(np.arange(20), maxlen=20)
        deque_1.appendleft(99)
        self.assertEqual(deque_1[0], 99)

    def test_append_with_list(self):
        deque_1 = NumpyDeque(np.arange(20), maxlen=20)
        with self.assertRaises(TypeError):
            deque_1.append([99])

    def test_append_left_with_list(self):
        deque_1 = NumpyDeque(np.arange(20), maxlen=20)
        with self.assertRaises(TypeError):
            deque_1.appendleft([99])

    def test_insert_single_item(self):
        deque_1 = NumpyDeque(np.arange(20), maxlen=20)
        deque_1.insert(99, 10)
        self.assertEqual(deque_1[10], 99)


if __name__ == "__main__":
    unittest.main()
