"""
Test for RT_EQcorrscan's buffer implementation
"""

import unittest

from obspy import read

from rt_eqcorrscan.streaming.buffers import (
    Buffer, BufferStats, TraceBuffer, NumpyDeque)


class TestBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.st = read()

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

    def test_reset_maxlen(self) -> None:
        buffer = Buffer(traces=self.st.traces)
        self.assertEqual(buffer.maxlen, 30.)
        buffer.maxlen = 20.
        self.assertEqual(buffer.maxlen, 20.)
        for tr in buffer:
            self.assertEqual(len(tr), 20. * tr.stats.sampling_rate)

    def test_addition(self) -> None:
        st1 = self.st.slice(
            self.st[0].stats.starttime, self.st[0].stats.starttime + 10)
        st2 = self.st.slice(
            self.st[0].stats.starttime + 10, self.st[0].stats.endtime)
        buffer = Buffer(traces=st1, maxlen=30.)
        self.assertEqual(buffer.stream.split(), st1)
        buffer.add_stream(st2)
        buffer_stream = buffer.stream
        for tr in buffer_stream:
            tr.stats.pop("processing")
        self.assertEqual(buffer_stream, self.st)


if __name__ == "__main__":
    unittest.main()
