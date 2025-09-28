"""
Tests for waveform database access.
"""

import os
import shutil
import unittest
import logging

from collections import Counter

from obspy.clients.fdsn import Client
from obspy import UTCDateTime

from obsplus import WaveBank

from rt_eqcorrscan.plugins.waveform_access import InMemoryWaveBank


Logger = logging.getLogger(__name__)


class TestWaveformAccess(unittest.TestCase):
    clean_up = []

    @classmethod
    def setUpClass(cls):
        cls.n_stations = 10
        cls.wavebank_path = ".test_waveform_access_wavebank"
        client = Client("GEONET")
        cls.cat = client.get_events(
            minlatitude=-41, maxlatitude=-40, minlongitude=175,
            maxlongitude=176, starttime=UTCDateTime(2024, 1, 1),
            endtime=UTCDateTime(2024, 1, 2))
        bulk = Counter(
            (p.waveform_id.network_code or "*",
             p.waveform_id.station_code or "*",
             p.waveform_id.location_code or "*",
             p.waveform_id.channel_code or "*",
             UTCDateTime(2024, 1, 1).datetime,
             UTCDateTime(2024, 1, 2).datetime)
            for event in cls.cat for p in event.picks)
        bulk = [b[0] for b in bulk.most_common(cls.n_stations)]
        bulk = [(b[0], b[1], b[2], b[3], UTCDateTime(b[4]), UTCDateTime(b[5]))
                for b in bulk]
        st = client.get_waveforms_bulk(bulk)
        cls.wavebank = WaveBank(cls.wavebank_path)
        # Break into chunks
        st_start = min(tr.stats.starttime for tr in st)
        st_end = max(tr.stats.endtime for tr in st)
        chunk_len, _start, _end = 600, st_start, st_start + 600
        while _end <= st_end:
            chunk = st.slice(_start, _end)
            cls.wavebank.put_waveforms(chunk)
            _start += chunk_len
            _end += chunk_len
        cls.clean_up.append(cls.wavebank_path)

    def test_waveform_access(self):
        wb = InMemoryWaveBank(self.wavebank_path)
        wb.get_data_availability()
        self.assertEqual(len(wb.data_availability), self.n_stations)

    def test_get_data(self):
        wb = InMemoryWaveBank(self.wavebank_path)
        wb.get_data_availability()
        st = wb.get_event_waveforms(event=self.cat[0], pre_pick=5, length=25)
        self.assertGreater(len(st), 0)

    @classmethod
    def tearDownClass(cls, clean=True):
        if clean:
            for thing in cls.clean_up:
                if os.path.isdir(thing):
                    shutil.rmtree(thing)
                else:
                    os.remove(thing)


if __name__ == "__main__":
    unittest.main()
