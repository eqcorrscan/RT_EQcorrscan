"""
Utilities for emulating obspy clients using local data. Relies on obsplus
"""

import logging
import os
import fnmatch

from typing import Union, Iterable, List
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, read
from obsplus.bank import WaveBank, EventBank, StationBank


Logger = logging.getLogger(__name__)


class ClientBank(object):
    """
    Thin routing wrapper for obsplus Banks to act as a client.

    Parameters
    ----------
    wave_bank
        WaveBank with seismic data
    event_bank
        EventBank with event data
    station_bank
        StationBank with station data. Note that as of 16/07/2019 StationBank
        was incomplete

    Notes
    -----
        All attributes can be substituted for different (or the same) client.
    """
    def __init__(
        self,
        wave_bank: Union[Client, WaveBank] = None,
        event_bank: Union[Client, EventBank] = None,
        station_bank: Union[Client, StationBank] = None,
    ):
        self.wave_bank = wave_bank
        self.station_bank = station_bank
        self.event_bank = event_bank
        self.base_url = "I'm not a real client!"

    def get_stations(self, *args, **kwargs):
        if self.station_bank is None:
            raise NotImplementedError("No station_bank provided")
        return self.station_bank.get_stations(*args, **kwargs)

    def get_stations_bulk(self, *args, **kwargs):
        if self.station_bank is None:
            raise NotImplementedError("No station_bank provided")
        return self.station_bank.get_stations_bulk(*args, **kwargs)

    def get_waveforms(self, *args, **kwargs):
        if self.wave_bank is None:
            raise NotImplementedError("No wave_bank provided")
        return self.wave_bank.get_waveforms(*args, **kwargs)

    def get_waveforms_bulk(self, *args, **kwargs):
        if self.wave_bank is None:
            raise NotImplementedError("No wave_bank provided")
        return self.wave_bank.get_waveforms_bulk(*args, **kwargs)

    def get_events(self, *args, **kwargs):
        if self.event_bank is None:
            raise NotImplementedError("No event_bank provided")
        return self.event_bank.get_events(*args, **kwargs)


def _stream_info_from_file(filename: str) -> dict:
    """ Extract start, end and seed id from file """
    Logger.debug(f"Reading from {filename}")
    waveform_db = dict()
    try:
        st = read(filename, headonly=True)
    except Exception as e:
        Logger.warning(f"Could not read {filename} due to {e}")
        return waveform_db
    for tr in st:
        nslc = tr.id
        starttime, endtime = tr.stats.starttime, tr.stats.endtime
        tr_db = waveform_db.get(nslc, dict())
        tr_db.update({(starttime.datetime, endtime.datetime): filename})
        waveform_db.update({nslc: tr_db})
    return waveform_db


class LocalClient(object):
    """
    Thin local waveform client.
    """
    _waveform_db = dict()

    def __init__(
        self, 
        base_path: str,
        max_threads: int = None,
    ):
        self.base_path = base_path
        if max_threads and max_threads > 1:
            self._executor = ThreadPoolExecutor(max_workers=max_threads)
        else:
            self._executor = None
        Logger.info("Building database for LocalClient")
        self._build_db()
        Logger.info(f"Have data between {self.starttime} and {self.endtime}")
        self.base_url = "I'm not a real client!"

    def _build_db(self):
        for dirpath, dirnames, filenames in os.walk(self.base_path):
            if len(filenames) == 0:
                continue
            filenames = [os.path.abspath(os.path.join(dirpath, f)) for f in filenames]
            if self._executor:
                future_dbs = [
                    (f, self._executor.submit(_stream_info_from_file, f))
                    for f in filenames]
                for f, future_db in future_dbs:
                    try:
                        db = future_db.result()
                    except Exception as e:
                        Logger.error(f"Could not extract file info from {f} due to {e}")
                        continue
                    else:
                        for nslc, tr_db in db.items():
                            main_tr_db = self._waveform_db.get(nslc, dict())
                            main_tr_db.update(tr_db)
                            self._waveform_db.update({nslc: main_tr_db})
            else:
                for f in filenames:
                    db = _stream_info_from_file(f)
                    for nslc, tr_db in db.items():
                        main_tr_db = self._waveform_db.get(nslc, dict())
                        main_tr_db.update(tr_db)
                        self._waveform_db.update({nslc: main_tr_db})

        return

    @property
    def starttime(self):
        return min(k[0] for tr_db in self._waveform_db.values() for k in tr_db.keys())

    @property
    def endtime(self):
        return max(k[1] for tr_db in self._waveform_db.values() for k in tr_db.keys())

    def _file_reader(
        self,
        files: Iterable, 
        starttime: UTCDateTime, 
        endtime: UTCDateTime
    ) -> Stream:
        st = Stream()
        if self._executor:
            future_streams = [
                (f, self._executor.submit(_cache_read, f))
                for f in files]
            for f, future_stream in future_streams:
                try:
                    st += future_stream.result().copy()
                except Exception as e:
                    Logger.warning(f"Could not read {f} due to {e}")
        else:
            for f in files:
                try:
                    st += _cache_read(f).copy()  # Note, need to add a copy, otherwise
                    # anything done to stream in the future will affect the
                    # cached version as well.
                except Exception as e:
                    Logger.warning(f"Could not read {f} due to {e}")
        return st.merge().trim(starttime, endtime)

    def _db_lookup(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
    ) -> List:
        tr_id = f"{network}.{station}.{location}.{channel}"
        # Need to be able to match wildcards
        known_tr_ids = self._waveform_db.keys()
        matched_tr_ids = fnmatch.filter(known_tr_ids, tr_id)
        files = []
        for key in matched_tr_ids:
            tr_db = self._waveform_db.get(key)
            files.extend(sorted([
                value for key, value in tr_db.items()
                if key[0] <= starttime.datetime <= key[1]  # start within file
                or key[0] <= endtime.datetime <= key[1]  # end within file
                or (starttime <= key[0] and endtime >= key[1])  # file between start and end
            ]))
        return files

    def get_waveforms(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        *args, **kwargs
    ) -> Stream:
        # Logger.info(
        #     f"Querying for data with args: "
        #     f"({network}, {station}, {location}, {channel}, {starttime}, {endtime}")
        files = self._db_lookup(
            network, station, location, channel, starttime, endtime)
        return self._file_reader(files, starttime, endtime)

    def get_waveforms_bulk(
        self,
        bulk: Iterable,
        *args, **kwargs
    ) -> Stream:
        files = []
        for _b in bulk:
            files.extend(
                self._db_lookup(_b[0], _b[1], _b[2], _b[3],
                                _b[4], _b[5]))
        starttime = min(b[4] for b in bulk)
        endtime = max(b[5] for b in bulk)
        st = self._file_reader(files, starttime, endtime).merge()
        # Trim
        for _b in bulk:
            st.select(
                network=_b[0],
                station=_b[1],
                location=_b[2],
                channel=_b[3]).trim(_b[4], _b[5])
        return st

    def get_stations(self, *args, **kwargs):
        raise NotImplementedError("No stations attached to this client")

    def get_stations_bulk(self, *args, **kwargs):
        raise NotImplementedError("No stations attached to this client")

    def get_events(self, *args, **kwargs):
        raise NotImplementedError("No events attached to this client")


@lru_cache(maxsize=20)
def _cache_read(filename: str) -> Stream:
    return read(filename)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
