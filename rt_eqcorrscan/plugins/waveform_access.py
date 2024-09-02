""" Helpers for accessing on-disk waveforms without using a WaveBank. """

import datetime as dt
import logging
import os
import fnmatch

from typing import Iterable, List

from collections import namedtuple

from obspy.core.event import Event
from obspy import Stream, read, UTCDateTime


Logger = logging.getLogger(__name__)

# Convenience file info
_FileInfo = namedtuple("FileInfo",
                       ["filename", "seed_id", "starttime", "endtime"])
PickData = namedtuple("PickData", ["seed_id", "time", "files"])

class InMemoryWaveBank:
    data_availability = dict()  # Keyed by seed ID, values as _FileInfo

    def __init__(self, wavedir: str):
        self.wavedir = os.path.abspath(wavedir)

    def __repr__(self):
        return f"InMemoryWaveBank(wavedir={self.wavedir})"

    @property
    def scanned_files(self):
        if len(self.data_availability):
            return {finfo.filename for value in self.data_availability.values()
                    for finfo in value}
        return set()

    def get_files(
        self,
        network: str = None,
        station: str = None,
        location: str = None,
        channel: str = None,
        starttime: UTCDateTime = None,
        endtime: UTCDateTime = None,
    ) -> List[str]:
        sid = ".".join((
            network or "*",
            station or "*",
            location or "*",
            channel or "*"))
        keys = fnmatch.filter(self.data_availability.keys(), sid)
        # Filter on time
        files = []
        for key in keys:
            for finfo in self.data_availability[key]:
                if finfo.starttime <= starttime <= finfo.endtime:
                    # Starttime is within file, we want file
                    files.append(finfo.filename)
                elif finfo.starttime <= endtime <= finfo.endtime:
                    # Endtime is within file, we want file
                    files.append(finfo.filename)
                elif starttime <= finfo.starttime and endtime >= finfo.endtime:
                    # File starts and ends within requested start and end
                    files.append(finfo.filename)
        return files

    def get_waveforms(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime
    ) -> Stream:
        st = Stream()
        files = self.get_files(
            network=network, station=station, location=location,
            channel=channel, starttime=starttime, endtime=endtime)
        for file in files:
            st += read(file).trim(starttime=starttime, endtime=endtime)
        st = st.merge()
        return st

    def get_waveforms_bulk(self, bulk: Iterable) -> Stream:
        st = Stream()
        for _bulk in bulk:
            st += self.get_waveforms(*_bulk)
        return st

    def get_event_waveforms(
        self,
        event: Event,
        pre_pick: float,
        length: float,
        phases: set = None,
        check_new_files: bool = True,
    ) -> Stream:
        # convenience pick named tuple
        if phases is None:
            phases = {"P", "Pn", "Pg", "Pb", "S", "Sn", "Sg", "Sb"}
        # Check for new files
        if check_new_files:
            self.get_data_availability(scan_all=False)
        # Filter on phase hint
        used_picks = [
            PickData(pick.waveform_id.get_seed_string(),
                     pick.time.datetime, [None])
            for pick in event.picks if pick.phase_hint in phases]
        # Filter on availability
        used_picks = [p for p in used_picks
                      if p.seed_id in self.data_availability.keys()]
        # Get relevant files
        _used_picks = []
        for i, pick in enumerate(used_picks):
            seed_availability = self.data_availability[pick.seed_id]
            tr_start = pick.time - dt.timedelta(seconds=pre_pick)
            tr_end = ((pick.time - dt.timedelta(seconds=pre_pick)) +
                      dt.timedelta(seconds=length))
            files = {f for f in seed_availability
                     if f.starttime < tr_start < f.endtime}
            # Starts within file
            files.update({f for f in seed_availability
                          if f.starttime < tr_end < f.endtime})
            # Ends within file
            files.update({f for f in seed_availability
                          if tr_start <= f.starttime and tr_end <= f.endtime})
            # File completely within timespan
            _used_picks.append(pick._replace(files=files))
        used_picks = _used_picks
        del _used_picks
        # Filter picks without useful times available
        used_picks = [p for p in used_picks if p.files]

        # Read in waveforms
        st = Stream()
        for pick in used_picks:
            tr_start = pick.time - dt.timedelta(seconds=pre_pick)
            tr_end = ((pick.time - dt.timedelta(seconds=pre_pick))
                      + dt.timedelta(seconds=length))
            for file in pick.files:
                if file.filename is None:
                    continue
                st += read(file.filename, starttime=UTCDateTime(tr_start),
                           endtime=UTCDateTime(tr_end))
        return st

    def get_data_availability(self, scan_all: bool = True):
        """ Scan a waveform dir and work out what is in it. """
        scanned_files = self.scanned_files  # Cache this
        for root, dirs, files in os.walk(self.wavedir):
            for f in files:
                if f == ".index.h5":
                    # Skip the common wavebank index file
                    continue
                filepath = os.path.join(root, f)
                if filepath in scanned_files and not scan_all:
                    Logger.debug(f"Skipping {filepath} - already scanned")
                    continue
                Logger.debug(f"Scanning {filepath}")
                st = None
                try:  # Try to just read the header
                    st = read(filepath, headonly=True)
                except Exception as e:
                    Logger.warning(f"Could not read headonly for {f} due to {e}")
                    try:
                        st = read(filepath)
                    except Exception as e2:
                        Logger.warning(f"Could not read {f} due to {e2}")
                if st is None:
                    continue
                for tr in st:
                    seed_availability = self.data_availability.get(tr.id, [])
                    seed_availability.append(
                        _FileInfo(
                            filepath,
                            tr.id,
                            tr.stats.starttime.datetime,  # these need to be datetimes to be hashable
                            tr.stats.endtime.datetime
                        ))
                    self.data_availability.update({tr.id: seed_availability})
        return


if __name__ == "__main__":
    import doctest

    doctest.testmod()
