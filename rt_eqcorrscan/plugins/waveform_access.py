""" Helpers for accessing on-disk waveforms without using a WaveBank. """

import datetime as dt
import logging
import os
import fnmatch

from typing import Iterable, List, Union

from collections import namedtuple

from obspy.core.event import Event
from obspy import Stream, read, UTCDateTime


Logger = logging.getLogger(__name__)

# Convenience file info
_FileInfo = namedtuple("FileInfo",
                       ["filename", "seed_id", "starttime", "endtime"])
PickData = namedtuple("PickData", ["seed_id", "time", "files"])

class InMemoryWaveBank:
    def __init__(self, wavedir: str):
        self.wavedir = os.path.abspath(wavedir)
        self.data_availability = dict()  # Keyed by seed ID, values as _FileInfo

    def __repr__(self):
        return f"InMemoryWaveBank(wavedir={self.wavedir})"

    @property
    def scanned_files(self):
        if len(self.data_availability):
            return {finfo.filename for value in self.data_availability.values()
                    for finfo in value}
        return set()

    @property
    def starttime(self) -> Union[UTCDateTime, None]:
        if len(self.data_availability):
            return min(
                [finfo.starttime for value in self.data_availability.values()
                 for finfo in value])
        return None

    @property
    def endtime(self) -> Union[UTCDateTime, None]:
        if len(self.data_availability):
            return max(
                [finfo.endtime for value in self.data_availability.values()
                 for finfo in value])
        return None

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
        endtime: UTCDateTime,
        check_new_files: bool = True,
    ) -> Stream:
        if check_new_files:
            self.get_data_availability(scan_all=False)
        st = Stream()
        files = self.get_files(
            network=network, station=station, location=location,
            channel=channel, starttime=starttime, endtime=endtime)
        for file in files:
            try:
                tr = read(file).trim(starttime=starttime, endtime=endtime)
            except Exception as e:
                Logger.error(f"Could not read from {file} due to {e}")
                continue
            Logger.debug(
                f"Read in {tr} from {file} for {network}.{station}."
                f"{location}.{channel} between {starttime} and {endtime}")
            st += tr
        st = st.merge()
        return st

    def get_waveforms_bulk(
        self, bulk: Iterable,
        check_new_files: bool = True
    ) -> Stream:
        st = Stream()
        for _bulk in bulk:
            st += self.get_waveforms(*_bulk, check_new_files=check_new_files)
        return st

    def get_event_waveforms(
        self,
        event: Event,
        pre_pick: float,
        length: float,
        phases: set = None,
        check_new_files: bool = True,
        same_starttimes: bool = True,
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
            Logger.debug(f"{tr_start} - {tr_end} found files: {files}")
            _used_picks.append(pick._replace(files=files))
        used_picks = _used_picks
        del _used_picks
        # Filter picks without useful times available
        used_picks = [p for p in used_picks if p.files]

        # If we want all the same start times then set the starttime to the
        # minimum start time
        if same_starttimes:
            min_picktime = min(p.time for p in used_picks)
            for pick in used_picks:
                pick._replace(time=min_picktime)

        # Read in waveforms
        st = Stream()
        for pick in used_picks:
            tr_start = pick.time - dt.timedelta(seconds=pre_pick)
            tr_end = ((pick.time - dt.timedelta(seconds=pre_pick))
                      + dt.timedelta(seconds=length))
            Logger.info(f"Looking for data between {tr_start} and {tr_end}")
            Logger.info(
                f"Looking for data in "
                f"{', '.join([f.filename for f in pick.files])}")
            for file in pick.files:
                if file.filename is None:
                    continue
                Logger.debug(f"Getting data between {tr_start} - {tr_end} "
                             f"from {file.filename}")
                try:
                    st += read(file.filename, starttime=UTCDateTime(tr_start),
                               endtime=UTCDateTime(tr_end))
                except Exception as e:
                    Logger.error(f"Could not read from {file.filename} due to {e}")
                    continue
        return st

    def get_data_availability(self, scan_all: bool = True):
        """ Scan a waveform dir and work out what is in it. """
        scanned_files = self.scanned_files  # Cache this
        new_files = 0
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
                            os.path.abspath(filepath),
                            tr.id,
                            tr.stats.starttime.datetime,  # these need to be datetimes to be hashable
                            tr.stats.endtime.datetime
                        ))
                    Logger.debug(seed_availability[-1])
                    self.data_availability.update({tr.id: seed_availability})
                new_files += 1
        if new_files > 0:
            Logger.info(f"Scanned {new_files} new files")
        else:
            Logger.info("No new files found")
        Logger.info(
            f"Data available between {self.starttime} and {self.endtime}")
        return


if __name__ == "__main__":
    import doctest

    doctest.testmod()
