""" Helpers for accessing on-disk waveforms without using a WaveBank. """

import datetime as dt
import logging
import os

from collections import namedtuple

from obspy.core.event import Event
from obspy import Stream, read, UTCDateTime


Logger = logging.getLogger(__name__)


def _get_event_waveforms(
    event: Event,
    data_availability: dict,
    pre_pick: float,
    length: float,
    phases: set = {"P", "Pn", "Pg", "Pb", "S", "Sn", "Sg", "Sb"},
) -> Stream:
    # convenience pick named tuple
    SparsePick = namedtuple("SparsePick", ["seed_id", "time", "files"])
    # Filter on phase hint
    used_picks = [
        SparsePick(pick.waveform_id.get_seed_string(), pick.time.datetime, None)
        for pick in event.picks if pick.phase_hint in phases]
    # Filter on availability
    used_picks = [p for p in used_picks if p.seed_id in data_availability.keys()]

    # Get relevant files
    for i, pick in enumerate(used_picks):
        seed_availability = data_availability[pick.seed_id]
        tr_start = pick.time - dt.timedelta(seconds=pre_pick),
        tr_end = ((pick.time - dt.timedelta(seconds=pre_pick)) +
                  dt.timedelta(seconds=length))
        files = {f for f in seed_availability
                 if f.starttime < tr_start < f.endtime}  # Starts within file
        files.update({f for f in seed_availability
                      if f.starttime < tr_end < f.endtime})  # Ends within file
        used_picks[i] = pick._replace(files=files)

    # Filter picks without useful times available
    used_picks = [p for p in used_picks if p.files]

    # Read in waveforms
    st = Stream()
    for pick in used_picks:
        tr_start, tr_end = (
            pick.time - dt.timedelta(seconds=pre_pick),
            (pick.time - dt.timedelta(seconds=pre_pick)) + dt.timedelta(seconds=length))
        for file in pick.files:
            st += read(file.filename, starttime=UTCDateTime(tr_start), endtime=UTCDateTime(tr_end))
    return st


def _get_data_availability(wavedir: str) -> dict:
    """ Scan a waveform dir and work out what is in it. """
    # Convenience file info
    FileInfo = namedtuple("FileInfo", ["filename", "seed_id", "starttime", "endtime"])
    data_availability = dict()
    for root, dirs, files in os.walk(wavedir):
        for f in files:
            filepath = os.path.join(root, f)
            st = None
            try:  # Try to just read the header
                st = read(filepath, headonly=True)
            except Exception as e:
                Logger.debug(f"Could not read headonly for {f} due to {e}")
                try:
                    st = read(filepath)
                except Exception as e2:
                    Logger.debug(f"Could not read {f} due to {e2}")
            if st is None:
                continue
            for tr in st:
                seed_availability = data_availability.get(tr.id, [])
                seed_availability.append(
                    FileInfo(
                        filepath,
                        tr.id,
                        tr.stats.starttime.datetime,  # these need to be datetimes to be hashable
                        tr.stats.endtime.datetime
                    ))
                data_availability.update({tr.id: seed_availability})

    return data_availability


if __name__ == "__main__":
    import doctest

    doctest.testmod()
