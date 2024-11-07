"""
Class and methods to lazily compute correlations for new events.
"""

import fnmatch
import glob
import os
import warnings
import tqdm
import csv

import h5py
import numpy as np
import logging

from typing import Iterable, Union, Dict, List

from obspy import Catalog, Stream, read
from obspy.core.event import Event
from obspy.clients.fdsn import Client

from obsplus import WaveBank

from eqcorrscan.utils.catalog_to_dd import (
    _compute_dt_correlations, SparseEvent, _DTObs, _EventPair)
from eqcorrscan.utils.catalog_to_dd import _filter_stream
from eqcorrscan.utils.clustering import dist_array_km

from rt_eqcorrscan.plugins.waveform_access import InMemoryWaveBank

Logger = logging.getLogger(__name__)


class Correlations:
    """
    Holder for correlations backed by an on-disk file system

    Correlations are stored in directories of:

    eid1
        eid2
            station1.csv - file of phase,tt1,tt2,weight
    """
    def __init__(self, correlation_directory: str = None):
        if correlation_directory is None:
            correlation_directory = ".correlations"
        self._correlation_directory = os.path.abspath(correlation_directory)
        if not os.path.isdir(self._correlation_directory):
            os.makedirs(self._correlation_directory)
        # Work out what we have!
        self.eventids = self._get_event_indexes()
        # Getting stations has to happen after getting eventids
        self.stations = self._get_stations()

    def __repr__(self):
        return f"Correlations(correlation_directory={self.correlation_directory})"

    def _get_correlation_directory(self):
        return self._correlation_directory

    correlation_directory = property(fget=_get_correlation_directory)

    __csv_format = {
        "delimiter": ",", "lineterminator": "\r\n", "skipinitialspace": True}

    def _eid_dir(self, eid1, eid2):
        return os.path.join(self.correlation_directory, str(eid1), str(eid2))

    def _correlation_file(self, eid1, eid2, station):
        return os.path.join(self._eid_dir(eid1, eid2), f"{station}.csv")

    def _get_event_indexes(self):
        """ Directories are named by event indexes. """
        Logger.info("Scanning for event ids")
        event_indexes = {
            os.path.basename(thing[0])
            for thing in os.walk(self.correlation_directory)}
        return event_indexes

    def _get_stations(self):
        """ Files are named by station beneath event indexes. """
        stations = set()
        Logger.info("Scanning for stations")
        station_files = glob.glob(f"{self.correlation_directory}/*/*/*.csv")
        stations = {os.path.splitext(os.path.basename(_))[0]
                    for _ in station_files}
        return stations

    def _get_correlations(
        self,
        eid1: str,
        eid2: str,
        station: str,
        phase: str,
        min_weight: float = None
    ) -> Union[_EventPair, None]:
        """ Read correlations from a file. """
        if eid1 not in self.eventids or eid2 not in self.eventids or station not in self.stations:
            Logger.info(f"Unknown request")
            return None
        corr_file = self._correlation_file(
            eid1=eid1, eid2=eid2, station=station)
        if not os.path.isfile(corr_file):
            Logger.debug(f"No correlations on disk for {eid1}:{eid2}:{station}")
            return None
        event_pair = _EventPair(event_id_1=eid1, event_id_2=eid2, obs=[])
        with open(corr_file, newline='') as csvfile:
            corr_reader = csv.reader(csvfile, **self.__csv_format)
            for row in corr_reader:
                _phase = row[0]
                tt1, tt2, weight = map(float, row[1:])
                if weight >= min_weight and fnmatch.fnmatch(_phase, phase):
                    event_pair.obs.append(_DTObs(
                        station=station, tt1=tt1, tt2=tt2,
                        weight=weight, phase=_phase))
        return event_pair

    def _write_correlations(
        self,
        event_pair: _EventPair,
        update: bool = False
    ):
        # Split into stations to write individual station files
        stations = {obs.station for obs in event_pair.obs}
        for station in stations:
            # Check if eid1/eid2/station.csv or eid2/eid1/station.csv exists
            corr_file = self._correlation_file(
                eid1=event_pair.event_id_1, eid2=event_pair.event_id_2,
                station=station)
            anti_corr_file = self._correlation_file(
                eid2=event_pair.event_id_1, eid1=event_pair.event_id_2,
                station=station)
            if os.path.isfile(corr_file) or os.path.isfile(anti_corr_file):
                if not update:
                    continue
            obs = [o for o in event_pair.obs if o.station == station]
            self._write_obs(filename=corr_file, obs=obs)
        # Update the stations and event ids in use
        self.eventids.update({event_pair.event_id_1, event_pair.event_id_2})
        self.stations.update(stations)
        return

    def _write_obs(self, filename: str, obs: List[_DTObs]):
        obs_dir = os.path.dirname(filename)
        if not os.path.isdir(obs_dir):
            os.makedirs(obs_dir)
        with open(filename, 'w', newline='') as csvfile:
            corr_writer = csv.writer(csvfile, **self.__csv_format)
            for _obs in obs:
                corr_writer.writerow([
                    str(_obs.phase), str(_obs.tt1), str(_obs.tt2),
                    str(_obs.weight)])
        return

    def update(self, other: List[_EventPair], update: bool = False):
        for event_pair in tqdm.tqdm(other):
            self._write_correlations(event_pair=event_pair, update=update)
        return

    def select(
        self,
        station: str = None,
        phase: str = None,
        eventid_1: Union[str, int] = None,
        eventid_2: Union[str, int] = None,
        min_weight: float = 0.0,
    ) -> List[_EventPair]:
        """ Supports globbing """
        station = station or "*"
        phase = phase or "*"
        eid_ints = True  # Should be ints by default because that is what we expect
        if isinstance(eventid_1, int):
            eventid_1 = str(eventid_1)
        elif isinstance(eventid_1, str):
            eid_ints = False

        if isinstance(eventid_2, int):
            eventid_2 = str(eventid_2)
        elif isinstance(eventid_2, str):
            eid_ints = False

        eventid_1 = eventid_1 or "*"
        eventid_2 = eventid_2 or "*"

        if not eid_ints:
            Logger.warning("Returning event ids as strings - this may cause "
                           "issues for formatting output strings")

        # Work out indexes
        stations = fnmatch.filter(self.stations, station)
        if len(stations) == 0:
            raise NotImplementedError(
                f"{station} not found in correlations")

        event1_ids = fnmatch.filter(self.eventids, eventid_1)
        if len(event1_ids) == 0:
            raise NotImplementedError(
                f"{eventid_1} not found in correlations")

        event2_ids = fnmatch.filter(self.eventids, eventid_2)
        if len(event2_ids) == 0:
            raise NotImplementedError(
                f"{eventid_2} not found in correlations")
        out = []
        for eid1 in event1_ids:
            if not os.path.isdir(os.path.join(self.correlation_directory, str(eid1))):
                continue
            for eid2 in event2_ids:
                for station in stations:
                    _out = self._get_correlations(
                        eid1=eid1, eid2=eid2, station=station, phase=phase,
                        min_weight=min_weight)
                    if _out:
                        out.append(_out)
        # Group event_pairs
        output_event_pairs = {} # dict keyed by eid1, eid2
        for _out in out:
            if _out.event_id_1 in output_event_pairs.keys():
                if _out.event_id_2 in output_event_pairs[_out.event_id_1].keys():
                    output_event_pairs[_out.event_id_1][_out.event_id_2].extend(_out.obs)
                else:
                    output_event_pairs[_out.event_id_1].update(
                        {_out.event_id_2: _out.obs})
            else:
                output_event_pairs.update(
                    {_out.event_id_1: {_out.event_id_2: _out.obs}})
        out = [_EventPair(event_id_1=eid1, event_id_2=eid2, obs=obs)
               for eid1, value in output_event_pairs.items()
               for eid2, obs in value.items()]
        return out


class H5Correlations:
    """
    Object to hold correlations backed by an hdf5 file.

    Correlations are stored as an m x n x n array where n is the number of
    events and m is the number of channels.
    """
    def __init__(self, correlation_file: str = None):
        self._string_encoding = 'utf-8'
        if correlation_file is None:
            correlation_file = f".correlations_{id(self)}.h5"
        self._correlation_file = os.path.abspath(correlation_file)
        if not os.path.isfile(correlation_file):
            self._make_correlation_file()
        self._validate_correlation_file()
        self.stations = self._get_station_indexes()
        self.eventids = self._get_event_indexes()
        self.phases = self._get_phase_indexes()

    def __repr__(self):
        return f"Correlations(correlation_file={self.correlation_file})"

    def _get_string_dtype(self):
        return h5py.string_dtype(encoding=self._string_encoding, length=None)

    _string_dtype = property(fget=_get_string_dtype)

    def _get_correlation_file(self):
        return self._correlation_file

    correlation_file = property(fget=_get_correlation_file)

    def _get_station_indexes(self):
        with h5py.File(self.correlation_file, "r") as f:
            sids = [_.decode(self._string_encoding)
                    for _ in list(f['stations'][:])]
        return sids

    def _get_event_indexes(self):
        with h5py.File(self.correlation_file, "r") as f:
            eids = [_.decode(self._string_encoding)
                    for _ in list(f['eventids'][:])]
        return eids

    def _get_phase_indexes(self):
        with h5py.File(self.correlation_file, "r") as f:
            pids = [_.decode(self._string_encoding)
                    for _ in list(f['phases'][:])]
        return pids

    def _validate_correlation_file(self):
        if not os.path.isfile(self.correlation_file):
            return False
        # Check that the data structure is as expected
        with h5py.File(self.correlation_file, "r") as f:
            assert "tt1" in f.keys(), f"Missing tt1 in {f.keys()}"
            assert "tt2" in f.keys(), f"Missing tt2 in {f.keys()}"
            assert "weight" in f.keys(), f"Missing weight in {f.keys()}"
            assert "eventids" in f.keys(), f"Missing eventids in {f.keys()}"
            assert "stations" in f.keys(), f"Missing stations in {f.keys()}"
            assert "phases" in f.keys(), f"Missing stations in {f.keys()}"
            assert f['eventids'].maxshape == (None, ), "eventids is not expandable"
            assert f['stations'].maxshape == (None, ), "stations is not expandable"
            assert f['phases'].maxshape == (None,), "phases is not expandable"
        return True

    def _make_correlation_file(self):
        with h5py.File(self.correlation_file, "w") as f:
            # Top-level group - correlations which hosts len(seed_ids) groups
            # of len(eventid) datasets of len(eventid)
            _ = f.create_group(name="tt1")  # tt1 if travel-time for event1
            _ = f.create_group(name="tt2")  # tt2 is travel time for event 2
            _ = f.create_group(name="weight")  # Weight is correlation
            _ = f.create_dataset(
                name="eventids", data=[""], dtype=self._string_dtype,
                maxshape=(None, ))
            _ = f.create_dataset(
                name="stations", data=[""], dtype=self._string_dtype,
                maxshape=(None,))
            _ = f.create_dataset(
                name="phases", data=[""], dtype=self._string_dtype,
                maxshape=(None,))
        return

    def _get_correlation(
        self,
        station: str,
        phase: str,
        event1_id: str,
        event2_index: int
    ) -> _DTObs:
        with h5py.File(self.correlation_file, "r") as f:
            tt1 = f['tt1'][station][phase][event1_id][event2_index]
            tt2 = f['tt2'][station][phase][event1_id][event2_index]
            weight = f['weight'][station][phase][event1_id][event2_index]
        dt = _DTObs(station=station, tt1=tt1, tt2=tt2,
                    weight=weight, phase=phase)
        return dt

    def _empty_correlation_array(self):
        if len(self.eventids) == 0:
            return [np.nan]
        return np.ones(len(self.eventids)) * np.nan

    def _new_station(self, station: str, file_handle: h5py.File = None):
        """ Add a new seed-id to the file. """
        close_file = False
        if file_handle is None:
            file_handle = h5py.File(self.correlation_file, "r+")
            close_file = True
        # Cope with starting with an empty file
        if len(self.stations) == 1 and self.stations[0] == '':
            self.stations[0] = station
        else:
            # Resize seed-ids and add new seed-id in place
            self.stations.append(station)
        file_handle['stations'].resize((len(self.stations), ))
        file_handle['stations'][-1] = station
        # Add new groups as needed
        for group in ('tt1', 'tt2', 'weight'):
            file_handle[group].create_group(name=station)
            for phase in self.phases:
                if phase == "":
                    continue
                file_handle[group][station].create_group(name=phase)
                for event1_id in self.eventids:
                    if event1_id == '':
                        continue
                    Logger.info(f"Creating dataset for {event1_id} of "
                                f"{len(self.eventids)} nans")
                    file_handle[group][station][phase].create_dataset(
                        name=event1_id, data=self._empty_correlation_array(),
                        dtype=float, maxshape=(None, ))
        if close_file:
            file_handle.close()
        return

    def _new_event_id(self, event_id: str, file_handle: h5py.File = None):
        """ Add a new event-id to the file. """
        close_file = False
        if file_handle is None:
            file_handle = h5py.File(self.correlation_file, "r+")
            close_file = True
        if len(self.eventids) == 1 and self.eventids[0] == '':
            self.eventids[0] = event_id
        else:
            # Resize and add new event-id in place
            self.eventids.append(event_id)
        file_handle['eventids'].resize((len(self.eventids), ))
        file_handle['eventids'][-1] = event_id

        # TODO: This is slow for large datasets
        for group in ("tt1", "tt2", "weight"):
            for station in self.stations:
                if station == '':
                    continue
                for phase in self.phases:
                    if phase == '':
                        continue
                    # Add new array for this eventid
                    file_handle[group][station][phase].create_dataset(
                        name=event_id, data=self._empty_correlation_array(),
                        dtype=float, maxshape=(None, ))
                    # Add a new entry to all the other arrays
                    for event1_id in self.eventids:
                        file_handle[group][station][phase][event1_id].resize(
                            (len(self.eventids), ))
                        file_handle[group][station][phase][event1_id][-1] = np.nan
        if close_file:
            file_handle.close()
        return

    def _new_phase(self, phase: str, file_handle: h5py.File = None):
        """ Add a new event-id to the file. """
        close_file = False
        if file_handle is None:
            file_handle = h5py.File(self.correlation_file, "r+")
            close_file = True
        if len(self.phases) == 1 and self.phases[0] == '':
            self.phases[0] = phase
        else:
            # Resize and add new event-id in place
            self.phases.append(phase)
        file_handle['phases'].resize((len(self.phases), ))
        file_handle['phases'][-1] = phase

        for group in ("tt1", "tt2", "weight"):
            for station in self.stations:
                if station == '':
                    continue
                file_handle[group][station].create_group(name=phase)
                for event1_id in self.eventids:
                    if event1_id == '':
                        continue
                    Logger.info(f"Creating dataset for {event1_id} of "
                                f"{len(self.eventids)} nans")
                    file_handle[group][station][phase].create_dataset(
                        name=event1_id, data=self._empty_correlation_array(),
                        dtype=float, maxshape=(None, ))
        if close_file:
            file_handle.close()
        return

    def update(self, other: List[_EventPair]):
        """
        Update the values in the correlation file.

        Other should be a list of _EventPair objects.

        other = [
            _EventPair(
                event_id_1=1, event_id_2=2,
                obs=[_DTObs(weight=0.2, tt1=12.1, tt2=13.2, station="WVZ", phase="P"),
                     _DTObs(weight=0.1, tt1=15.1, tt2=16.2, station="WVZ", phase="S"),
                     _DTObs(weight=0.8, tt1=9.1, tt2=9.2, station="FOZ", phase="P")]),
            _EventPair(
                event_id_1=1, event_id_2=3,
                obs=[_DTObs(weight=-0.2, tt1=12.1, tt2=15.2, station="WVZ", phase="P"),
                     _DTObs(weight=0.7, tt1=15.1, tt2=17.2, station="WVZ", phase="S"),
                     _DTObs(weight=0.9, tt1=9.1, tt2=9.1, station="FOZ", phase="P")])]


        Will assume that the correlation for event1 <-> event2 is equal
        either way, so will set both correlations.
        """
        file_handle = h5py.File(self.correlation_file, "r+")
        for event_pair in tqdm.tqdm(other):
            for obs in event_pair.obs:
                # Make the necessary structural changes - heirachy is station/phase/eventids
                if obs.station not in self.stations:
                    self._new_station(obs.station, file_handle=file_handle)
                if obs.phase not in self.phases:
                    self._new_phase(obs.phase, file_handle=file_handle)
                if str(event_pair.event_id_1) not in self.eventids:
                    self._new_event_id(
                        str(event_pair.event_id_1), file_handle=file_handle)
                if str(event_pair.event_id_2) not in self.eventids:
                    self._new_event_id(
                        str(event_pair.event_id_2), file_handle=file_handle)
                Logger.debug(
                    f"Updated values for {obs.station} {obs.phase} "
                    f"{event_pair.event_id_1} -- {event_pair.event_id_2}")
                # Mirror the changes
                event_pairings = [
                    (str(event_pair.event_id_1), str(event_pair.event_id_2)), # forward
                    (str(event_pair.event_id_2), str(event_pair.event_id_1))  # reverse
                ]
                tt_pairings = [(obs.tt1, obs.tt2), (obs.tt2, obs.tt2)]

                for (eid1, eid2), (tt1, tt2) in zip(event_pairings, tt_pairings):
                    eid2_index = self.eventids.index(eid2)
                    file_handle['tt1'][obs.station][obs.phase][eid1][
                        eid2_index] = tt1
                    file_handle['tt2'][obs.station][obs.phase][eid1][
                        eid2_index] = tt2
                    file_handle['weight'][obs.station][obs.phase][eid1][
                        eid2_index] = obs.weight
        file_handle.close()
        return

    def select(
        self,
        station: str = None,
        phase: str = None,
        eventid_1: Union[str, int] = None,
        eventid_2: Union[str, int] = None,
    ) -> List[_EventPair]:
        """ Supports glob patterns """
        station = station or "*"
        phase = phase or "*"
        eid_ints = True  # Should be ints by default because that is what we expect
        if isinstance(eventid_1, int):
            eventid_1 = str(eventid_1)
        elif isinstance(eventid_1, str):
            eid_ints = False

        if isinstance(eventid_2, int):
            eventid_2 = str(eventid_2)
        elif isinstance(eventid_2, str):
            eid_ints = False

        eventid_1 = eventid_1 or "*"
        eventid_2 = eventid_2 or "*"

        if not eid_ints:
            Logger.warning("Returning event ids as strings - this may cause "
                           "issues for formatting output strings")

        # Work out indexes
        sids = fnmatch.filter(self.stations, station)
        if len(sids) == 0:
            raise NotImplementedError(
                f"{station} not found in correlations")

        pids = fnmatch.filter(self.phases, phase)
        if len(pids) == 0:
            raise NotImplementedError(
                f"{phase} not found in correlations")

        event1_ids = fnmatch.filter(self.eventids, eventid_1)
        if len(event1_ids) == 0:
            raise NotImplementedError(
                f"{eventid_1} not found in correlations")

        event2_ids = fnmatch.filter(self.eventids, eventid_2)
        if len(event2_ids) == 0:
            raise NotImplementedError(
                f"{eventid_2} not found in correlations")
        else:
            event2_indexes = [
                (s, self.eventids.index(s)) for s in event2_ids]
        out = []
        for event1_id in event1_ids:
            if event1_id == '':
                continue
            if eid_ints:
                event1_id_out = int(event1_id)
            else:
                event1_id_out = event1_id
            for event2_id, event2_index in event2_indexes:
                if event2_id == '':
                    continue
                if eid_ints:
                    event2_id_out = int(event2_id)
                else:
                    event2_id_out = event2_id
                event_pair = _EventPair(
                    event_id_1=event1_id_out, event_id_2=event2_id_out, obs=[])
                for station in sids:
                    if station == '':
                        continue
                    for phase in pids:
                        if phase == '':
                            continue
                        obs = self._get_correlation(
                            station=station, phase=phase, event1_id=event1_id,
                            event2_index=event2_index)
                        if not np.isnan(obs.weight):
                            event_pair.obs.append(obs)
                if len(event_pair.obs) > 0:
                    out.append(event_pair)
        return out


class Correlator:
    def __init__(
        self,
        minlink: int,
        min_cc: float,
        maxsep: float,
        shift_len: float,
        pre_pick: float,
        length: float,
        lowcut: float,
        highcut: float,
        interpolate: bool,
        client: Union[Client, WaveBank, InMemoryWaveBank],
        max_event_links: int = None,
        outfile: str = "dt.cc",
        weight_by_square: bool = False,
        # correlation_cache: str = None
    ):
        self.minlink = minlink
        self.maxsep = maxsep
        self.max_event_links = max_event_links
        self.shift_len = shift_len
        self.pre_pick = pre_pick
        self.length = length
        self.lowcut = lowcut
        self.highcut = highcut
        self.interpolate = interpolate
        self.client = client
        if os.path.isfile(outfile):
            Logger.warning(f"{outfile} exists, removing.")
            os.remove(outfile)
        self.outfile = outfile
        self.min_cc = min_cc
        self.weight_by_square = weight_by_square
        # self.correlation_cache = Correlations(
        #     correlation_directory=correlation_cache)
        self._catalog = set()  # List of Sparse Events
        self._pairs_run = set()  # Cache of what work has already been done
        self.event_mapper = dict()  # Key to map event ids to dt.cc ids
        self._wf_cache_dir = os.path.abspath(("./.dt_waveforms"))
        self._wf_naming = "{cache_dir}/{event_id}.ms"

    def _get_waveforms(
        self,
        event: Union[Event, SparseEvent],
    ) -> Dict[str, Stream]:
        """
        Get and process stream - look in database first, get from client second
        """
        rid = event.resource_id.id
        if not os.path.isdir(self._wf_cache_dir):
            os.makedirs(self._wf_cache_dir)
        waveform_filename = self._wf_naming.format(
            cache_dir=self._wf_cache_dir,
            event_id=rid.split('/')[-1])
        if os.path.isfile(waveform_filename):
            Logger.debug(f"Reading cached waveforms from {waveform_filename}")
            st = read(waveform_filename)
            Logger.debug(f"Read in {len(st)} traces")
            return {rid: st}
        # Get from the client and process - get an excess of data
        bulk = [(p.waveform_id.network_code,
                 p.waveform_id.station_code,
                 p.waveform_id.location_code,
                 p.waveform_id.channel_code,
                 p.time - self.pre_pick * 4,
                 p.time + 4 * (self.length - self.pre_pick))
                for p in event.picks
                if p.phase_hint.upper().startswith(("P", "S"))]
        Logger.debug(f"Trying to get data from {self.client} using bulk: {bulk}")
        try:
            st = self.client.get_waveforms_bulk(bulk)
        except Exception as e:
            Logger.error(e)
            Logger.debug("Trying chunked")
            st = Stream()
            for _b in bulk:
                try:
                    st += self.client.get_waveforms(*_b)
                except Exception as e:
                    Logger.error(e)
                    Logger.info(f"Skipping {_b}")
                    continue
        st = st.merge()
        Logger.debug(f"Read in {len(st)} traces")
        if len(st) == 0:
            return {rid: Stream()}
        st_dict = _filter_stream(
            rid, st.split(), self.lowcut, self.highcut)
        Logger.debug(f"Writing waveform to {waveform_filename}")
        # Catch and ignore warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                st_dict[rid].write(
                    waveform_filename, format="MSEED")
            except Exception as e:
                Logger.error(f"Could not write {st_dict[rid]} to "
                             f"{waveform_filename} due to {e}")
        return st_dict

    @property
    def _nexteid(self):
        last_eid = 0
        for eid in self.event_mapper.values():
            if eid > last_eid:
                last_eid = eid
        return last_eid + 1

    @property
    def _catalog_event_ids(self):
        return {ev.resource_id.id for ev in self._catalog}

    def _append_event(self, event: Union[Event, SparseEvent]):
        if event.resource_id.id in self._catalog_event_ids:
            Logger.info(f"Not adding {event.resource_id.id} to working catalog: "
                        f"event id is already in catalog")
        if isinstance(event, Event):
            self._catalog.add(SparseEvent.from_event(event))
        else:
            self._catalog.add(event)
        return

    def add_event(
        self,
        event: Union[Event, SparseEvent],
        max_workers: int = 1,
    ) -> int:
        if event.resource_id.id in self.event_mapper.keys():
            Logger.info(f"Event {event.resource_id.id} already included, skipping")
            self._append_event(event)
            return 0
        self.event_mapper.update({event.resource_id.id: self._nexteid})
        Logger.info("Getting waveforms")
        st_dict = self._get_waveforms(event=event)
        if len(st_dict[event.resource_id.id]) == 0:
            Logger.warning(
                f"No waveforms for event {event.resource_id.id}: skipping")
            self._append_event(event)
            return 0
        Logger.info("Computing distance array")
        ordered_catalog = list(self._catalog)
        distance_array = dist_array_km(
            master=event, catalog=ordered_catalog)
        if (distance_array <= self.maxsep).sum() == 0:
            Logger.warning(
                f"No events within {self.maxsep} km - no correlations.")
            self._append_event(event)
            return 0
        # We need to retain the distances used for max_event_link
        events_to_correlate, distance_array = zip(*[
            (ev, dist) for ev, dist in zip(ordered_catalog, distance_array)
            if dist <= self.maxsep])
        # Convert from tuple to Catalog
        events_to_correlate = Catalog(events=list(events_to_correlate))
        Logger.info(
            f"There are {len(events_to_correlate)} events to correlate")
        if len(events_to_correlate) == 0:
            # We don't need to do anymore work
            self._append_event(event)
            return 0
        if self.max_event_links and len(events_to_correlate) > self.max_event_links:
            # We just want the n closest events
            order = np.argsort(distance_array)
            # Order them, and keep the closest n events
            events_to_correlate.events = [
                events_to_correlate[i] for i in order][0:self.max_event_links]
            Logger.info(
                f"Correlating the {len(events_to_correlate)} closest events")
            Logger.info(
                f"Maximum inter-event distance: "
                f"{distance_array[order[self.max_event_links]]}")
        # Get waveforms for all events in events to correlate
        Logger.info("Getting waveforms for other events")
        for ev in tqdm.tqdm(events_to_correlate):
            event_st_dict = self._get_waveforms(event=ev)
            if len(event_st_dict[ev.resource_id.id]):
                st_dict.update(event_st_dict)
            else:
                Logger.warning(
                    f"Could not get waveforms for {ev.resource_id.id}")
        Logger.info(f"Running correlations for {len(events_to_correlate)} "
                    f"events")
        # Run _compute_dt_correlations
        differential_times = _compute_dt_correlations(
            catalog=events_to_correlate, master=event,
            min_link=0, event_id_mapper=self.event_mapper,
            stream_dict=st_dict, min_cc=0.0, extract_len=self.length,
            pre_pick=self.pre_pick, shift_len=self.shift_len,
            interpolate=self.interpolate, max_workers=max_workers,
            shm_data_shape=None, shm_dtype=None,
            weight_by_square=False)
        Logger.info("Got the following differential times:")
        for dt in differential_times:
            Logger.info(dt)
        # Differential times is a list of _EventPairs
        # Logger.info("Updating the cache")
        # self.correlation_cache.update(differential_times)
        Logger.info("Writing correlations")
        written_links = self.write_correlations(differential_times)
        Logger.info(f"Wrote {written_links} event pairs")
        self._append_event(event)
        return written_links

    def add_events(
        self,
        catalog: Union[Catalog, Iterable[SparseEvent]],
        max_workers: int = 1,
    ) -> int:
        n, written_links = len(catalog), 0
        for i, event in enumerate(catalog):
            Logger.info(f"Adding event {i} for {n}")
            written_links += self.add_event(event, max_workers=max_workers)
        return written_links

    def write_correlations(
        self,
        differential_times: List[_EventPair]
    ) -> int:
        """ Write the correlations to a dt.cc file """
        written_links = 0
        with open(self.outfile, "a") as f:
            for event_pair in tqdm.tqdm(differential_times):
                obs = [o for o in event_pair.obs if o.weight >= self.min_cc]
                if len(obs) == 0:
                    continue
                if self.weight_by_square:
                    sq_obs = []
                    for o in obs:
                        sq_obs.append(_DTObs(
                            station=o.station, tt1=o.tt1, tt2=o.tt2,
                            weight=o.weight ** 2, phase=o.phase))
                    obs = sq_obs
                event_pair.obs = obs
                f.write(event_pair.cc_string)
                f.write("\n")
                written_links += 1
        return written_links

        # Write links for correlation cache.
        # Conserve memory and just get one event at a time
        # written_links = 0
        # with open(outfile, "w") as f:
        #     for event1_id in self.correlation_cache.eventids:
        #         if event1_id == '':
        #             # Skip
        #             continue
        #         try:
        #             event1_id = int(event1_id)
        #         except ValueError:
        #             continue
        #         event_pairs = self.correlation_cache.select(
        #             eventid_1=event1_id, min_weight=min_cc)
        #         for event_pair in event_pairs:
        #             if event_pair.event_id_1 == event_pair.event_id_2:
        #                 continue
        #             event_pair.event_id_1 = int(event_pair.event_id_1)
        #             event_pair.event_id_2 = int(event_pair.event_id_2)
        #             if weight_by_square:
        #                 for obs in event_pair.obs:
        #                     obs.weight **= 2
        #             if len(event_pair.obs) == 0:
        #                 # Don't write empty pairs
        #                 continue
        #             f.write(event_pair.cc_string)
        #             f.write("\n")
        #             written_links += 1
        # return written_links


if __name__ == "__main__":
    import doctest

    doctest.testmod()
