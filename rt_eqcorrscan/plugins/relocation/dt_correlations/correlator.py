"""
Class and methods to lazily compute correlations for new events.
"""

import fnmatch
import os
import warnings

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
        for event_pair in other:
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
        maxsep: float,
        shift_len: float,
        pre_pick: float,
        length: float,
        lowcut: float,
        highcut: float,
        interpolate: bool,
        client: Union[Client, WaveBank, InMemoryWaveBank],
        correlation_cache: str = None
    ):
        self.minlink = minlink
        self.maxsep = maxsep
        self.shift_len = shift_len
        self.pre_pick = pre_pick
        self.length = length
        self.lowcut = lowcut
        self.highcut = highcut
        self.interpolate = interpolate
        self.client = client
        self.correlation_cache = Correlations(
            correlation_file=correlation_cache)
        self._catalog = list()  # List of Sparse Events
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
            return {rid: read(waveform_filename)}
        # Get from the client and process - get an excess of data
        bulk = [(p.waveform_id.network_code,
                 p.waveform_id.station_code,
                 p.waveform_id.location_code,
                 p.waveform_id.channel_code,
                 p.time - self.pre_pick * 4,
                 p.time + 4 * (self.length - self.pre_pick))
                for p in event.picks
                if p.phase_hint.upper().startswith(("P", "S"))]
        Logger.info(f"Trying to get data from {self.client} using bulk: {bulk}")
        try:
            st = self.client.get_waveforms_bulk(bulk)
        except Exception as e:
            Logger.error(e)
            Logger.info("Trying chunked")
            st = Stream()
            for _b in bulk:
                try:
                    st += self.client.get_waveforms(*_b)
                except Exception as e:
                    Logger.error(e)
                    Logger.info(f"Skipping {_b}")
                    continue
        st = st.merge()
        Logger.info(f"Read in {len(st)} traces")
        if len(st) == 0:
            return {rid: Stream()}
        st_dict = _filter_stream(
            rid, st.split(), self.lowcut, self.highcut)
        Logger.info(f"Writing waveform to {waveform_filename}")
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

    def _append_event(self, event: Union[Event, SparseEvent]):
        if isinstance(event, Event):
            self._catalog.append(SparseEvent.from_event(event))
        else:
            self._catalog.append(event)
        return

    def add_event(
        self,
        event: Union[Event, SparseEvent],
        max_workers: int = 1,
    ):
        if event.resource_id.id in self.event_mapper.keys():
            Logger.info(f"Event {event.resource_id.id} already included, skipping")
            self._append_event(event)
            return
        # TODO: Increment the event id mapper and add event to mapper
        self.event_mapper.update({event.resource_id.id: self._nexteid})
        Logger.info("Getting waveforms")
        st_dict = self._get_waveforms(event=event)
        if len(st_dict[event.resource_id.id]) == 0:
            Logger.info(f"No waveforms for event {event.resource_id.id}: skipping")
            return
        Logger.info("Computing distance array")
        distance_array = dist_array_km(master=event, catalog=self._catalog)
        events_to_correlate = [ev for i, ev in enumerate(self._catalog)
                               if distance_array[i] <= self.maxsep]
        if len(events_to_correlate) == 0:
            # We don't need to do anymore work
            self._append_event(event)
        Logger.info(
            f"There are {len(events_to_correlate)} events to correlate")
        # Get waveforms for all events in events to correlate
        Logger.info("Getting waveforms for other events")
        for event in events_to_correlate:
            event_st_dict = self._get_waveforms(event=event)
            if len(event_st_dict[event.resource_id.id]):
                st_dict.update(event_st_dict)
            else:
                Logger.warning(
                    f"Could not get waveforms for {event.resource_id.id}")
        Logger.info(f"Running correlations for {len(st_dict.keys())} events")
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
        Logger.info("Updating the cache")
        self.correlation_cache.update(differential_times)
        self._append_event(event)
        return

    def add_events(
        self,
        catalog: Union[Catalog, Iterable[SparseEvent]],
        max_workers: int = 1,
    ):
        # TODO: This could be parallel, but probably wouldn't help that much
        for event in catalog:
            self.add_event(event, max_workers=max_workers)

    def write_correlations(
        self,
        outfile: str = "dt.cc",
        min_cc: float = 0.0,
        weight_by_square: bool = True
    ) -> int:
        """ Write the correlations to a dt.cc file """
        # Conserve memory and just get one event at a time
        written_links = 0
        with open(outfile, "w") as f:
            for event1_id in self.correlation_cache.eventids:
                if event1_id == '':
                    # Skip
                    continue
                event1_id = int(event1_id)  # Correlation cache stores as strings, but we know they are ints
                event_pairs = self.correlation_cache.select(
                    eventid_1=event1_id)
                for event_pair in event_pairs:
                    if event_pair.event_id_1 == event_pair.event_id_2:
                        continue
                    # Threshold observations
                    retained_observations = []
                    for observation in event_pair.obs:
                        if observation.weight < min_cc:
                            continue
                        if weight_by_square:
                            observation.weight **= 2
                        retained_observations.append(observation)
                    event_pair.obs = retained_observations
                    if len(event_pair.obs) == 0:
                        # Don't write empty pairs
                        continue
                    f.write(event_pair.cc_string)
                    f.write("\n")
                    written_links += 1
        return written_links


if __name__ == "__main__":
    import doctest

    doctest.testmod()
