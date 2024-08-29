"""
Class and methods to lazily compute correlations for new events.
"""

import fnmatch
import os
import h5py
import numpy as np
import logging

from typing import Iterable, Union

from bokeh.core.property.validation import validate
from obspy import Catalog

from eqcorrscan.utils.catalog_to_dd import (
    _compute_dt_correlations, _make_event_pair, _make_sparse_event,
    SparseEvent, SparsePick, SeedPickID, _meta_filter_stream)


Logger = logging.getLogger(__name__)


class Correlations:
    """
    Object to hold correlations backed by an hdf5 file.

    Correlations are stored as an m x n x n array where n is the number of
    events and m is the number of channels.
    """
    _string_encoding = 'utf-8'
    _string_dtype = h5py.string_dtype(encoding=_string_encoding, length=None)

    def __init__(self, correlation_file: str = None):
        if correlation_file is None:
            correlation_file = f".correlations_{id(self)}.h5"
        self._correlation_file = correlation_file
        if not os.path.isfile(correlation_file):
            self._make_correlation_file()
        self._validate_correlation_file()
        self.seedids = self._get_seed_indexes()
        self.eventids = self._get_event_indexes()

    def __repr__(self):
        return f"Correlations(correlation_file={self.correlation_file})"

    def _get_correlation_file(self):
        return self._correlation_file

    correlation_file = property(fget=_get_correlation_file)

    def _get_seed_indexes(self):
        with h5py.File(self.correlation_file, "r") as f:
            sids = [_.decode(self._string_encoding)
                    for _ in list(f['seedids'][:])]
        return sids

    def _get_event_indexes(self):
        with h5py.File(self.correlation_file, "r") as f:
            sids = [_.decode(self._string_encoding)
                    for _ in list(f['eventids'][:])]
        return sids

    def _validate_correlation_file(self):
        if not os.path.isfile(self.correlation_file):
            return False
        # Check that the data structure is as expected
        with h5py.File(self.correlation_file, "r") as f:
            assert "ccs" in f.keys(), f"Missing ccs in {f.keys()}"
            assert "eventids" in f.keys(), f"Missing eventids in {f.keys()}"
            assert "seedids" in f.keys(), f"Missing seedids in {f.keys()}"
            assert f['eventids'].maxshape == (None, ), "eventids is not expandable"
            assert f['seedids'].maxshape == (None, ), "seedids is not expandable"
        return True

    def _make_correlation_file(self):
        with h5py.File(self.correlation_file, "w") as f:
            # Top-level group - correlations which hosts len(seed_ids) groups
            # of len(eventid) datasets of len(eventid)
            _ = f.create_group(name="ccs")
            _ = f.create_dataset(
                name="eventids", data=[""], dtype=self._string_dtype,
                maxshape=(None, ))
            _ = f.create_dataset(
                name="seedids", data=[""], dtype=self._string_dtype,
                maxshape=(None,))
        return

    def _get_correlation(
        self,
        seed_id: str,
        event1_id: str,
        event2_index: int
    ) -> float:
        with h5py.File(self.correlation_file, "r") as f:
            correlation = f['ccs'][seed_id][event1_id][event2_index]
        return correlation

    def _empty_correlation_array(self):
        if len(self.eventids) == 0:
            return [np.nan]
        return np.ones(len(self.eventids)) * np.nan

    def _new_seed_id(self, seed_id: str, file_handle: h5py.File = None):
        """ Add a new seed-id to the file. """
        close_file = False
        if file_handle is None:
            file_handle = h5py.File(self.correlation_file, "r+")
            close_file = True
        # Cope with starting with an empty file
        if len(self.seedids) == 1 and self.seedids[0] == '':
            self.seedids[0] = seed_id
        else:
            # Resize seed-ids and add new seed-id in place
            self.seedids.append(seed_id)
        file_handle['seedids'].resize((len(self.seedids), ))
        file_handle['seedids'][-1] = seed_id
        # Add new groups as needed
        file_handle['ccs'].create_group(name=seed_id)
        for event1_id in self.eventids:
            if event1_id == '':
                continue
            Logger.info(f"Creating dataset for {event1_id} of "
                        f"{len(self.eventids)} nans")
            file_handle['ccs'][seed_id].create_dataset(
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

        for sid in self.seedids:
            if sid == '':
                continue
            # Add new array for this eventid
            file_handle['ccs'][sid].create_dataset(
                name=event_id, data=self._empty_correlation_array(),
                dtype=float, maxshape=(None, ))
            # Add a new entry to all the other arrays
            for event1_id in self.eventids:
                file_handle['ccs'][sid][event1_id].resize(
                    (len(self.eventids), ))
                file_handle['ccs'][sid][event1_id][-1] = np.nan
        if close_file:
            file_handle.close()
        return

    def update(self, other: dict):
        """
        Update the values in the correlation file.

        Other should be a nested dictionary of correlations keyed by seed-id,
        event1-id, event2-id. For example a dictionary for other might look
        like:

        other = {
            "NZ.WEL.10.HHZ": {
                "2019p230876": {
                    "2020p2386755": 0.2,
                    "2012p2367858": -0.6,
                    }
                }
            }

        Will assume that the correlation for event1 <-> event2 is equal
        either way, so will set both correlations.
        """
        file_handle = h5py.File(self.correlation_file, "r+")
        for seed_id, seed_dict in other.items():
            if seed_id not in self.seedids:
                self._new_seed_id(seed_id, file_handle=file_handle)
            for event1_id, event1_dict in seed_dict.items():
                if event1_id not in self.eventids:
                    self._new_event_id(event1_id, file_handle=file_handle)
                event1_index = self.eventids.index(event1_id)
                for event2_id, cc in event1_dict.items():
                    if event2_id not in self.eventids:
                        self._new_event_id(event2_id, file_handle=file_handle)
                    event2_index = self.eventids.index(event2_id)
                    Logger.debug(
                        f"Updating correlation at ({seed_id}, "
                        f"{event1_id}, {event2_index}) to {cc}")
                    file_handle['ccs'][seed_id][event1_id][event2_index] = cc
                    # Set the mirrored correlation
                    file_handle['ccs'][seed_id][event2_id][event1_index] = cc
        file_handle.close()
        return

    def select(
        self,
        seed_id: str = None,
        eventid_1: str = None,
        eventid_2: str = None,
    ) -> dict:
        """ Supports glob patterns """
        seed_id = seed_id or "*"
        eventid_1 = eventid_1 or "*"
        eventid_2 = eventid_2 or "*"

        # Work out indexes
        sids = fnmatch.filter(self.seedids, seed_id)
        if len(sids) == 0:
            raise NotImplementedError(
                f"{seed_id} not found in correlations")

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
        out = dict()
        for sid in sids:
            if sid == '':
                continue
            event1_dict = dict()
            for event1_id in event1_ids:
                if event1_id == '':
                    continue
                event2_dict = dict()
                for event2_id, event2_index in event2_indexes:
                    if event2_id == '':
                        continue
                    event2_dict.update({event2_id: self._get_correlation(
                        seed_id=sid,
                        event1_id=event1_id,
                        event2_index=event2_index)})
                event1_dict.update({event1_id: event2_dict})
            out.update({sid: event1_dict})
        return out


class Correlator:
    _pairs_run = set()
    _dt_cache_file = os.path.abspath("./.dt.h5")  # TODO: Use object ID to make unique names
    _wf_cache_file = os.path.abspath(("./.dt_waveforms"))

    def __init__(self, minlink, ...):  # TODO: Set correlation params here
        self.minlink = minlink

    def add_events(self, catalog: Union[Catalog, Iterable[SparseEvent]]):
        # TODO: stuff



if __name__ == "__main__":
    import doctest

    doctest.testmod()
