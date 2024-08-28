"""
Class and methods to lazily compute correlations for new events.
"""

import os
import h5py
import numpy as np

from typing import Iterable, Union

from bokeh.core.property.validation import validate
from obspy import Catalog

from eqcorrscan.utils.catalog_to_dd import (
    _compute_dt_correlations, _make_event_pair, _make_sparse_event,
    SparseEvent, SparsePick, SeedPickID, _meta_filter_stream)


class Correlations:
    """
    Object to hold correlations backed by an hdf5 file.

    Correlations are stored as an m x n x n array where n is the number of
    events and m is the number of channels.
    """


    def __init__(self, correlation_file: str = None):
        if correlation_file is None:
            correlation_file = f".correlations_{id(self)}.h5"
        self._correlation_file = correlation_file
        if not os.path.isfile(correlation_file):
            self._make
        self._validate_correlation_file()

    def _validate_correlation_file(self):
        if not os.path.isfile(self.correlation_file):
            return False
        # Check that the data structure is as expected
        with h5py.File(self.correlation_file, "r") as f:
            assert "ccs" in f.keys(), f"Missing ccs in {f.keys()}"
            assert "eventids" in f.keys(), f"Missing eventids in {f.keys()}"
            assert "seedids" in f.keys(), f"Missing seedids in {f.keys()}"
            assert f["ccs"].maxshape == (None, None, None), "ccs is not expandable"
            assert f['eventids'].maxshape == (None, ), "eventids is not expandable"
            assert f['seedids'].maxshape == (None, ), "seedids is not expandable"
        return True

    def _make_correlation_file(self):
            with h5py.File(self.correlation_file, "w") as f:
                # Needs to be resizeable, and so can't be empty
                ccs_dataset = f.create_dataset(
                    "ccs", data=[[[0.0]]], dtype=float,
                    maxshape=(None, None, None))
                eventid_dataset = f.create_dataset(
                    name="eventids", data=[""], dtype=str,
                    maxshape=(None, ))
                seedids_dataset = f.create_dataset(
                    name="seedids", data=[""], dtype=str,
                    maxshape=(None,))

    def _get_correlation_file(self):
        return self._correlation_file

    correlation_file = property(fget=_get_correlation_file)

    def _get_correlations(
        self,
        seed_index: int = None,
        event1_index: int = None,
        event2_index: int = None
    ) -> Union[np.ndarray[float], float]:
        with open(self.correlation_file, "r") as f:
            # TODO: Lookup indexes in eventids and seedids

    def append(self, other):
        # TODO: Cope with things...

    def extend(self, other):
        # TODO: Cope with things...

    def lookup(
        self,
        eventid_1: str = None,
        eventid_2: str = None,
        seed_id: str = None,
    ) -> Union[np.ndarray[float], float]:
        if eventid_1 is None and eventid_2 is None and seed_id is None:
            return self._get_correlations()
        # TODO: Return to correct correlation or correlations.


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
