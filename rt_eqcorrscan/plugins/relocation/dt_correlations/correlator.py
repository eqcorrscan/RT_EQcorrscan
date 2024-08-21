"""
Class and methods to lazily compute correlations for new events.
"""

import os

from typing import Iterable

from obspy import Catalog

from eqcorrscan.utils.catalog_to_dd import (
    _compute_dt_correlations, _make_event_pair, _make_sparse_event,
    SparseEvent, SparsePick, SeedPickID, _meta_filter_stream)


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
