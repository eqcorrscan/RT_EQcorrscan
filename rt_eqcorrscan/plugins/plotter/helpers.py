"""
Helpers for working with catalogs for plotting
"""

import logging

from collections import namedtuple
from typing import Union, Iterable, List

from obspy.core.event import Event, Origin, Magnitude, Catalog


Logger = logging.getLogger(__name__)


def get_origin(event: Event) -> Union[Origin, None]:
    if len(event.origins) == 0:
        return None
    return event.preferred_origin() or event.origins[-1]


def get_origin_attr(event: Event, attr: str):
    ori = get_origin(event)
    if ori:
        return ori.get(attr, None)
    return None


def get_magnitude(event: Event) -> Union[Magnitude, None]:
    if len(event.magnitudes) == 0:
        return None

    return event.preferred_magnitude() or event.magnitudes[-1]


def get_magnitude_attr(event: Event, attr: str):
    magnitude = get_magnitude(event)
    if magnitude:
        return magnitude.get(attr, None)
    return None


SparseOrigin = namedtuple(
    "SparseOrigin",
    ["latitude", "longitude", "depth", "time", "method_id"])


class SparseEvent:
    def __init__(self,
                 origins: Iterable[SparseOrigin],
                 preferred_origin_index: int = None):
        self.origins = tuple(origins)
        if preferred_origin_index is not None:
            self.preferred_origin_index = preferred_origin_index
        else:
            self._preferred_origin_index = None

    def __repr__(self):
        return (f"SparseEvent(origins=[{len(self.origins)} origins], "
                f"preferred_origin_id={self.preferred_origin_index})")

    @property
    def preferred_origin_index(self):
        return self._preferred_origin_index

    @preferred_origin_index.setter
    def preferred_origin_index(self, index):
        if not isinstance(index, int):
            Logger.error(
                f"Trying to set index with non-int ({index}), aborting")
            return
        try:
            _ = self.origins[index]
        except IndexError as e:
            raise e
        self._preferred_origin_index = index

    def preferred_origin(self):
        if self.preferred_origin_index is not None:
            try:
                return self.origins[self.preferred_origin_index]
            except IndexError:
                return None
        return None


def _sparsify_origin(origin: Origin) -> SparseOrigin:
    try:
        method_id = origin.method_id.id
    except AttributeError:
        method_id = None
    return SparseOrigin(
        latitude=origin.latitude,
        longitude=origin.longitude,
        depth=origin.depth,
        time=origin.time,
        method_id=method_id)


def _sparsify_event(event: Event) -> SparseEvent:
    origins = [_sparsify_origin(ori) for ori in event.origins]
    ev = SparseEvent(origins=origins)
    pref_ind = None
    if event.preferred_origin_id:
        for i, ori in enumerate(event.origins):
            if ori.resource_id == event.preferred_origin_id:
                break
        else:
            i = None
        pref_ind = i
    if pref_ind is not None:
        ev.preferred_origin_index = pref_ind
    return ev


def sparsify_catalog(catalog: Catalog) -> List[SparseEvent]:
    """ Make a sparse verion of the catalog """
    return [_sparsify_event(ev) for ev in catalog]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
