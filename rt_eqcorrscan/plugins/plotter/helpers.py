"""
Helpers for working with catalogs for plotting
"""

import datetime as dt
import logging

from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Iterable, List

from obspy import UTCDateTime
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


@dataclass
class SparseOrigin:
    latitude: float = None
    longitude: float = None
    depth: float = None
    time: Union[dt.datetime, UTCDateTime] = None
    method_id: str = None

    def get(self, key: str, default=None):
        # Return the default if the key actually returns None
        return self.__dict__.get(key, default) or default


@dataclass
class SparseMagnitude:
    mag: float = None
    method_id: str = None

    def get(self, key: str, default=None):
        return self.__dict__.get(key, default) or default


class SparseEvent:
    _preferred_magnitude_index = None
    _preferred_origin_index = None

    def __init__(self,
                 resource_id: str,
                 origins: Iterable[SparseOrigin],
                 magnitudes: Iterable[SparseMagnitude],
                 preferred_origin_index: int = None,
                 preferred_magnitude_index: int = None):
        self.resource_id = resource_id
        self.origins = tuple(origins)
        self.magnitudes = tuple(magnitudes)
        if preferred_origin_index is not None:
            self.preferred_origin_index = preferred_origin_index

        if preferred_magnitude_index is not None:
            self.preferred_magnitude_index = preferred_magnitude_index

    def __repr__(self):
        return (f"SparseEvent(resource_id={self.resource_id}, "
                f"origins=[{len(self.origins)} origins], "
                f"preferred_origin_index={self.preferred_origin_index},"
                f"preferred_magnitude_index={self.preferred_magnitude_index})")

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

    @property
    def preferred_magnitude_index(self):
        return self._preferred_magnitude_index

    @preferred_magnitude_index.setter
    def preferred_magnitude_index(self, index):
        if not isinstance(index, int):
            Logger.error(
                f"Trying to set index with non-int ({index}), aborting")
            return
        try:
            _ = self.magnitudes[index]
        except IndexError as e:
            raise e
        self._preferred_magnitude_index = index

    def preferred_magnitude(self):
        if self.preferred_magnitude_index is not None:
            try:
                return self.magnitudes[self.preferred_magnitude_index]
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


def _sparsify_magnitude(magnitude: Magnitude) -> SparseMagnitude:
    try:
        method_id = magnitude.method_id.id
    except AttributeError:
        method_id = None
    return SparseMagnitude(mag=magnitude.mag, method_id=method_id)


def _sparsify_event(event: Event) -> SparseEvent:
    origins = [_sparsify_origin(ori) for ori in event.origins]
    magnitudes = [_sparsify_magnitude(mag) for mag in event.magnitudes]
    ev = SparseEvent(resource_id=event.resource_id.id,
                     origins=origins, magnitudes=magnitudes)
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

    pref_mag_ind = None
    if event.preferred_magnitude_id:
        for i, mag in enumerate(event.magnitudes):
            if mag.resource_id == event.preferred_magnitude_id:
                break
        else:
            i = None
        pref_mag_ind = i
    if pref_mag_ind is not None:
        ev.preferred_magnitude_index = pref_mag_ind
    return ev


def sparsify_catalog(catalog: Catalog) -> List[SparseEvent]:
    """ Make a sparse verion of the catalog """
    return [_sparsify_event(ev) for ev in catalog]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
