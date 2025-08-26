"""
Helpers for working with catalogs
"""

import datetime as dt
import logging

from dataclasses import dataclass
from typing import Union, Iterable, List

from obspy import UTCDateTime
from obspy.core.event import (
    Event, Origin, Magnitude, Catalog, WaveformStreamID, Pick,
    ResourceIdentifier, QuantityError, Comment)


Logger = logging.getLogger(__name__)


class SparseID:
    id: str = None
    def __init__(self, id):
        self.id = id

    def __eq__(self, other):
        if isinstance(other, str):
            return self.id == other
        elif hasattr(other, "id"):
            return self.id == other.id
        else:
            return False

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return self.id

    def to_obspy(self):
        return ResourceIdentifier(**self.__dict__)

@dataclass
class SparseError:
    uncertainty: float = None
    lower_uncertainty: float = None
    upper_uncertainty: float = None
    confidence_level: float = None

    def get(self, key: str, default=None):
        # Return the default if the key actually returns None
        return self.__dict__.get(key, default) or default

    def to_obspy(self):
        return QuantityError(**self.__dict__)

@dataclass
class SparseOrigin:
    latitude: float = None
    longitude: float = None
    depth: float = None
    time: Union[dt.datetime, UTCDateTime] = None
    method_id: SparseID = None
    depth_errors: SparseError = None
    time_errors: SparseError = None
    latitude_errors: SparseError = None
    longitude_errors: SparseError = None

    def get(self, key: str, default=None):
        # Return the default if the key actually returns None
        return self.__dict__.get(key, default) or default

    def to_obspy(self):
        ori = Origin(
            latitude=self.latitude,
            longitude=self.longitude,
            depth=self.depth,
            time=UTCDateTime(self.time))
        for attr in ["method_id", "depth_errors", "latitude_errors", "longitude_errors"]:
            if self.__dict__[attr]:
                ori.__dict__.update({attr: self.__dict__[attr].to_obspy()})
        return ori

@dataclass
class SparseMagnitude:
    mag: float = None
    method_id: SparseID = None
    magnitude_type: str = None
    mag_errors: SparseError = None

    def get(self, key: str, default=None):
        return self.__dict__.get(key, default) or default

    def to_obspy(self):
        magnitude = Magnitude(
            mag=self.mag,
            magnitude_type=self.magnitude_type)
        if self.method_id:
            magnitude.method_id = self.method_id.to_obspy()
        if self.mag_errors:
            magnitude.mag_errors = self.mag_errors.to_obspy()
        return magnitude


@dataclass
class SparseComment:
    text: str = None

    def get(self, key: str, default=None):
        return self.__dict__.get(key, default) or default

    def to_obspy(self):
        return Comment(text=self.text)


@dataclass
class SparseWaveform_ID:
    network_code: str = None
    station_code: str = None
    location_code: str = None
    channel_code: str = None

    def get(self, key: str, default=None):
        return self.__dict__.get(key, default) or default

    def get_seed_string(self):
        return ".".join([self.network_code or "", self.station_code or "",
                         self.location_code or "", self.channel_code or ""])

    id = property(get_seed_string)

    def to_obspy(self):
        return WaveformStreamID(
            network_code=self.network_code,
            station_code=self.station_code,
            location_code=self.location_code,
            channel_code=self.channel_code)


@dataclass
class SparsePick:
    time: UTCDateTime = None
    phase_hint: str = None
    waveform_id: SparseWaveform_ID = None

    def get(self, key: str, default=None):
        return self.__dict__.get(key, default) or default

    def to_obspy(self):
        pick = Pick(time=self.time, phase_hint=self.phase_hint)
        if self.waveform_id:
            pick.waveform_id = self.waveform_id.to_obspy()
        return pick


class SparseEvent:
    _preferred_magnitude_index = None
    _preferred_origin_index = None
    _self_det_id = None

    def __init__(self,
                 resource_id: SparseID,
                 origins: Iterable[SparseOrigin] = [],
                 magnitudes: Iterable[SparseMagnitude] = [],
                 picks: Iterable[SparsePick] = [],
                 preferred_origin_index: int = None,
                 preferred_magnitude_index: int = None,
                 comments: Iterable[SparseComment] = []):
        self.resource_id = resource_id
        self.origins = list(origins)
        self.magnitudes = list(magnitudes)
        self.picks = list(picks)
        self.comments = list(comments)
        if preferred_origin_index is not None:
            self.preferred_origin_index = preferred_origin_index

        if preferred_magnitude_index is not None:
            self.preferred_magnitude_index = preferred_magnitude_index

    def __repr__(self):
        return (f"SparseEvent(resource_id={self.resource_id.id}, "
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

    def to_obspy(self):
        ev = Event(resource_id=self.resource_id.id)
        ev.origins = [ori.to_obspy() for ori in self.origins]
        ev.magnitudes = [mag.to_obspy() for mag in self.magnitudes]
        ev.picks = [pick.to_obspy() for pick in self.picks]
        ev.comments = [comm.to_obspy() for comm in self.comments]
        if self.preferred_origin_index:
            ev.preferred_origin_id = ev.origins[
                self.preferred_origin_index].resource_id
        if self.preferred_magnitude_index:
            ev.preferred_magnitude_id = ev.magnitudes[
                self.preferred_magnitude_index].resource_id
        return ev


###################### Functions for sparsificiation ############################


def _sparsify_origin(origin: Origin) -> SparseOrigin:
    try:
        method_id = SparseID(origin.method_id.id)
    except AttributeError:
        method_id = None
    return SparseOrigin(
        latitude=origin.latitude,
        longitude=origin.longitude,
        depth=origin.depth,
        time=origin.time,
        method_id=method_id,
        depth_errors=SparseError(**origin.depth_errors.__dict__),
        time_errors=SparseError(**origin.time_errors.__dict__),
        latitude_errors=SparseError(**origin.latitude_errors.__dict__),
        longitude_errors=SparseError(**origin.longitude_errors.__dict__),
    )


def _sparsify_magnitude(magnitude: Magnitude) -> SparseMagnitude:
    try:
        method_id = SparseID(magnitude.method_id.id)
    except AttributeError:
        method_id = None
    return SparseMagnitude(
        mag=magnitude.mag,
        method_id=method_id,
        magnitude_type=magnitude.magnitude_type,
        mag_errors=SparseError(**magnitude.mag_errors.__dict__))


def _sparsify_waveform_id(waveform_id: WaveformStreamID) -> SparseWaveform_ID:
    return SparseWaveform_ID(
        network_code=waveform_id.network_code,
        station_code=waveform_id.station_code,
        location_code=waveform_id.location_code,
        channel_code=waveform_id.channel_code
    )


def _sparsify_pick(pick: Pick) -> SparsePick:
    return SparsePick(time=pick.time, phase_hint=pick.phase_hint,
                      waveform_id=_sparsify_waveform_id(pick.waveform_id))


def _sparsify_event(event: Event, include_picks: bool = False) -> SparseEvent:
    origins = [_sparsify_origin(ori) for ori in event.origins]
    magnitudes = [_sparsify_magnitude(mag) for mag in event.magnitudes]
    comments = [SparseComment(text=comment.text) for comment in event.comments]
    picks = []
    if include_picks:
        picks = [_sparsify_pick(pick) for pick in event.picks]
    ev = SparseEvent(resource_id=SparseID(event.resource_id.id),
                     origins=origins, magnitudes=magnitudes,
                     picks=picks, comments=comments)
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


def sparsify_catalog(
    catalog: Union[List[Event], Catalog],
    include_picks: bool = False
) -> List[SparseEvent]:
    """ Make a sparse verion of the catalog """
    return [_sparsify_event(ev, include_picks=include_picks) for ev in catalog]


def sparse_catalog_to_obspy(sparse_catalog: List[SparseEvent]):
    cat = Catalog([ev.to_obspy() for ev in sparse_catalog])
    return cat


############################ Helper functions for events #############################


def get_origin(event: Union[Event, SparseEvent]) -> Union[Origin, SparseOrigin, None]:
    if len(event.origins) == 0:
        return None
    return event.preferred_origin() or event.origins[-1]


def get_origin_attr(event: Union[Event, SparseEvent], attr: str):
    ori = get_origin(event)
    if ori:
        return ori.get(attr, None)
    return None


def get_event_time(event: Event | SparseEvent) -> UTCDateTime | None:
    try:
        time = get_origin_attr(event, "time")
    except Exception as e:
        Logger.exception(f"Could not get origin time due to {e}")
        time = None
    if time is None:
        pick_times = [p.time for p in event.picks]
        if len(pick_times) == 0:
            Logger.error("Could not find a time")
            return None
        time = min(pick_times)
    return time


def get_magnitude(event: Union[Event, SparseEvent]) -> Union[Magnitude, SparseMagnitude, None]:
    if len(event.magnitudes) == 0:
        return None

    return event.preferred_magnitude() or event.magnitudes[-1]


def get_magnitude_attr(event: Union[Event, SparseEvent], attr: str):
    magnitude = get_magnitude(event)
    if magnitude:
        return magnitude.get(attr, None)
    return None


if __name__ == "__main__":
    import doctest

    doctest.testmod()
