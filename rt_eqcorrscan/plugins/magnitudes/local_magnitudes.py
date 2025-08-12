"""
Script and functions to compute local magnitudes for events.

Steps:
1. Watch directory for detections
2. If new detections, read in to memory (event and waveform)
3. Run mag-calc.amp_pick_event
4. Compute magnitude according to some defined scale
5. Output to csv and json catalogue

Designed to be run as continuously running subprocess managed by rt_eqcorrscan
"""
import json
import os
import logging

from math import log10
from typing import List, Union

from obspy.geodetics import gps2dist_azimuth
from obspy.core.event import (
    Event, Amplitude, StationMagnitude, CreationInfo, Magnitude)
from obspy import Inventory, UTCDateTime, read_events, read_inventory

from rt_eqcorrscan.config.config import _PluginConfig, PLUGIN_CONFIG_MAPPER
from rt_eqcorrscan.plugins.plugin import _Plugin

Logger = logging.getLogger(__name__)

MLNZ20_CONSTANTS = {
    "alpha": 0.51,
    "beta": -0.79 * 10 ** -3,
    "gamma": -1.67,
    "delta": 2.38 * 10 ** -3,
}

# --------------- Magnitude functions -----------------------------------------

def get_amplitude_time(amplitude: Amplitude) -> UTCDateTime:
    """
    Get the time of the amplitude observation.

    Parameters
    ----------
    amplitude

    Returns
    -------

    """
    related_pick = amplitude.pick_id.get_referred_object()
    if related_pick:
        if related_pick.phase_hint == amplitude.type:
            # This is actually the pick - EQcorrscan standard
            return related_pick.time
    # Seiscomp standard doesn't give a useful pick, or a time
    if amplitude.time_window:
        return amplitude.time_window.reference
    raise NotImplementedError(f"No time found for amplitude:\n{amplitude}")

def mlnz20(
    event: Event,
    inventory: Inventory,
    station_corrections: dict,
    *args, **kwargs
) -> Event:
    """
    Compute magnitude using the Rhoades et al. 2021 magnitude scale

    Parameters
    ----------
    event:
        The event to compute magnitudes for - will
        append to magnitude and station magnitudes
    inventory:
        The stations used for the event - must have at least station
        locations.
    station_corrections:
        Station corrections dictionary keyed by station. If station is
        not in dictionary a default correction of 0.0 will be applied.

    Returns
    -------
    Original event with station magnitudes and magnitudes appended.
    """
    origin = event.preferred_origin() or event.origins[-1]
    station_magnitudes = []
    for amplitude in event.amplitudes:
        if amplitude.type not in ["ML", "MLv"]:
            Logger.info(f"Skipping amplitude of type {amplitude.type}")
        # Find the related station
        station = inventory.select(
            network=amplitude.waveform_id.network_code or "*",
            station=amplitude.waveform_id.station_code or "*",
            location=amplitude.waveform_id.location_code or "*",
            time=get_amplitude_time(amplitude))
        # Note skipping channel - we don't usually care about the
        # channel, and seiscomp doesn't always return full channel
        # codes cos... why would you do that!?
        if len(station) == 0:
            Logger.warning(
                f"No station found for "
                f"{amplitude.waveform_id.get_seed_string()}")
            continue
        station_locations = {(s.latitude, s.longitude, s.elevation)
                             for n in station for s in n}
        if len(station_locations) > 1:
            Logger.warning(
                f"Found multiple locations for "
                f"{amplitude.waveform_id.get_seed_string()}")
        lat, lon, elev_m = station_locations.pop()
        # Calculate slant-distance
        epi_dist_m, _, _ = gps2dist_azimuth(
            lat1=lat,
            lon1=lon,
            lat2=origin.latitude,
            lon2=origin.longitude)
        d_depth_m = elev_m + origin.depth
        slant_dist_m = (epi_dist_m ** 2 + d_depth_m ** 2) ** 0.5
        slant_dist_km = slant_dist_m / 1000.0

        # Cope with differences between seiscomp and EQcorrscan
        if amplitude.unit and amplitude.unit == "m/s":
            amp = amplitude.generic_amplitude * 100.0
            # For some reason we have to scale by 100 to match GeoNet amplitudes...
        elif amplitude.unit == None:
            amp = amplitude.generic_amplitude
        else:
            Logger.error("Unknown amplitude type")
            continue
        station_correction = station_corrections.get(
            amplitude.waveform_id.station_code, None)
        if station_correction is None:
            Logger.warning(f"No station correction found for "
                           f"{amplitude.waveform_id.station_code}, setting to 0")
            station_correction = 0.0
        if origin.depth > 40000:
            h40 = (origin.depth / 1000.0) - 40.0
        else:
            h40 = 0.0
        station_mag = (
            log10(amp) - (
                MLNZ20_CONSTANTS["alpha"] +
                MLNZ20_CONSTANTS["beta"] * slant_dist_km +
                MLNZ20_CONSTANTS["gamma"] * log10(slant_dist_km) +
                MLNZ20_CONSTANTS["delta"] * h40 +
                station_correction))
        station_mag = StationMagnitude(
            origin_id=origin.resource_id,
            mag=station_mag,
            station_magnitude_type=amplitude.type,
            amplitude_id=amplitude.resource_id,
            waveform_id=amplitude.waveform_id,
            creation_info=CreationInfo(
                agency_id="RTEQC",
                author="RTEQcorrscan",
                creation_time=UTCDateTime.now()))
        station_magnitudes.append(station_mag)
    # Compute overall magnitude as mean
    for mag_type in {m.station_magnitude_type for m in station_magnitudes}:
        magnitudes = [m for m in station_magnitudes if m.station_magnitude_type == mag_type]
        magnitude = sum([m.mag for m in magnitudes]) / len(magnitudes)
        event.magnitudes.append(Magnitude(
            mag=magnitude,
            magnitude_type=mag_type,
            origin_id=origin.resource_id,
            method_id="mean",
            station_count=len(magnitudes),
            creation_info=CreationInfo(
                agency_id="RTEQC",
                author="RTEQcorrscan",
                creation_time=UTCDateTime.now()),
            station_magnitude_contributions=magnitudes))
    return event


# ----------------------- Management --------------------------------

MAG_FUNCS = {"MLNZ20": mlnz20}  # Cache of possible magnitude functions


class MagnitudeConfig(_PluginConfig):
    """
    Configuration for the magnitude plugin
    """
    defaults = _PluginConfig.defaults.copy()
    defaults.update({
        "magnitude_function": "MLNZ20",
        "station_correction_file": None,  # Should be a json.
    })
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"magnitude": MagnitudeConfig})


class Magnituder(_Plugin):
    name = "Magnituder"
    inventory_cache = (None, None, None)  # Tuple of (inventory, file, mtime)
    station_correction_cache = (None, None, None)  # Tuple of (station corrections, file, mtime)

    def _read_config(self, config_file: str):
        return MagnitudeConfig.read(config_file=config_file)

    @property
    def inventory(self) -> Union[Inventory, None]:
        return self.inventory_cache[0]

    @property
    def station_corrections(self) -> Union[dict, None]:
        return self.station_correction_cache[0]

    def core(self, new_files: List[str]) -> List[str]:
        processed_files = []

        # Load the stations
        Logger.info("Getting inventory")
        read_inv = False
        if self.inventory_cache[2] is not None:
            # We have read the inventory file, check if we need to re-read
            if self.inventory_cache[1] != self.config.station_file:
                # Different file, we need to read
                read_inv = True
            elif os.path.getmtime(self.config.station_file) > self.inventory_cache[2]:
                # Updated, we need to read
                read_inv = True
        else:
            # No file has been read previously
            read_inv = True
        if read_inv:
            self.inventory_cache = (
                read_inventory(self.config.station_file),
                self.config.station_file,
                os.path.getmtime(self.config.station_file))

        # Getting station corections
        Logger.info("Getting station corrections")
        read_corrections = False
        if self.station_correction_cache[2] is not None:
            if self.station_correction_cache[1] != self.config.station_correction_file:
                read_corrections = True
            elif os.path.getmtime(self.config.station_correction_file) > self.station_correction_cache[2]:
                read_corrections = True
        else:
            read_corrections = True

        if read_corrections:
            with open(self.config.station_correction_file, "r") as f:
                stn_corrs = json.load(f)
            self.station_correction_cache = (
                stn_corrs,
                self.config.station_correction_file,
                os.path.getmtime(self.config.station_correction_file))


        Logger.info(f"Processing {len(new_files)} events")
        for f in new_files:
            self.process_event(filename=f)
            processed_files.append(f)
        return processed_files

    def process_event(self, filename: str):
        cat = read_events(filename)
        for ev in cat:
            MAG_FUNCS.get(self.config.magnitude_function)(
                event=ev, inventory=self.inventory,
                station_corrections=self.station_corrections)
        fname = os.path.basename(filename)
        outpath = os.path.join(self.config.out_dir, fname)
        if not os.path.isdir(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        cat.write(outpath, format="QUAKEML")
        return