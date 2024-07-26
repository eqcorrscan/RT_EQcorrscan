"""
Locate events using hyp.

"""

import copy
import os
import logging
import warnings
import subprocess
import time

from typing import List, Iterable

from obspy import read_events, Catalog, read_inventory
from obspy.core.event import Event, Origin
from obspy.core.inventory import Inventory, Station
from obspy.io.nordic.core import read_nordic, write_select

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)


Logger = logging.getLogger(__name__)


class HypConfig(_PluginConfig):
    """
    Configuration for the hyp plugin.
    """
    defaults = {
        "vmodel_file": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "vmodel.txt"),
        "station_file": "stations.xml",
        "min_stations": 5,
        "max_rms": 2.0,
        "sleep_interval": 600,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"hyp": HypConfig})


# ############################# VELOCITY MODEL CLASSES #########################


class Velocity(object):
    """
    Simple class holding velocity layer.
    Parameters
    ----------
    top:
        Top of layer in km
    vp:
        P-wave velocity of layer in km/s
    moho:
        Mark whether this layer is the moho.
    """
    def __init__(self, top: float, vp: float, moho: bool = False):
        self.top = top
        self.vp = vp
        self.moho = moho

    def __repr__(self):
        return "Velocity(top={0}, vp={1}, moho={2})".format(
            self.top, self.vp, self.moho)

    def __str__(self):
        if self.moho:
            return "{0:7.3f}   {1:7.3f}    N     ".format(self.vp, self.top)
        return "{0:7.3f}   {1:7.3f}".format(self.vp, self.top)


class VelocityModel(object):
    """
    Simple class containing a 1D velocity model.
    """
    def __init__(self, velocities: List[Velocity], vpvs: float):
        self.velocities = velocities
        self.vpvs = vpvs

    def __repr__(self):
        return (f"VelocityModel(<{len(self.velocities)} "
                f"layers>, ..., vpvs={self.vpvs})")

    def __str__(self, format: str = "SEISAN"):
        if format.upper() == "SEISAN":
            lines = ["VelocityModel"]
            for v in self.velocities:
                lines.append(f"{v.top},{v.vp},{v.moho}")
            lines.append(f"vpvs: {self.vpvs}")
            return "\n".join(lines)
        elif format.upper() == "GROWCLUST":
            lines = []
            for v in self.velocities:
                lines.append(
                    f"{v.top:4.1f} {v.vp:2.1f} {v.vp / self.vpvs:2.1f}")
            return "\n".join(lines)

    def write(self, filename: str, format: str = "SEISAN"):
        with open(filename, "w") as f:
            f.write(self.__str__(format=format))

    @classmethod
    def read(cls, filename: str):
        with open(filename, "r") as f:
            lines = f.read().splitlines()
        assert lines[0] == "VelocityModel", "Unknown format"

        velocities = []
        i = 1
        while True:
            line = lines[i]
            if line.startswith("vpvs"):
                break
            parts = line.split(',')
            velocities.append(
                Velocity(
                    top=float(parts[0]),
                    vp=float(parts[1]),
                    moho=parts[2] == "True"
                    )
                )
            i += 1
        vpvs = float(line.split()[-1])
        return cls(velocities=velocities, vpvs=vpvs)


# ##################################### END OF VELOCITY MODELS ################

# ################################# SEISAN HYP CALLS ##########################

def seisan_hyp(
    event: Event,
    inventory: Inventory,
    velocities: List[Velocity],
    vpvs: float,
    remodel: bool = True,
    clean: bool = True
) -> Event:
    """
    Use SEISAN's Hypocentre program to locate an event.
    """


    # Write STATION0.HYP file
    _write_station0(inventory, velocities, vpvs)

    if remodel:
        subprocess.run(['remodl'], capture_output=True)
        subprocess.run(['setbrn'], capture_output=True)

    event_out = event.copy()
    try:
        old_origin = event.preferred_origin() or event.origins[0]
        origin = Origin(time=old_origin.time)
    except IndexError:
        origin = Origin(
            time=sorted(event.picks, key=lambda p: p.time)[0].time)
    event_out.origins = [origin]
    event_out.comments = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_select(catalog=Catalog([event_out]), high_accuracy=False,
                     filename="to_be_located")
    loc_proc = subprocess.run(
        ['hyp', "to_be_located"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    for line in loc_proc.stdout.decode().splitlines():
        Logger.info(">>> " + line.rstrip())
        # print(">>> " + line.rstrip())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            event_back = read_nordic("hyp.out")
        except Exception as e:
            Logger.error(f"Could not read hyp.out due to {e}")
            return None
    # We lose some info in the round-trip to nordic
    event_out.origins[0] = event_back[0].origins[0]
    event_out.magnitudes = event_back[0].magnitudes
    # Fix the seed ids in the seisan picks
    for pick in event_back[0].picks:
        matched_pick = [
            p for p in event_out.picks
            if p.waveform_id.station_code == pick.waveform_id.station_code and
            p.waveform_id.channel_code[-1] == pick.waveform_id.channel_code[-1] and
            abs(p.time - pick.time) < 0.1]
        assert len(matched_pick) > 0, "No picks matched"
        assert len(set(p.waveform_id.get_seed_string()
                       for p in matched_pick)) == 1,\
            f"Multiple seed ids for matched picks:\n{matched_pick}"
        Logger.info(f"Matched {pick.waveform_id.get_seed_string()} to "
                    f"{matched_pick[0].waveform_id.get_seed_string()}")
        pick.waveform_id.network_code = matched_pick[0].waveform_id.network_code
        pick.waveform_id.station_code = matched_pick[0].waveform_id.station_code
        pick.waveform_id.location_code = matched_pick[0].waveform_id.location_code
        pick.waveform_id.channel_code = matched_pick[0].waveform_id.channel_code

    event_out.picks = event_back[0].picks
    event_out.comments = copy.deepcopy(event.comments)  # add comments back in
    if clean:
        _cleanup()
    return event_out


def iterate_over_event(
    event: Event,
    inventory: Inventory,
    velocities: List[Velocity],
    vpvs: float,
    max_rms: float = 1.0,
    min_stations: int = 5,
    clean: bool = True,
    remodel: bool = True,
) -> Event:
    """ Iterataivly remove picks until quality criteria are met. """
    n_stations = len({p.waveform_id.station_code for p in event.picks})
    if n_stations < min_stations:
        Logger.info("{0} stations picked, returning None".format(n_stations))
        return None
    event_to_be_located = event.copy()  # Keep the original event safe
    while n_stations >= min_stations:
        event_located = seisan_hyp(
            event=event_to_be_located, inventory=inventory,
            velocities=velocities, vpvs=vpvs, clean=clean, remodel=remodel)
        remodel = False
        if event_located is None:
            Logger.error("Issue locating event, returning None")
            return None
        if len(event_located.origins) == 0 or event_located.origins[0].latitude == None:
            Logger.warning("Issue locating event, returning None")
            return None
        if event_located.origins[0].quality.standard_error <= max_rms:
            Logger.info("RMS below max_RMS, good enough!")
            event_to_be_located.origins.append(event_located.origins[0])
            return event_to_be_located
        # Remove least well fit pick and go again
        worst_arrival = event_located.origins[0].arrivals[0]
        for arr in event_located.origins[0].arrivals[1:]:
            try:
                if arr.time_residual > worst_arrival.time_residual:
                    worst_arrival = arr
            except TypeError:
                Logger.error(arr)
        for pick in event_located.picks:
            if pick.resource_id == worst_arrival.pick_id:
                worst_pick = pick
                Logger.info("Removing pick at {0} on {1}".format(
                    pick.time, pick.waveform_id.get_seed_string()))
                break
        _picks = []
        for pick in event_to_be_located.picks:
            if pick.waveform_id.station_code == worst_pick.waveform_id.station_code and abs(pick.time - worst_pick.time) < 0.01:
                continue
            _picks.append(pick)
        assert len(_picks) < len(event_to_be_located.picks)
        event_to_be_located.picks = _picks
        n_stations = len({p.waveform_id.station_code
                          for p in event_located.picks})


    Logger.info("{0} stations picked, returning None".format(n_stations))
    return None



_HYP_TMP_FILES = [
        "hyp.out", "to_be_located", "remodl.tbl", "remodl1.lis", "remodl2.lis",
        "print.out", "gmap.cur.kml", "hypmag.out", "hypsum.out", "remodl.hed",
        "IASP91_linux.HED", "IASP91_linux.TBL", "setbrn1.lis", "setbrn2.lis",
        "setbrn3.lis", "STATION0.HYP", "focmec.dat", "focmec.inp", "fort.17",
        "fps.out", "hash_seisan.out", "pspolar.inp", "scratch1.out",
        "scratch2.out", "scratch3.out"]


def _cleanup():
    # Clean up
    for f in _HYP_TMP_FILES:
        if os.path.isfile(f):
            os.remove(f)


def _stationtoseisan(station: Station) -> str:
    """
    Convert obspy inventory to string formatted for Seisan STATION0.HYP file.
    :param station: Inventory containing a single station.
    .. note::
        Only works to the low-precision level at the moment (see seisan
        manual for explanation).
    """

    if station.latitude < 0:
        lat_str = 'S'
    else:
        lat_str = 'N'
    if station.longitude < 0:  # Stored in =/- 180, not 0-360
        lon_str = 'W'
    else:
        lon_str = 'E'
    if len(station.code) > 4:
        sta_str = station.code[0:4]
    else:
        sta_str = station.code.ljust(4)
    if len(station.channels) > 0:
        depth = station.channels[0].depth
    else:
        msg = 'No depth found in station.channels, have you set the level ' +\
              'of stationXML download to channel if using obspy.get_stations?'
        raise IOError(msg)
    elev = str(int(round(station.elevation - depth))).rjust(4)
    # lat and long are written in STATION0.HYP in deg,decimal mins
    lat = abs(station.latitude)
    lat_degree = int(lat)
    lat_decimal_minute = (lat - lat_degree) * 60
    lon = abs(station.longitude)
    lon_degree = int(lon)
    lon_decimal_minute = (lon - lon_degree) * 60
    lat = ''.join([str(int(abs(lat_degree))),
                   '{0:.2f}'.format(lat_decimal_minute).rjust(5)])
    lon = ''.join([str(int(abs(lon_degree))),
                   '{0:.2f}'.format(lon_decimal_minute).rjust(5)])
    station_str = ''.join(['  ', sta_str, lat, lat_str, lon, lon_str, elev])
    return station_str


def _write_station0(
    inventory: Inventory,
    velocities: List[Velocity],
    vpvs: float
):
    out = (
        "RESET TEST(02)=500.0\nRESET TEST(07)=-3.0\nRESET TEST(08)=2.6\n"
        "RESET TEST(09)=0.001\nRESET TEST(11)=99.0\nRESET TEST(13)=5.0\n"
        "RESET TEST(34)=1.5\nRESET TEST(35)=2.5\nRESET TEST(36)=0.0\n"
        "RESET TEST(41)=20000.0\nRESET TEST(43)=5.0\nRESET TEST(51)=3.6\n"
        "RESET TEST(50)=1.0\nRESET TEST(56)= 1.0\nRESET TEST(58)= 99990.0\n"
        "RESET TEST(40)=0.0\nRESET TEST(60)=0.0\nRESET TEST(71)=1.0\n"
        "RESET TEST(75)=1.0\nRESET TEST(76)=0.910\nRESET TEST(77)=0.00087\n"
        "RESET TEST(78)=-1.67\nRESET TEST(79)=1.0\nRESET TEST(80)=3.0\n"
        "RESET TEST(81)=1.0\nRESET TEST(82)=1.0\nRESET TEST(83)=1.0\n"
        "RESET TEST(88)=1.0\nRESET TEST(85)=0.1\nRESET TEST(91)=0.1\n")
    for network in inventory:
        for station in network:
            out += "\n" + _stationtoseisan(station)
    out += "\n\n"
    # Add velocity model
    for layer in velocities:
        out += "{0}\n".format(layer)
    out += "\n15.0 1100.2200. {0:.2f} \nTES\n".format(vpvs)
    with open("STATION0.HYP", "w") as f:
        f.write(out)
    return


# ################################# END OF SEISAN HYP CALLS ###################


# ################################# CONTROL FUNCS #############################

class Hyp(_Plugin):
    remodel = True

    def _read_config(self, config_file: str):
        return HypConfig.read(config_file)

    def _cleanup(self):
        _cleanup()

    def core(self, new_files: Iterable) -> List:
        internal_config = self.config.copy()

        in_dir = internal_config.pop("in_dir")
        out_dir = internal_config.pop("out_dir")
        vmodel_file = internal_config.pop("vmodel_file")
        station_file = internal_config.pop("station_file")

        # TODO: Figure this out - we need to read the vmodel and station inv
        inv = read_inventory(station_file)
        vmodel = VelocityModel.read(vmodel_file)

        processed_files = []
        for i, infile in enumerate(new_files):
            Logger.info(
                f"Working on event-file {i} of {len(new_files)}:\t{infile}")
            try:
                _cat = read_events(infile)
            except Exception as e:
                Logger.error(f"Could not read {infile} due to {e}")
                continue
            cat_out = Catalog()
            failed = False
            for event in _cat:
                try:
                    event_located = iterate_over_event(
                        event=event, inventory=inv,
                        velocities=vmodel.velocities, vpvs=vmodel.vpvs,
                        remodel=self.remodel, clean=False,
                        max_rms=internal_config.max_rms,
                        min_stations=internal_config.min_stations)
                    # event_located = seisan_hyp(
                    #     event=event, inventory=inv,
                    #     velocities=vmodel.velocities,
                    #     vpvs=vmodel.vpvs, remodel=self.remodel, clean=False)
                except Exception as e:
                    Logger.error(f"Could not locate {event.resource_id.id} due "
                                 f"to {e}")
                    failed = True
                    continue
                if event_located:
                    self.remodel = False
                    # Do not redo that work if we don't need to
                    cat_out += event_located
                else:
                    failed = True
            fname = infile.split(in_dir)[-1]
            fname = fname.lstrip(os.path.sep)  # Strip pathsep if it is there
            outpath = os.path.join(out_dir, fname)
            Logger.info(f"Writing located event to {outpath}")
            if not os.path.isdir(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            if len(cat_out):
                cat_out.write(outpath, format="QUAKEML")

            if not failed:
                processed_files.append(infile)
        return processed_files


if __name__ == "__main__":
    import doctest

    doctest.testmod()
