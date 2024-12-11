"""
Relocate!

Steps:
1. Read in event data
2. Read in relevant waveform data
3. Correlate
4. Output to GrowClust format
5. Run Growclust
6. Read growclust output and associate new locs with events
7. Output events
"""

import logging
import math
import os
import shutil
import tempfile

import numpy as np

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:  # pragma: no cover
    from yaml import Loader, Dumper

from scipy.stats import circmean

from typing import Iterable, List, Set

from obspy import (
    read_events, Catalog, UTCDateTime, read_inventory, Inventory)
from obspy.core.event import (
    Origin, ResourceIdentifier, OriginUncertainty, OriginQuality)
from obspy.geodetics import gps2dist_azimuth

from eqcorrscan.utils.catalog_to_dd import write_correlations, write_phase

from rt_eqcorrscan.plugins.waveform_access import InMemoryWaveBank
from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.relocation.hyp_runner import VelocityModel
from rt_eqcorrscan.plugins.plugin import _Plugin, PLUGIN_CONFIG_MAPPER
from rt_eqcorrscan.plugins.relocation.dt_correlations.correlator import (
    Correlator, SparseEvent)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
GROWCLUST_DEFAULT_VMODEL = f"{WORKING_DIR}/vmodel.txt"
GROWCLUST_SCRIPT = f"{WORKING_DIR}/run_growclust3D.jl"

_GC_TMP_FILES = []  # List of temporary files to remove

Logger = logging.getLogger(__name__)

# CORRELATION COMPUTATION PARAMETERS

class _GrowClustProj(_PluginConfig):
    defaults = {
        "proj": "tmerc",
        "ellps": "WGS84",
        "lon0": 0.0,
        "lat0": 0.0,
        "rotANG": 0.0,
        "latP1": None,
        "latP2": None}

    def __str__(self):
        _str = f"{self.proj} {self.ellps} {self.lon0} {self.lat0} {self.rotANG}"
        if not self.latP1 is None and not self.latP2 is None:
            _str += f" {self.latP1} {self.latP2}"
        return _str


class CorrelationConfig(_PluginConfig):
    """
    Subclass for correlation configuration for relocations.
    """
    defaults = {
        "extract_len": 2.0,
        "pre_pick": 0.2,
        "shift_len": 0.2,
        "lowcut": 1.0,
        "highcut": 20.0,
        "max_sep": 8,
        "min_link": 8,
        "min_cc": 0.2,
        "interpolate": False,
        "weight_by_square": True,
        "max_event_links": None,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_yaml_dict(self):
        """ Overload. """
        yaml_dict = dict()
        for key, value in self.__dict__.items():
            if hasattr(value, "to_yaml_dict"):
                yaml_dict.update({
                    key.replace("_", " "): value.to_yaml_dict()})
            else:
                yaml_dict.update({key.replace("_", " "): value})
        return yaml_dict


class GrowClustConfig(_PluginConfig):
    """
    Configuration holder for GrowClust.

    Note that this would work better if we just wrote a json file then read
    that json file in julia, but I can't be bothered to edit the julia code
    yet.
    """
    defaults = {
        "evlist_fmt": "1",
        "fin_evlist": "evlist.txt",
        "stlist_fmt": "2",
        "fin_stlist": "stlist.txt",
        "xcordat_fmt": "1 12",
        "fin_xcordat": "dt.cc",
        "ttabsrc": "trace",
        "fin_vzmdl": "vzmodel.txt",
        "fdir_ttab": "tt",
        "projection": _GrowClustProj(),  # TODO: For 3D, projection parameters must match the NLL Grid parameters.
        "vpvs_factor": 1.732,
        "rayparam_min": 0.0,
        "tt_zmin": -4.0,
        "tt_zmax": 500.0,
        "tt_zstep": 1.0,
        "tt_xmin": 0.0,
        "tt_xmax": 1000.0,
        "tt_ymin": None,
        "tt_ymax": None,
        "tt_xstep": 1.0,
        "tdifmax": 30.0,
        "hshiftmax": 2.0,
        "vshiftmax": 2.0,
        "rmin": 0.6,
        "delmax": 80,
        "rmsmax": 0.2,
        "rpsavgmin": 0,
        "rmincut": 0,
        "ngoodmin": 0,
        "iponly": 0,
        "nboot": 0,
        "nbranch_min": 2,
        "fout_cat": "growclust_out.cat",
        "fout_clust": "growclust_out.clust",
        "fout_log": "growclust_out.log",
        "fout_boot": "NONE",
        "station_file": "stations.xml",
        "vmodel_file": GROWCLUST_DEFAULT_VMODEL,
        "growclust_script": GROWCLUST_SCRIPT,
        "sleep_interval": 600,
        "correlation_config": CorrelationConfig(),
        "nll_config_file": None,
    }
    readonly = []
    _subclasses = {
        "projection": _GrowClustProj,
        "correlation_config": CorrelationConfig,
    }

    def __init__(self, *args, **kwargs):
        attribs = dict()
        for key, value in kwargs.items():
            if key in self._subclasses.keys():
                if isinstance(value, dict):
                    value = self._subclasses[key](**value)
            attribs.update({key: value})
        super().__init__(*args, **attribs)

    @classmethod
    def read(cls, config_file: str):
        from rt_eqcorrscan.config.config import _recursive_replace_space_underscore

        with open(config_file, "rb") as f:
            configuration = load(f, Loader=Loader)
        configuration = {key.replace(" ", "_"): value
                         for key, value in configuration.items()}
        # Cope with nested subclasses
        config_dict = {}
        for key, value in configuration.items():
            if key.replace(" ", "_") in cls._subclasses.keys():
                config_dict.update(
                    _recursive_replace_space_underscore({key: value}))
            else:
                config_dict.update({key: value})
        return cls(**config_dict)

    def to_yaml_dict(self):
        """ Overload. """
        yaml_dict = dict()
        for key, value in self.__dict__.items():
            if hasattr(value, "to_yaml_dict"):
                yaml_dict.update({
                    key.replace("_", " "): value.to_yaml_dict()})
            else:
                yaml_dict.update({key.replace("_", " "): value})
        return yaml_dict

    def write_growclust(self, filename: str = "growclust.inp"):
        """
        Write the input file for growclust.

        The growclust file is fixed lines with comments between.
        """
        if self.tt_ymin is not None and self.tt_ymax is not None and self.ttabsrc == "nllgrid":
            tt_lines = [
                "* tt_zmin  tt_zmax",
                f"{self.tt_zmin}   {self.tt_zmax}",
                "* tt_xmin  tt_xmax  tt_ymin  tt_ymax",
                f"{self.tt_xmin}  {self.tt_xmax}  {self.tt_ymin}  {self.tt_ymax}",
            ]
        else:
            tt_lines = [
                "* tt_zmin  tt_zmax  tt_zstep",
                f"{self.tt_zmin}   {self.tt_zmax}  {self.tt_zstep}",
                "* tt_xmin  tt_xmax  tt_xstep",
                f"{self.tt_xmin}  {self.tt_xmax}  {self.tt_xstep}",
            ]
        lines = [
            "* Growclust parameter file written by RT-EQcorrscan",
            "* evlist_fmt",
            self.evlist_fmt,
            "* fin_evlist",
            self.fin_evlist,
            "* "
            "* stlist (1: station name, 2: incl. elev)",
            self.stlist_fmt,
            "* fin_stlist (station list file name)",
            self.fin_stlist,
            "*"
            "* xcordat_fmt (1 = text), tdif_fmt (21 = tt2-tt1, 12 = tt1-tt2)",
            self.xcordat_fmt,
            "* fin_xcordat",
            self.fin_xcordat,
            "*",
            "* ttabsrc: travel time table source ('trace' or 'nllgrid')",
            self.ttabsrc,
            "* fin_vzmdl (model name)",
            self.fin_vzmdl,
            "* fdir_ttab (directory for travel time tables/grids or NONE)",
            self.fdir_ttab,
            "* projection (proj, ellps, lon0, lat0, rotANG, [latP1, latP2])",
            str(self.projection),
            "* vpvs_factor  rayparam_min",
            f"{self.vpvs_factor}         {self.rayparam_min}",
            tt_lines[0], tt_lines[1], tt_lines[2], tt_lines[3],
            "* rmin  delmax rmsmax",
            f"{self.rmin}  {self.delmax}  {self.rmsmax}",
            "* rpsavgmin, rmincut  ngoodmin   iponly",
            f"{self.rpsavgmin}  {self.rmincut}  {self.ngoodmin}  {self.iponly}",
            "* nboot  nbranch_min",
            f"{self.nboot}    {self.nbranch_min}",
            "* fout_cat (relocated catalog)",
            self.fout_cat,
            "* fout_clust (relocated cluster file)",
            self.fout_clust,
            "* fout_log (program log)",
            self.fout_log,
            "* fout_boot (bootstrap distribution)",
            self.fout_boot
        ]
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        return


PLUGIN_CONFIG_MAPPER.update({"growclust": GrowClustConfig})

#############################################################################
#                    Runners
#############################################################################


def get_catalog_mean_location(
    catalog: Catalog
) -> [float, float]:
    """
    Get the mean latitude and longitude of a catalog

    Parameters
    ----------
    catalog
        Catalog of events with origins

    Returns
    -------
    mean latitude, mean longitude
    """
    origins = []
    for ev in catalog:
        try:
            origin =ev.preferred_origin() or ev.origins[-1]
        except IndexError:
            continue
        origins.append(origin)
    lats = np.array([origin.latitude for origin in origins])
    lons = np.array([origin.longitude for origin in origins])

    mean_lat = np.mean(lats)
    mean_lon = np.degrees(circmean(np.radians(lons)))
    return mean_lat, mean_lon


def check_events_within_bounds(
    catalog: Catalog,
    mean_lat: float,
    mean_lon: float,
    tt_xmin: float,
    tt_xmax: float,
    tt_zmin: float,
    tt_zmax: float
) -> Catalog:
    """
    Check that events are within the bounds, and remove those outside bounds

    Parameters
    ----------
    catalog
    mean_lat
    mean_lon
    tt_xmin
    tt_xmax
    tt_zmin
    tt_zmax

    Returns
    -------

    """
    # Every event needs an origin
    thresholded_events = [ev for ev in catalog if len(ev.origins)]
    thresholded_events = [
        ev for ev in thresholded_events if
        tt_zmin <= (ev.preferred_origin() or
                    ev.origins[-1]).depth / 1000.0 <= tt_zmax]
    x_dists = np.array([gps2dist_azimuth(
        mean_lat, mean_lon,
        (ev.preferred_origin() or ev.origins[-1]).latitude,
        (ev.preferred_origin() or ev.origins[-1]).longitude)[0] / 1000.
               for ev in thresholded_events])
    x_mask = np.logical_and(x_dists >= tt_xmin, x_dists <= tt_xmax)
    thresholded_events = [ev for i, ev in enumerate(thresholded_events)
                          if x_mask[i]]

    return Catalog(thresholded_events)


def run_growclust(
    mean_lat: float,
    mean_lon: float,
    workers: [int, str] = "auto",
    config: GrowClustConfig = GrowClustConfig(),
    vmodel_file: str = GROWCLUST_DEFAULT_VMODEL,
    growclust_script: str = GROWCLUST_SCRIPT,
) -> str:
    """
    Requires a growclust julia runner script.
    """
    import subprocess
    # Need to edit the projection origin for the events

    internal_config = config.copy()
    if internal_config.ttabsrc == "trace":
        internal_config.projection.lat0 = mean_lat
        internal_config.projection.lon0 = mean_lon
        vmodel = VelocityModel.read(vmodel_file)
        vmodel.write(internal_config.fin_vzmdl, format="GROWCLUST")
    elif internal_config.ttabsrc == "nllgrid":
        with open(internal_config.nll_config_file, "r") as f:
            nll_config = {l.split()[0]: l.split()[1:] for l in f}
        # Match params to nllgrid
        config.fin_vzmdl = os.path.split(nll_config["GTFILES"][2])[-1]
        config.fdir_ttab = os.path.join(
            os.path.dirname(os.path.abspath(internal_config.nll_config_file)),
            os.path.split(nll_config["GTFILES"][2])[0])
        config.projection = _GrowClustProj()
        config.tt_zmin = float(nll_config["VGGRID"][5])
        config.tt_zmax = float(nll_config["VGGRID"][5]) + (
            float(nll_config["VGGRID"][2]) * float(nll_config["VGGRID"][8]))
        config.tt_xmin = float(nll_config["VGGRID"][3])
        config.tt_xmax = float(nll_config["VGGRID"][3]) + (
            float(nll_config["VGGRID"][0]) * float(nll_config["VGGRID"][6]))
        config.tt_ymin = float(nll_config["VGGRID"][4])
        config.tt_ymax = float(nll_config["VGGRID"][4]) + (
            float(nll_config["VGGRID"][1]) * float(nll_config["VGGRID"][7]))
        Logger.info(f"Config edited to match nonlinloc:\n{config}")
    else:
        raise NotImplementedError(
            "Only ttabsrc in ['trace', 'nllgrid'] supported")


    internal_config.write_growclust(f"growclust_control.inp")

    # TODO: Re-write Julia code to only do ray-tracing once?
    # TODO: If caching travel-times then the lat and lon or the
    #  origin of the coord system must be preserved - global mutable
    #  variables?

    arg_string = [
        "julia",
        "--threads", str(workers),
        growclust_script,
        "-c", f"growclust_control.inp",
        "--tdifmax", f"{config.tdifmax}",
        "--hshiftmax", f"{config.hshiftmax}",
        "--vshiftmax", f"{config.vshiftmax}"]
    Logger.info(f"Running call: {' '.join(arg_string)}")

    loc_proc = subprocess.run(
        arg_string,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    for line in loc_proc.stdout.decode().splitlines():
        Logger.info(">>> " + line.rstrip())
    try:
        loc_proc.check_returncode()
    except Exception as e:
        Logger.exception(e)
    return internal_config.fout_cat


def write_stations(
    seed_ids: Iterable,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    station_file: str,
):
    import fnmatch

    inv = read_inventory(station_file)
    inv = inv.select(starttime=starttime, endtime=endtime)
    # Select based on station
    used_inv = Inventory()
    for net in inv:
        used_stations = []
        for sta in net:
            if len(sta) == 0:
                # No associated channels - fine.
                sid = f"{net.code}.{sta.code}.*.*"
                filt = fnmatch.filter(seed_ids, sid)
                if len(filt):
                    used_stations.append(sta)
                continue
            # There are channels - we should check them
            used_channels = []
            for chan in sta:
                sid = f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}"
                if sid in seed_ids:
                    used_channels.append(chan)
            if len(used_channels):
                sta = sta.copy()
                sta.channels = used_channels
                used_stations.append(sta)
        if len(used_stations):
            net = net.copy()
            net.stations = used_stations
            used_inv += net
    inv = used_inv

    write_stlist(inv)
    return


def write_stlist(inv):
    lineformat = "{station:6s}{latitude:8.4f}{longitude:10.4f}{elevation:5d}"
    lines = []
    for net in inv:
        for sta in net:
            line = lineformat.format(
                station=sta.code,
                latitude=sta.latitude,
                longitude=sta.longitude,
                elevation=int(sta.elevation.real)
            )
            lines.append(line)
    with open("stlist.txt", "w") as f:
        f.write("\n".join(lines))


def phase_to_evlist(input_file: str):
    with open(input_file, "r") as f:
        lines = [l for l in f.read().splitlines() if l.startswith("#")]
    out_lines = [l.lstrip("#").lstrip() for l in lines]
    with open("evlist.txt", "w") as f:
        for line in out_lines:
            f.write(f"{line}\n")
    return


def process_output_catalog(
    catalog: Catalog,
    event_mapper: dict,
    outfile: str
) -> Catalog:
    """ Add Growclust origins back into catalog. """
    if not os.path.isfile(outfile):
        Logger.warning(f"{outfile} does not exist.")
        # return catalog
        return Catalog()
    gc_origins = read_growclust(
        outfile, event_mapper={val: key for key, val in event_mapper.items()})

    catalog_dict = {ev.resource_id.id: ev for ev in catalog}

    catalog_out, relocated, not_relocated = Catalog(), Catalog(), Catalog()
    for key, ev in catalog_dict.items():
        gc_origin, _relocated = gc_origins.get(key)
        if gc_origin is None or not _relocated:
            Logger.warning(f"Event {key} did not relocate")
            not_relocated += ev
            # catalog_out += ev
            continue
        ev.origins.append(gc_origin)
        ev.preferred_origin_id = ev.origins[-1].resource_id
        relocated += ev
        catalog_out += ev
    Logger.info(f"Of {len(catalog)}, {len(relocated)} were relocated, and "
                f"{len(not_relocated)} were not relocated")

    return catalog_out


########## GROWCLUST READING ############################

"""
2009  1 18  0 49 21.000         1 -42.23833  173.79283  19.900  2.85       \
1   12359       1     0     0     0  0.00  0.00  -1.000  -1.000  -1.000   \
-42.23833  173.79283  19.900
"""

FORMATTER = {
    "year": (0, int),
    "month": (1, int),
    "day": (2, int),
    "hour": (3, int),
    "minute": (4, int),
    "second": (5, float),
    "eventid": (6, int),
    "latitude": (7, float),
    "longitude": (8, float),
    "depth": (9, float),
    "magnitude": (10, float),
    "qID": (11, int),
    "cID": (12, int),
    "nbranch": (13, int),
    "qnpair": (14, int),
    "qndiffP": (15, int),
    "qndiffS": (16, int),
    "rmsP": (17, float),
    "rmsS": (18, float),
    "eh": (19, float),
    "ez": (20, float),
    "et": (21, float),
    "latitude_original": (22, float),
    "longitude_original": (23, float),
    "depth_origins": (24, float)}


def growclust_line_to_origin(line: str) -> [str, Origin, bool]:
    line = line.split()
    deserialized = {key: val[1](line[val[0]]) for key, val in FORMATTER.items()}
    relocated = deserialized["nbranch"] > 1
    # If the cluster only had one event, it wasn't relocated.
    # Replace nans with None
    for key, val in deserialized.items():
        if math.isnan(val):
            val = None
            deserialized.update({key: val})
    if deserialized["eh"] == -1.:
        # print(f"Event {deserialized['eventid']} not relocated")
        return deserialized["qID"], None
    origin_time = UTCDateTime(
        year=deserialized["year"], month=deserialized["month"],
        day=deserialized["day"], hour=deserialized["hour"],
        minute=deserialized["minute"]) + deserialized["second"]
    try:
        p_standard_error = deserialized["rmsP"] or 0 / deserialized["qndiffP"]
    except ZeroDivisionError:
        p_standard_error = 0.0
    try:
        s_standard_error = deserialized["rmsS"] or 0 / deserialized["qndiffS"]
    except ZeroDivisionError:
        s_standard_error = 0.0
    origin = Origin(
        latitude=deserialized["latitude"], longitude=deserialized["longitude"],
        depth=deserialized["depth"] * 1000, time=origin_time,
        method_id=ResourceIdentifier("GrowClust"),
        time_errors={"uncertainty": deserialized["et"]},
        depth_errors={
            "uncertainty": deserialized["ez"] * 1000.0
            if deserialized["ez"] else None},
        time_fixed=False,
        origin_uncertainty=OriginUncertainty(
            horizontal_uncertainty=deserialized["eh"] * 1000.0
            if deserialized["eh"] else None),
        quality=OriginQuality(
            used_phase_count=deserialized["qndiffP"] + deserialized["qndiffS"],
            standard_error=(
                deserialized["qndiffP"] + deserialized["qndiffS"]) *
                (p_standard_error + s_standard_error)))
    return deserialized["eventid"], origin, relocated


def read_growclust(
    fname: str = "OUT/growclust_out.trace1D.cat",
    event_mapper: dict = None) -> dict:
    """
    Read growclust origins from a relocated file.

    Parameters
    ----------
    fname:
        File to read from - should be a growclust_cat file.
    event_mapper:
        Event id mapping of {growclust id: desired id}

    Returns
    -------
    Dictionary of origins and whether they weere relocated keyed by event id.
    """
    with open(fname, "r") as f:
        lines = f.read().splitlines()

    origins = dict()
    for line in lines:
        event_id, growclust_origin, relocated = growclust_line_to_origin(line)
        if event_mapper:
            event_id = event_mapper.get(event_id, f"{event_id}_notmapped")
        origins.update({event_id: (growclust_origin, relocated)})
    return origins

#############################################################################
#               Core Growclust runner funcs and class
#############################################################################

def _cleanup():
    # Clean up
    for f in _GC_TMP_FILES:
        if os.path.isfile(f):
            os.remove(f)


class GrowClust(_Plugin):
    _cc_file = "dt.cc"
    def __init__(self, config_file: str, name: str = "GrowClustRunner"):
        super().__init__(config_file=config_file, name=name)
        self.in_memory_wavebank = InMemoryWaveBank(self.config.wavebank_dir)
        self.in_memory_wavebank.get_data_availability()
        self.correlator = Correlator(
            minlink=self.config.correlation_config.min_link,
            maxsep=self.config.correlation_config.max_sep,
            shift_len=self.config.correlation_config.shift_len,
            pre_pick=self.config.correlation_config.pre_pick,
            length=self.config.correlation_config.extract_len,
            lowcut=self.config.correlation_config.lowcut,
            highcut=self.config.correlation_config.highcut,
            interpolate=self.config.correlation_config.interpolate,
            client=self.in_memory_wavebank,
            min_cc=self.config.correlation_config.min_cc,
            weight_by_square=self.config.correlation_config.weight_by_square,
            outfile=self._cc_file,
            max_event_links=self.config.correlation_config.max_event_links,
            )
        self._all_event_files = dict()
        # Keep record of all event files keyed by event id

    def _read_config(self, config_file: str):
        return GrowClustConfig.read(config_file)

    def _cleanup(self):
        _cleanup()

    def run_growclust(
        self,
        outdir: str,
        station_file: str,
        event_files: dict,
        workers: [int, str] = "auto",
        config: GrowClustConfig = GrowClustConfig(),
        vmodel_file: str = GROWCLUST_DEFAULT_VMODEL,
        growclust_script: str = GROWCLUST_SCRIPT,
        cleanup: bool = True,
    ):
        catalog = Catalog()
        for value in event_files.values():
            try:
                cat = read_events(value)
            except Exception as e:
                Logger.debug(f"Could not read events from {value} due to {e}")
                continue
            catalog += cat
        if len(catalog) <= 2:
            Logger.info("Not enough events for relocation, skipping")
            return
        Logger.info(f"Prepping growclust files for {len(catalog)} events")
        # Do the mahi in a tempdir
        cwd = os.path.abspath(os.path.curdir)
        working_dir = os.path.join(cwd, ".growclust_working")
        if os.path.isdir(working_dir):
            Logger.warning(f"Planned working directory ({working_dir}) exists, "
                           f"files will be overwritten")
        else:
            os.makedirs(working_dir)
        outdir, station_file = map(os.path.abspath, (outdir, station_file))
        Logger.info(f"Working in {working_dir}")
        shutil.copyfile(self._cc_file,
                        os.path.join(working_dir, "dt.cc"))
        Logger.info(f"Copied correlation file from {self._cc_file} to "
                    f"{os.path.join(working_dir, 'dt.cc')}")
        os.chdir(working_dir)

        # In temp dir
        # Find centroid
        mean_lat, mean_lon = get_catalog_mean_location(catalog)
        # Check that locations fall within ray-tracing bounds
        input_events = len(catalog)
        inv = read_inventory(station_file)
        for net in inv:
            for sta in net:
                catalog = check_events_within_bounds(
                    catalog=catalog, mean_lat=sta.latitude, 
                    mean_lon=sta.longitude,
                    tt_xmin=config.tt_xmin, tt_xmax=config.tt_xmax,
                    tt_zmin=config.tt_zmin, tt_zmax=config.tt_zmax)
        Logger.info(
            f"Of {input_events} input events, {input_events - len(catalog)} "
            f"were removed due to not being within bounds")
        event_mapper = write_phase(
            catalog, event_id_mapper=self.correlator.event_mapper)
        phase_to_evlist("phase.dat")
        # Write station file
        seed_ids = {p.waveform_id.get_seed_string()
                    for ev in catalog for p in ev.picks}
        # Events don't *have* to have an origin, so we still need to use picks
        starttime = min(p.time for ev in catalog for p in ev.picks)
        endtime = max(p.time for ev in catalog for p in ev.picks)
        write_stations(seed_ids=seed_ids, starttime=starttime, endtime=endtime,
                       station_file=station_file)
        # Run growclust
        Logger.info("Running growclust")
        outfile = run_growclust(
            mean_lat=mean_lat, mean_lon=mean_lon, config=config,
            workers=workers, vmodel_file=vmodel_file,
            growclust_script=growclust_script)

        catalog_out = process_output_catalog(
            catalog=catalog, event_mapper=event_mapper, outfile=outfile)
        os.chdir(cwd)

        # Out of tempdir
        if cleanup:
            self._cleanup()
            Logger.info(f"Removing {working_dir} and all files therein")
            shutil.rmtree(working_dir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for ev in catalog_out:
            fname = os.path.split(event_files[ev.resource_id.id])[-1]
            fname = fname.lstrip(os.path.sep)  # Strip pathsep if it is there
            outpath = os.path.join(outdir, fname)
            Logger.info(f"Writing out to {outpath}")
            if not os.path.isdir(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            ev.write(f"{outpath}", format="QUAKEML")
        return


    def core(
        self,
        new_files: Iterable,
        workers: int = None,
        cleanup: bool = True
    ) -> List:
        workers = workers or 1
        internal_config = self.config.copy()
        # indir = internal_config.pop("in_dir")
        outdir = internal_config.pop("out_dir")
        station_file = internal_config.pop("station_file")
        growclust_script = internal_config.pop(
            "growclust_script", GROWCLUST_SCRIPT)
        vmodel_file = internal_config.pop(
            "vmodel_file", GROWCLUST_DEFAULT_VMODEL)

        new_events = []
        for f in new_files:
            f = os.path.abspath(f)
            try:
                _cat = read_events(f)
            except Exception as e:
                Logger.warning(f"Could not read from {f} due to {e}")
                continue
            for ev in _cat:
                if len(ev.picks) == 0:
                    Logger.warning(
                        f"No picks for event: {ev.resource_id.id}, not relocating")
                    continue
                new_events.append(SparseEvent.from_event(ev))
                self._all_event_files.update({ev.resource_id.id: f})

        Logger.info(f"Computing correlations for {len(new_events)} events")
        written_links = self.correlator.add_events(
            catalog=new_events, max_workers=workers)

        if written_links > 0:
            self.run_growclust(
                outdir=outdir, station_file=station_file,
                event_files=self._all_event_files, workers=workers,
                config=internal_config, vmodel_file=vmodel_file,
                growclust_script=growclust_script, cleanup=cleanup)
        else:
            Logger.info("No new links written, not running growclust")

        return list(new_files)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
