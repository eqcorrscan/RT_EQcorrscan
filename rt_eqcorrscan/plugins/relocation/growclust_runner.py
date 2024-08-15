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
import glob
import math
import os
import tempfile

import numpy as np

from scipy.stats import circmean

from typing import Iterable

from obspy import (
    read_events, Catalog, UTCDateTime, read_inventory, Inventory)
from obspy.core.event import (
    Origin, ResourceIdentifier, OriginUncertainty, OriginQuality)

from eqcorrscan.utils.catalog_to_dd import write_correlations, write_phase

from rt_eqcorrscan.plugins.waveform_access import InMemoryWaveBank
from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.relocation.hyp_runner import VelocityModel

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
GROWCLUST_DEFAULTS = f"{WORKING_DIR}/growclust.inp"
GROWCLUST_DEFAULT_VMODEL = f"{WORKING_DIR}/vmodel.txt"
GROWCLUST_SCRIPT = f"{WORKING_DIR}/run_growclust3D.jl"

_GC_TMP_FILES = []  # List of temporary files to remove

Logger = logging.getLogger(__name__)

# CORRELATION COMPUTATION PARAMETERS
XCORR_PARAMS = {
    "extract_len": 2.0,
    "pre_pick": 0.2,
    "shift_len": 0.2,
    "lowcut": 1.0,
    "highcut": 20.0,
    "max_sep": 8,
    "min_link": 8,
    "min_cc": 0.2,
    "interpolate": False,
    "all_horiz": False,
    "max_workers": None,
    "parallel_process": False
}


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
        "fin_vzmld": "vzmodel.txt",
        "fdir_ttab": "tt",
        "projection": "tmerc WGS84 0.0 0.0 0.0",
        "vpvs_factor": 1.732,
        "rayparam_min": 0.0,
        "tt_zmin": -4.0,
        "tt_zmax": 500.0,
        "tt_zstep": 1.0,
        "tt_xmin": 0.0,
        "tt_xmax": 1000.0,
        "tt_xstep": 1.0,
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
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_growclust(self, filename: str = "growclust.inp"):
        """
        Write the input file for growclust.

        The growclust file is fixed lines with comments between.
        """
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
            self.scordat_fmt,
            "* fin_xcordat",
            self.fin_xcordat,
            "*",
            "* ttabsrc: travel time table source ('trace' or 'nllgrid')",
            self.ttabsrc,
            "* fin_vzmdl (model name)",
            self.fin_vzmdl,
            "* fdir_ttab (directory for travel time tables/grids or NONE)",
            self.fdit_ttab,
            "* projection (proj, ellps, lon0, lat0, rotANG, [latP1, latP2])",
            self.projection,
            "* vpvs_factor  rayparam_min",
            f"{self.vpvs_factor}         {self.rayparam_min}",
            "* tt_zmin  tt_zmax  tt_zstep",
            f"{self.tt_zmin}   {self.tt_zmax}  {self.tt_zstep}",
            "* tt_xmin  tt_xmax  tt_xstep",
            f"{self.tt_xmin}  {self.tt_xmax}  {self.tt_xstep}",
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


def run_growclust(
    catalog: Catalog,
    workers: [int, str] = "auto",
    control_file: str = GROWCLUST_DEFAULTS,
    vmodel_file: str = GROWCLUST_DEFAULT_VMODEL,
    growclust_script: str = GROWCLUST_SCRIPT,
) -> str:
    """
    Requires a growclust julia runner script.
    """
    import subprocess
    
    # Need to edit the projection origin for the events
    lats = np.array([(ev.preferred_origin() or ev.origins[-1]).latitude
                     for ev in catalog])
    lons = np.array([(ev.preferred_origin() or ev.origins[-1]).longitude
                     for ev in catalog])

    mean_lat = np.mean(lats)
    mean_lon = np.degrees(circmean(np.radians(lons)))

    with open(control_file, "r") as f:
        control_lines = f.read().splitlines()

    # Growclust uses fixed line formats - the line we need to edit is line 9
    counter, outlines = 0, []
    for line in control_lines:
        if line.startswith("*"):
            continue
        if counter == 9:
            parts = line.split()
            parts[2] = str(mean_lon)
            parts[3] = str(mean_lat)
            _line = " ".join(parts)
        else:
            _line = line
        counter += 1
        outlines.append(_line)

    # Line 16 is output file
    outfile = outlines[16]

    with open(f"{WORKING_DIR}/.growclust_control.inp", "w") as f:
        f.write("\n".join(outlines))

    vmodel = VelocityModel.read(vmodel_file)
    vmodel.write("./vzmodel.txt", format="GROWCLUST")

    # TODO: Re-write Julia code to only do ray-tracing once?
    # TODO: If caching travel-times then the lat and lon or the origin of the coord system must be preserved - global mutable variables?

    # TODO: Use a variable for working dir - we make lots of temp files

    loc_proc = subprocess.run([
        "julia",
        "--threads", workers,
        growclust_script,
        "-c", f"{WORKING_DIR}/.growclust_control.inp"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    for line in loc_proc.stdout.decode().splitlines():
        Logger.info(">>> " + line.rstrip())
    loc_proc.check_returncode()
    return outfile


def write_stations(
    seed_ids: Iterable,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    indir: str,
):
    import fnmatch

    inv = read_inventory(f"{indir}/stations.xml")
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


def process_input_catalog(catalog: Catalog, wavedir: str, indir: str) -> dict:
    """ Compute inter-event cross-correlations for growclust. """
    in_memory_wavebank = InMemoryWaveBank(wavedir=wavedir)

    # This could be threaded to allow parallel IO?
    stream_dict = {
        event.resource_id.id: in_memory_wavebank.get_event_waveforms(
            event=event, pre_pick=XCORR_PARAMS["pre_pick"] + 1,
            length=XCORR_PARAMS["extract_len"] + 2)
        for event in catalog
    }

    event_mapper = write_correlations(
        catalog=catalog,
        stream_dict=stream_dict,
        event_id_mapper=None,
        **XCORR_PARAMS)
    # TODO: Writing event format incorrect
    event_mapper = write_phase(catalog, event_id_mapper=event_mapper)
    phase_to_evlist("phase.dat")
    # Write station file
    seed_ids = {tr.id for st in stream_dict.values() for tr in st}
    starttime = min(tr.stats.starttime for st in stream_dict.values()
                    for tr in st)
    endtime = max(tr.stats.endtime for st in stream_dict.values()
                  for tr in st)
    write_stations(seed_ids=seed_ids, starttime=starttime, endtime=endtime,
                   indir=indir)
    return event_mapper


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
    gc_origins = read_growclust(
        outfile, event_mapper={val: key for key, val in event_mapper.items()})

    catalog_dict = {ev.resource_id.id: ev for ev in catalog}

    catalog_out, relocated, not_relocated = Catalog(), Catalog(), Catalog()
    for key, ev in catalog_dict.items():
        gc_origin = gc_origins.get(key)
        if gc_origin is None:
            Logger.warning(f"Event {key} did not relocate")
            not_relocated += ev
            catalog_out += ev
            continue
        ev.origins.append(gc_origin)
        ev.preferred_origin_id = ev.origins[-1].resource_id
        relocated += ev
        catalog_out += ev
    Logger.info(f"Of {len(catalog_out)}, {len(relocated)} were relocated, and "
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


def growclust_line_to_origin(line: str) -> [str, Origin]:
    line = line.split()
    deserialized = {key: val[1](line[val[0]]) for key, val in FORMATTER.items()}
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
    return deserialized["eventid"], origin


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
    Dictionary of origins keyed by event id.
    """
    with open(fname, "r") as f:
        lines = f.read().splitlines()

    origins = dict()
    for line in lines:
        event_id, growclust_origin = growclust_line_to_origin(line)
        if event_mapper:
            event_id = event_mapper.get(event_id, f"{event_id}_notmapped")
        origins.update({event_id: growclust_origin})
    return origins


def main(indir: str, outdir: str, wavedir: str):
    event_files = glob.glob(f"{indir}/*")
    catalog = Catalog()
    for event_file in event_files:
        if event_file.split('/')[-1] == "stations.xml":
            # Skip the inventory
            continue
        try:
            catalog += read_events(event_file)
        except Exception as e:
            Logger.error(f"Could not read from {event_file} due to {e}")
    Logger.info(f"Read in {len(catalog)} events to relocate")
    # Do the mahi in a tempdir
    working_dir = tempfile.TemporaryDirectory()
    cwd = os.path.abspath(os.path.curdir)
    indir, outdir, wavedir = map(os.path.abspath, (indir, outdir, wavedir))
    os.chdir(working_dir.name)
    event_mapper = process_input_catalog(
        catalog=catalog, wavedir=wavedir, indir=indir)
    outfile = run_growclust(catalog=catalog)
    catalog_out = process_output_catalog(
        catalog=catalog, event_mapper=event_mapper, outfile=outfile)
    os.chdir(cwd)
    working_dir.cleanup()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    catalog_out.write(f"{outdir}/relocated.xml", format="QUAKEML")


def _cleanup():
    # Clean up
    for f in _GC_TMP_FILES:
        if os.path.isfile(f):
            os.remove(f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Relocate events using GrowClust")

    parser.add_argument(
        "-i", "--indir", type=str, required=True,
        help="Input directory to read events from")
    parser.add_argument(
        "-w", "--wavedir", type=str, required=True,
        help="Input directory for waveforms.")
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory for events")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Increase verbosity")

    args = parser.parse_args()

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG

    logging.basicConfig(
        level=level, format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s"
    )

    main(indir=args.indir, outdir=args.outdir, wavedir=args.wavedir)
