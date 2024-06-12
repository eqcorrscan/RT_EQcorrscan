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
import datetime as dt

from collections import namedtuple
from typing import Iterable

from obspy import (
    read_events, Catalog, UTCDateTime, read, Stream, read_inventory, Inventory)
from obspy.core.event import (
    Event, Origin, Magnitude, ResourceIdentifier, OriginUncertainty,
    OriginQuality)

from eqcorrscan.utils.catalog_to_dd import write_correlations, write_phase

from rt_eqcorrscan.config.config import _ConfigAttribDict

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
GROWCLUST_DEFAULTS = f"{WORKING_DIR}/growclust.inp"

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


class GrowClustConfig(_ConfigAttribDict):
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


def run_growclust(workers: int = 1, control_file: str = GROWCLUST_DEFAULTS):
    """
    Requires a growclust julia runner script.
    """
    import subprocess

    subprocess.run([
        "julia",
        f"-p{workers}",
        f"{WORKING_DIR}/run_growclust.jl",
        control_file])
    return


def _get_event_waveforms(
    event: Event,
    data_availability: dict,
    pre_pick: float,
    length: float,
    phases: set = {"P", "Pn", "Pg", "Pb", "S", "Sn", "Sg", "Sb"},
) -> Stream:
    # convenience pick named tuple
    SparsePick = namedtuple("SparsePick", ["seed_id", "time", "files"])
    # Filter on phase hint
    used_picks = [
        SparsePick(pick.waveform_id.get_seed_string(), pick.time.datetime, None)
        for pick in event.picks if pick.phase_hint in phases]
    # Filter on availability
    used_picks = [p for p in used_picks if p.seed_id in data_availability.keys()]

    # Get relevant files
    for i, pick in enumerate(used_picks):
        seed_availability = data_availability[pick.seed_id]
        tr_start, tr_end = (
            pick.time - dt.timedelta(seconds=pre_pick),
            (pick.time - dt.timedelta(seconds=pre_pick)) + dt.timedelta(seconds=length))
        files = {f for f in seed_availability if f.starttime < tr_start < f.endtime}  # Starts within file
        files.update({f for f in seed_availability if f.starttime < tr_end < f.endtime})  # Ends within file
        used_picks[i] = pick._replace(files=files)

    # Filter picks without useful times available
    used_picks = [p for p in used_picks if p.files]

    # Read in waveforms
    st = Stream()
    for pick in used_picks:
        tr_start, tr_end = (
            pick.time - dt.timedelta(seconds=pre_pick),
            (pick.time - dt.timedelta(seconds=pre_pick)) + dt.timedelta(seconds=length))
        for file in pick.files:
            st += read(file.filename, starttime=UTCDateTime(tr_start), endtime=UTCDateTime(tr_end))
    return st


def _get_data_availability(wavedir: str) -> dict:
    """ Scan a waveform dir and work out what is in it. """
    # Convenience file info
    FileInfo = namedtuple("FileInfo", ["filename", "seed_id", "starttime", "endtime"])
    data_availability = dict()
    for root, dirs, files in os.walk(wavedir):
        for f in files:
            filepath = os.path.join(root, f)
            st = None
            try:  # Try to just read the header
                st = read(filepath, headonly=True)
            except Exception as e:
                Logger.debug(f"Could not read headonly for {f} due to {e}")
                try:
                    st = read(filepath)
                except Exception as e2:
                    Logger.debug(f"Could not read {f} due to {e2}")
            if st is None:
                continue
            for tr in st:
                seed_availability = data_availability.get(tr.id, [])
                seed_availability.append(
                    FileInfo(
                        filepath,
                        tr.id,
                        tr.stats.starttime.datetime,  # these need to be datetimes to be hashable
                        tr.stats.endtime.datetime
                    ))
                data_availability.update({tr.id: seed_availability})

    return data_availability


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
    # Construct stream dict first
    data_availability = _get_data_availability(wavedir=wavedir)

    # This could be threaded to allow parallel IO?
    stream_dict = {
        event.resource_id.id: _get_event_waveforms(
            event=event, data_availability=data_availability,
            pre_pick=XCORR_PARAMS["pre_pick"] + 1,
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
    starttime = min(tr.stats.starttime for st in stream_dict.values() for tr in st)
    endtime = max(tr.stats.endtime for st in stream_dict.values() for tr in st)
    write_stations(seed_ids=seed_ids, starttime=starttime, endtime=endtime, indir=indir)
    return event_mapper


def phase_to_evlist(input_file: str):
    with open(input_file, "r") as f:
        lines = [l for l in f.read().splitlines() if l.startswith("#")]
    out_lines = [l.lstrip("#").lstrip() for l in lines]
    with open("evlist.txt", "w") as f:
        for line in out_lines:
            f.write(f"{line}\n")
    return


def process_output_catalog(catalog: Catalog, event_mapper: dict) -> Catalog:
    """ Add Growclust origins back into catalog. """
    catalog_out = catalog.copy()
    growclust_origins = read_growclust(
        "/tmp/growclust_out/out.nboot100.cat",
        event_mapper={val: key for key, val in event_mapper.items()})
    for ev in catalog_out:
        growclust_origin = growclust_origins.get(ev.resource_id.id, None)
        if growclust_origin is None:
            Logger.error(f"No relocation for {ev.resource_id.id}")
            continue
        ev.origins.append(growclust_origin)
        ev.preferred_origin_id = growclust_origin.resource_id

    return catalog_out


########## GROWCLUST READING ############################

"""
2009  1 18  0 49 21.000         1 -42.23833  173.79283  19.900  2.85       1   12359       1     0     0     0  0.00  0.00  -1.000  -1.000  -1.000   -42.23833  173.79283  19.900
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
        depth_errors={"uncertainty": deserialized["ez"] * 1000.0 if deserialized["ez"] else None},
        time_fixed=False,
        origin_uncertainty=OriginUncertainty(
            horizontal_uncertainty=deserialized["eh"] * 1000.0 if deserialized["eh"] else None),
        quality=OriginQuality(
            used_phase_count=deserialized["qndiffP"] + deserialized["qndiffS"],
            standard_error=(
                deserialized["qndiffP"] + deserialized["qndiffS"]) *
                (p_standard_error + s_standard_error)))
    return deserialized["eventid"], origin


def read_growclust(
    fname: str = "OUT/out.growclust_cat",
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
    event_mapper = process_input_catalog(catalog=catalog, wavedir=wavedir, indir=indir)
    run_growclust()
    catalog_out = process_output_catalog(catalog=catalog, event_mapper=event_mapper)
    catalog_out.write(f"{outdir}/relocated.xml", format="QUAKEML")


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
        "-v", "--verbose", action="store_true", help="Increase verbosity")

    args = parser.parse_args()

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG

    logging.basicConfig(
        level=level, format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s"
    )

    main(indir=args.indir, outdir=args.outdir, wavedir=args.wavedir)
