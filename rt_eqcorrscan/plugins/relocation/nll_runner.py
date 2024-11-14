"""
Code to run NonLinLoc as a plugin for RTEQcorrscan
"""

import subprocess
import os
import glob
import tqdm
import shutil
import tempfile
import platform
import logging

from math import radians, sin, cos

from typing import Union, Iterable, List

from pyproj import CRS, Transformer

from obspy.core.event import Event
from obspy import read_events, Catalog, Inventory, read_inventory

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import _Plugin, PLUGIN_CONFIG_MAPPER
from rt_eqcorrscan.plugins.relocation.helpers.extract_vmodel import (
    RotatedTransformer, _round_up, _round_down)

MINDEPTH, MAXDEPTH = -3.0, 500.0
PICK_UNCERTAINTY = 0.2


Logger = logging.getLogger(__name__)


# NonLinLoc configuration

class NLLConfig(_PluginConfig):
    defaults = {
        "infile": os.path.abspath("nonlinloc_files/NonLinLoc_3d.in"),
        "veldir": os.path.abspath("nonlinloc_files/VEL"),
        "maxlat": None,
        "maxlon": None,
        "minlat": None,
        "minlon": None,
        "maxdepth": None,  # All to be set by a call from the plugin master.
        "mindepth": -3.0,
        "nodespacing": 1.0,
        "station_file": "stations.xml",
        "sleep_interval": 10,
    }
    _readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infile = os.path.abspath(self.infile)
        self.veldir = os.path.abspath(self.veldir)


PLUGIN_CONFIG_MAPPER.update({"nll": NLLConfig})


def _get_transform(origin_line: str) -> Union[Transformer, RotatedTransformer]:
    if len(origin_line.split()) != 6:
        raise NotImplementedError(f"Don't know how to cope with {origin_line}")
    _, proj, ellipsoid, ori_lat, ori_lon, ori_rotation = origin_line.split()
    ori_lat, ori_lon, ori_rotation = map(
        float, (ori_lat, ori_lon, ori_rotation))

    if proj == "TRANS_MERC":
        proj = "tmerc"
    else:
        raise NotImplementedError(f"Unknown projection: {proj}")
    if ellipsoid == "GRS-80":
        ellipsoid = "GRS80"
    else:
        raise NotImplementedError(f"Unknown ellipsoid: {ellipsoid}")
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_proj4(f"+proj={proj} +lat_0={ori_lat} +lon_0={ori_lon} "
                       f"+ellps={ellipsoid}"))
    if ori_rotation == 0:
        return transformer
    return RotatedTransformer(transformer=transformer,
                              rotation=ori_rotation)

#  IO

def inv_to_nll(
        inv: Inventory
) -> str:
    """
    Convert an obspy inventory to a nll formatted string of station locations.
    """
    lats, lons, elevs, depths = dict(), dict(), dict(), dict()
    for net in inv:
        for sta in net:
            for chan in sta:
                depth = chan.depth / 1000.0
                elev = chan.elevation / 1000.0
                lat, lon = chan.latitude, chan.longitude

                # Check values are consistent
                if depths.get(sta.code) not in [None, depth]:
                    print(
                        f"{depth} for {sta.code} is changed from {depths.get(sta.code)}")
                    choice = input(
                        f"Use old depth ({depths.get(sta.code)}) or new depth ({depth}) (o/n)?")
                    assert choice in "on", "Need to chose [o]ld or [n]ew"
                    if choice == "o":
                        depth = depths.get(sta.code)

                assert elevs.get(sta.code) in [None,
                                               elev], f"{elev} for {sta.code} is changed from {elevs.get(sta.code)}"
                assert lats.get(sta.code) in [None,
                                              lat], f"{lat} for {sta.code} is changed from {lats.get(sta.code)}"
                assert lons.get(sta.code) in [None,
                                              lon], f"{lon} for {sta.code} is changed from {lons.get(sta.code)}"

                depths[sta.code] = depth
                elevs[sta.code] = elev
                lats[sta.code] = lat
                lons[sta.code] = lon

    lines = [
        f"GTSRCE {sta.code:5s} LATLON {lats[sta.code]:7.4f}  "
        f"{lons[sta.code]:7.4f} {depths[sta.code]:02.1f} "
        f"{elevs[sta.code]:.3f}"
        for net in inv for sta in net]

    return "\n".join(lines)


def setup_nll(
        nlldir: str,
        veldir: str,
        inv: Inventory,
        control_file: str,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        min_depth: float = -3.0,
        max_depth: float = 300.0,
        node_spacing: float = 1.0,
        verbose: bool = True,
):
    """
    Set up the necessary directories and run Vel2Grid and Grid2Time.

    Parameters
    ----------
    nlldir:
        Directory to work in for NLL - does not have to exist
    inv:
        Inventory of stations to be used in NLL
    min_lat:
        Minimum latitude for box to locate events in.
    min_lon:
        Minimum longitude for box to locate events in.
    max_lat:
        Maximum latitude for box to locate events in.
    max_lon:
        Maximum longitude for box to locate events in.
    min_depth:
        Minimum depth for box to locate events in (km).
    max_depth:
        Maximum depth for box to locate events in (km).
    node_spacing:
        Node spacing for vmodel in km
    """
    if not os.path.isdir(nlldir):
        os.makedirs(nlldir)

    cwd = os.getcwd()

    # Read config file
    with open(control_file, "r") as f:
        control_lines = f.read().splitlines()

    station_file = "stations.nll"

    os.chdir(nlldir)
    # Write station file
    for n in inv:
        for s in n:
            if len(s) == 0:
                Logger.warning(
                    f"{s.code} has no channels and will not be used.")
            for c in s:
                if c.elevation > 3000.0:
                    Logger.warning(f"Non-sensical elevation ({c.elevation}), "
                                   f"setting to 0 for {s.code}")
                    c.elevation = 0.0
    nll_stations = inv_to_nll(inv=inv)
    with open(station_file, "w") as f:
        f.write(nll_stations)

    # Check box
    sta_lats = [c.latitude for n in inv for s in n for c in s]
    sta_lons = [c.longitude for n in inv for s in n for c in s]
    sta_depths = [(c.depth / - 1000) - (c.elevation / 1000) for n in inv for s
                  in n for c in s]

    min_depth = min(min_depth or MINDEPTH, min(sta_depths))
    max_depth = max(max_depth or MAXDEPTH, max(sta_depths))

    # Get origin of grid from control file
    origin_line = [l for l in control_lines if l.startswith("TRANS")]
    assert len(origin_line) == 1, "Missing or duplicated TRANS line."
    transformer = _get_transform(origin_line=origin_line[0])

    sta_x, sta_y = [], []
    for lat, lon in zip(sta_lats, sta_lons):
        x, y = transformer.transform(lat, lon)
        sta_x.append(x)
        sta_y.append(y)

    if min_lat is not None and min_lon is not None:
        min_x, min_y = transformer.transform(min_lat, min_lon)
        min_x = min(min(sta_x), min_x)
        min_y = min(min(sta_y), min_y)
    else:
        min_x, min_y = min(sta_x), min(sta_y)
    if max_lat is not None and max_lon is not None:
        max_x, max_y = transformer.transform(max_lat, max_lon)
        max_x = max(max(sta_x), max_x)
        max_y = max(max(sta_y), max_y)
    else:
        max_x, max_y = max(sta_x), max(sta_y)

    min_x /= 1000.
    max_x /= 1000.
    min_y /= 1000.
    max_y /= 1000.

    # Note Donna's models use a r-hand (+ve E) coord system, NLL uses
    # l-hand (+ve W), both +ve down and +ve N
    # numbers from transformer are +ve E
    min_x, min_y = map(_round_down, (min_x, min_y))
    max_x, max_y = map(_round_up, (max_x, max_y))
    min_depth, max_depth = _round_down(min_depth, 1), _round_up(max_depth, 10)
    delta_x = abs(max_x - min_x)
    delta_y = abs(max_y - min_y)
    delta_z = max_depth - min_depth

    xnum = int(delta_x / node_spacing)
    ynum = int(delta_y / node_spacing)
    znum = int(delta_z / node_spacing)

    vggrid_line = (
        f"VGGRID  {xnum} {ynum} {znum} {min_x:.1f} {min_y:.1f} {min_depth} "
        f"{node_spacing:.1f} {node_spacing:.1f} {node_spacing:.1f} SLOW_LEN"
    )

    locgrid_line = (
        f"LOCGRID  {xnum} {ynum} {znum} {min_x:.1f} {min_y:.1f} {min_depth} "
        f"{node_spacing:.1f} {node_spacing:.1f} {node_spacing:.1f} "
        "PROB_DENSITY SAVE"
    )

    # Get VGTYPE and VGINP lines
    vg_lines = [l for l in control_lines if "VGINP" in l or "VGTYPE" in l]
    # Assume that the VGINP that follows a VGTYPE is the correct file for that wave type
    p_vg_lines, s_vg_lines, pre_line = [], [], None
    for l in vg_lines:
        l = l.lstrip("#").lstrip()
        if l.split() == ["VGTYPE", "P"]:
            p_vg_lines.append(l)
        elif pre_line and pre_line.split() == ["VGTYPE", "P"] and "VGINP" in l:
            p_vg_lines.append(l)
        elif l.split() == ["VGTYPE", "S"]:
            s_vg_lines.append(l)
        elif pre_line and pre_line.split() == ["VGTYPE", "S"] and "VGINP" in l:
            s_vg_lines.append(l)
        pre_line = l

    gtline = [l for l in control_lines if l.startswith("GTFILES")][0]
    gt_parts = gtline.split()
    gtline = " ".join(gt_parts[0:-1])
    gts = gtline + " S"
    gtp = gtline + " P"

    other_lines = [l for l in control_lines if not "VGGRID" in l
                   and not "LOCGRID" in l and not "VGTYPE" in l
                   and not "VGINP" in l and not l.startswith("#") and len(l)
                   and not "GTFILES" in l and not "INCLUDE" in l]

    # Make requisite directories
    for _dir in ["TIME", "VEL", "IN", "OUT"]:
        os.makedirs(_dir, exist_ok=True)

    for vg_lines, _gtline in zip([p_vg_lines, s_vg_lines], [gtp, gts]):
        lines_out = other_lines + vg_lines
        lines_out.append(vggrid_line)
        lines_out.append(locgrid_line)
        lines_out.append(f"INCLUDE {station_file}")
        lines_out.append(_gtline)
        with open(os.path.split(control_file)[-1], "w") as f:
            f.write("\n".join(lines_out))

        # Copy velocity files from originals
        v_file = vg_lines[-1].split()[1]
        shutil.copyfile(f"{veldir}/{v_file}", v_file)

        for cmd in ["Vel2Grid3D", "Grid2Time"]:
            kwargs = dict(bufsize=0, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
            args = [cmd, os.path.split(control_file)[-1]]
            print(f"Running {' '.join(args)} from {nlldir}")

            with subprocess.Popen(args, **kwargs) as p:
                _output, _err = [], []
                for line in p.stdout:
                    _out = line.decode('utf-8')
                    if verbose:
                        print(_out, end='', flush=True)
                    _output.append(_out)
                for line in p.stderr:
                    _err.append(line.decode('utf-8'))
                if p.returncode != 0:
                    _err = "\n".join(_err)
                    print(f"Could not run {cmd}: \n{_err}")
                print(f"{cmd} successful")

    os.chdir(cwd)
    return


# Runners

def run_nll(catalog: Catalog, control_file: str, verbose: bool = True) -> Catalog:
    """
    Locate all events in a catalog using NonLinLoc.

    Events located will have origins added to the origins for the event
    as located by NonLinLoc.
    """
    cat_back, input_files = dict(), list()
    # Write out all events
    for event in catalog:
        event_back = event.copy()
        # Weight the picks based on correlation
        for pick in event_back.picks:
            correlation_comment = [c for c in pick.comments if "cc_max=" in c.text]
            if len(correlation_comment) == 0:
                continue
            correlation = float(correlation_comment[0].text.split("cc_max=")[-1])
            # TODO: Is setting uncertainty in this way appropriate? Should
            # it just be a fixed value? Correlation doesn't really relate to
            # time uncertainty directly...
            # pick.time_errors.uncertainty = 1 - correlation
            # Testing with loggau 0.2 40 and locgau2 0.07 0.2 4.0 showed that
            # setting time uncertainty to 0.2 seconds gave good results
            pick.time_errors.uncertainty = PICK_UNCERTAINTY
        rid = event_back.resource_id.id.split('/')[-1]
        event_back.write(f"IN/{rid}.nll",
                         format="NLLOC_OBS")
        Logger.info(f"Written event {rid} to IN/{rid}.nll")
        cat_back.update({rid: event_back})
        input_files.append(f"IN/{rid}.nll")
    # Run NonLinLoc
    args = ["NLLoc", os.path.split(control_file)[-1]]

    Logger.info(f"Running call {' '.join(args)}")
    loc_proc = subprocess.run(args)

    try:
        loc_proc.check_returncode()
    except Exception as e:
        Logger.exception(e)
        return Catalog()

    Logger.info("LOCATION SUCCESSFUL")

    nll_cat = read_events("OUT/located.*.*.grid0.loc.hyp")
    Logger.info(f"Read in {len(nll_cat)} located events")
    # Clean-up
    Logger.info("Removing temporary NLL files")
    for file in tqdm.tqdm(glob.glob("OUT/*")):
        os.remove(file)
    for file in tqdm.tqdm(input_files):
        os.remove(file)

    nll_located_rids = set()
    for event in nll_cat:
        rid = event.creation_info.author.split('obs:')[-1].split()[0].lstrip("IN/").rstrip(".nll")
        nll_located_rids.update({rid})
        event_back = cat_back.get(rid, None)
        if event_back is None:
            Logger.warning("Event {rid} found not in input catalog")
            continue
        nlloc_origin = event.origins[0]
        # TODO: Check origin quality - if on edge of grid, warn
        for arrival in nlloc_origin.arrivals:
            nlloc_pick = arrival.pick_id.get_referred_object()
            original_pick = [
                p for p in event_back.picks
                if p.waveform_id.station_code == nlloc_pick.waveform_id.station_code
                and (p.phase_hint or "?")[0] == nlloc_pick.phase_hint[0]
                and abs(p.time - nlloc_pick.time) < 0.1]
            if len(original_pick) > 1:
                original_pick.sort(key=lambda p: abs(p.time - nlloc_pick.time))
            arrival.pick_id = original_pick[0].resource_id
        event_back.origins.append(nlloc_origin)
        event_back.preferred_origin_id = nlloc_origin.resource_id
    unlocated_events = set(cat_back.keys()).difference(nll_located_rids)
    if len(unlocated_events):
        print(f"{len(unlocated_events)} were not located by NLL:\n{unlocated_events}")
    return Catalog(list(cat_back.values()))


class NLL(_Plugin):
    working_dir = "nll_working"
    _setup = True
    def __init__(self, config_file: str, name: str = "NLLRunner"):
        super().__init__(config_file=config_file, name=name)

    def _read_config(self, config_file: str):
        return NLLConfig.read(config_file)

    def _cleanup(self):
        if os.path.isdir(self.working_dir):
            shutil.rmtree(self.working_dir)

    def run_nll(self, catalog: Catalog, control_file: str,
                verbose: bool = True) -> Catalog:
        cwd = os.getcwd()
        os.chdir(self.working_dir)

        cat_back = Catalog()
        try:
            cat_back = run_nll(catalog, control_file=control_file,
                               verbose=verbose)
        except Exception as e:
            Logger.error(f"Location failed due to {e}")
        finally:
            os.chdir(cwd)
        return cat_back


    def _get_bounds(self):
        internal_config = self.config.copy()
        min_lat = internal_config.pop("minlat", None)
        max_lat = internal_config.pop("maxlat", None)
        min_lon = internal_config.pop("minlon", None)
        max_lon = internal_config.pop("maxlon", None)
        maxdepth = internal_config.pop("maxdepth", None)
        nodespacing = internal_config.pop("nodespacing", 1.0)
        mindepth = internal_config.pop("mindepth", -3.0)
        return min_lat, max_lat, min_lon, max_lon, mindepth, maxdepth, nodespacing

    def setup(self):
        min_lat, max_lat, min_lon, max_lon, mindepth, maxdepth, nodespacing = (
            self._get_bounds())
        inv = read_inventory(self.config.station_file)
        setup_nll(nlldir=self.working_dir, inv=inv, min_lat=min_lat,
                  min_lon=min_lon, max_lat=max_lat, max_lon=max_lon,
                  max_depth=maxdepth, node_spacing=nodespacing,
                  min_depth=mindepth, verbose=True,
                  control_file=self.config.infile,
                  veldir=self.config.veldir)
        self._setup = False
        return

    def core(self, new_files: Iterable, cleanup: bool = False) -> List:
        internal_config = self.config.copy()
        in_dir = internal_config.pop("in_dir")
        out_dir = internal_config.pop("out_dir")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        station_file = internal_config.pop("station_file")
        min_lat, max_lat, min_lon, max_lon, mindepth, maxdepth, nodespacing = (
            self._get_bounds())
        inv = read_inventory(station_file)
        Logger.info(
            f"Read in inventory of {len([s for n in inv for s in n])} "
            f"stations")

        cat_to_locate, event_file_mapper = Catalog(), dict()
        Logger.info("Reading events")
        for f in tqdm.tqdm(new_files):
            cat = read_events(f)
            cat_to_locate += cat
            # Cope with possibility of multiple events in one file.
            event_file_mapper.update({f: [ev.resource_id.id.split('/')[-1]
                                          for ev in cat]})

        lats, lons, depths = [], [], []
        for ev in cat_to_locate:
            if len(ev.origins):
                lat = ev.origins[-1].get("latitude", None)
                lon = ev.origins[-1].get("longitude", None)
                depth = ev.origins[-1].get("depth", None)
                if lat is not None:
                    lats.append(lat)
                if lon is not None:
                    lons.append(lon)
                if depth is not None:
                    depths.append(depth / 1000.0)

        if min_lat is None and len(lats):
            min_lat = min(lats)
        if max_lat is None and len(lats):
            max_lat = max(lats)
        if min_lon is None and len(lons):
            min_lon = min(lons)
        if max_lon is None and len(lons):
            max_lon = max(lons)
        if maxdepth is None and len(depths):
            maxdepth = max(depths)

        if self._setup:
            Logger.info("Running setup")
            setup_nll(nlldir=self.working_dir, inv=inv, min_lat=min_lat,
                      min_lon=min_lon, max_lat=max_lat, max_lon=max_lon,
                      max_depth=maxdepth, node_spacing=nodespacing,
                      min_depth=mindepth, verbose=True,
                      control_file=internal_config.infile,
                      veldir=internal_config.veldir)
            # Run the setup only once.
            self._setup = False

        Logger.info("Running locations")
        cat_located = self.run_nll(catalog=cat_to_locate,
                                   control_file=internal_config.infile)
        Logger.info("Locations returned")
        cat_located_dict = {ev.resource_id.id.split('/')[-1]: ev
                            for ev in cat_located}
        for f, eids in event_file_mapper.items():
            subcat = Catalog()
            for eid in eids:
                event = cat_located_dict.get(eid, None)
                if event is not None:
                    subcat += event
                else:
                    Logger.warning(f"Did not find {eid} in located output")
                    Logger.warning(f"Known keys:\n\t{cat_located_dict.keys()}")
            if len(subcat):
                fname = os.path.split(f)[-1]
                fname = fname.lstrip(os.path.sep)
                outpath = os.path.join(out_dir, fname)
                Logger.info(f"Writing {len(subcat)} events to {outpath}")
                subcat.write(outpath, format="QUAKEML")

        if cleanup:
            self._cleanup()

        return new_files


if __name__ == "__main__":
    import doctest

    doctest.testmod()
