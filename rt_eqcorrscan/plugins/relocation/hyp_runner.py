"""
Locate events using hyp.

"""

import os
import glob
import logging
import warnings
import subprocess

from typing import List

from obspy import read_events, Catalog, read_inventory, UTCDateTime
from obspy.core.event import Event, Origin
from obspy.core.inventory import Inventory, Station
from obspy.io.nordic.core import write_select, read_nordic


Logger = logging.getLogger(__name__)

############################## VELOCITY MODEL CLASSES #################################

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
        return f"VelocityModel(<{len(self.velocities)} layers>, ..., vpvs={self.vpvs})"

    def __str__(self):
        lines = ["VelocityModel"]
        for v in self.velocities:
            lines.append(f"{v.top},{v.vp},{v.moho}")
        lines.append(f"vpvs: {self.vpvs}")
        return "\n".join(lines)


    def write(self, filename: str):
        with open(filename, "w") as f:
            f.write(self.__str__())

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


###################################### END OF VELOCITY MODELS #######################

################################## SEISAN HYP CALLS #################################

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
        subprocess.call(['remodl'])
        subprocess.call(['setbrn'])

    event_out = event.copy()
    try:
        old_origin = event.preferred_origin() or event.origins[0]
        origin = Origin(time=old_origin.time)
    except IndexError:
        origin = Origin(
            time=sorted(event.picks, key=lambda p: p.time)[0].time)
    event_out.origins = [origin]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        event_out.write(format="NORDIC", filename="to_be_located")
    subprocess.call(['hyp', "to_be_located"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            event_back = read_nordic("hyp.out")
        except Exception as e:
            Logger.error(e)
            return None
    # We lose some info in the round-trip to nordic
    event_out.origins[0] = event_back[0].origins[0]
    event_out.magnitudes = event_back[0].magnitudes
    event_out.picks = event_back[0].picks
    if clean:
        _cleanup()
    return event_out


def _cleanup():
    # Clean up
    files_to_remove = [
        "hyp.out", "to_be_located", "remodl.tbl", "remodl1.lis", "remodl2.lis",
        "print.out", "gmap.cur.kml", "hypmag.out", "hypsum.out", "remodl.hed",
        "IASP91_linux.HED", "IASP91_linux.TBL", "setbrn1.lis", "setbrn2.lis",
        "setbrn3.lis", "STATION0.HYP", "focmec.dat", "focmec.inp", "fort.17",
        "fps.out", "hash_seisan.out", "pspolar.inp", "scratch1.out",
        "scratch2.out", "scratch3.out"]
    for f in files_to_remove:
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


################################## END OF SEISAN HYP CALLS ##########################


################################## CONTROL FUNCS ####################################

def _get_bulk_for_cat(cat: Catalog) -> List[tuple]:
    picks = sorted([p for ev in cat for p in ev.picks], key=lambda p: p.time)
    starttime = picks[0].time
    endtime = picks[-1].time
    bulk = {
        (p.waveform_id.network_code or "*",
         p.waveform_id.station_code or "*",
         p.waveform_id.location_code or "*",
         p.waveform_id.channel_code or "*")
        for p in picks}
    bulk = [(n, s, l, c, starttime, endtime) for n, s, l, c in bulk]
    return bulk


def setup_testcase(
    indir: str,
    stationxml: str,
):
    """
    Download some event files and a stationxml to run.
    """
    from obspy.clients.fdsn import Client

    client = Client("GEONET")

    Logger.info("Setting up testcase")
    # Kaikōura - Cape Campbell ~ 200 events
    cat = client.get_events(
        starttime=UTCDateTime(2016, 11, 13),
        endtime=UTCDateTime(2016, 11, 14),
        minlatitude=-41.9,
        maxlatitude=-41.6,
        minlongitude=174.0,
        maxlongitude=174.4,
        maxdepth=40
    )
    Logger.info(f"Downloaded {len(cat)} events")

    bulk = _get_bulk_for_cat(cat=cat)
    inv = client.get_stations_bulk(bulk, level="channel")

    for event in cat:
        event.write(f"{indir}/{event.resource_id.__str__().split('/')[-1]}.xml", format="QUAKEML")

    inv.write(stationxml, format="STATIONXML")
    Logger.info("Completed test set-up")
    return


def main(
    indir: str,
    outdir: str,
    stationxmldir: str,
    velocitymodel: str,
):
    """
    Read files from input directory, locate them, and write the results to outdir.
    """
    infiles = glob.glob(f"{indir}/*")
    Logger.info(f"Found {len(infiles)} to locate")

    inv = read_inventory(f"{stationxmldir}/*.xml")
    vmodel = VelocityModel.read(velocitymodel)

    remodel = True  # Rebuild velocity tables first time.
    for i, infile in enumerate(infiles):
        Logger.info(f"Working on event-file {i} of {len(infiles)}:\t{infile}")
        try:
            _cat = read_events(infile)
        except Exception as e:
            Logger.error(f"Could not read {infile} due to {e}")
            continue
        cat_out = Catalog()
        for event in _cat:
            event_located = seisan_hyp(
                event=event, inventory=inv, velocities=vmodel.velocities,
                vpvs=vmodel.vpvs, remodel=remodel, clean=False)
            remodel = False  # Do not redo that work if we don't need to
            cat_out += event_located
        outfile = f"{outdir}/{os.path.basename(infile)}"
        Logger.info(f"Writing located event to {outfile}")
        cat_out.write(outfile, format="QUAKEML")

    return


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Locate events using SEISAN's HYPOCENTER")

    parser.add_argument(
        "-i", "--indir", type=str, required=True,
        help="Input directory to read events from")
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory for events")
    parser.add_argument(
        "-s", "--stationxmldir", type=str, required=True,
        help="Location of stationxml file containing at least station locations")
    parser.add_argument(
        "-vm", "--velocitymodel", type=str, required=True,
        help="Location of velocity model file")
    parser.add_argument(
        "--test", action="store_true",
        help="Run a test that will download events and stationxml for you")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase verbosity")

    args = parser.parse_args()

    level = logging.INFO
    if args.verbose:
        level =  logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    if args.test:
        setup_testcase(indir=args.indir, stationxml=args.stationxml)

    main(
        indir=args.indir,
        outdir=args.outdir,
        stationxmldir=args.stationxmldir,
        velocitymodel=args.velocitymodel
    )