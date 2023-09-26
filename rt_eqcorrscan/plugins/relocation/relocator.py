"""
Relocation workflow.

Steps:
1. Watch directory for detections
2. If new detections, read in to memory (event and waveform)
3. Run lag-calc against all old detections in memory that meet distance criteria
4. Append to dt.cc file
5. [Optional] Run initial location program (hyp by default)
6. Append to event.dat file
7. Run relocation code (growclust by default)
8. Output resulting catalogue to csv and json formatted Catalog
   (faster IO than quakeml)

Designed to be run as continuously running subprocess managed by rt_eqcorrscan
"""

import fnmatch
import logging
import glob
import os
import pickle
import subprocess

from typing import Tuple, List
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

from obspy import (
    read_events, Catalog, read_inventory, UTCDateTime, Stream, Inventory)
from obspy.core.event import Event
from obspy.io.mseed.core import _is_mseed, _read_mseed

from obsplus import WaveBank

from eqcorrscan import Tribe, Party, Family, Detection
from eqcorrscan.utils.catalog_to_dd import (
    _hypodd_event_str, write_station, _compute_dt_correlations,
    write_correlations, write_event, write_phase)

from rt_eqcorrscan.config import read_config
from rt_eqcorrscan.plugins.plugin import Watcher
from rt_eqcorrscan.plugins.relocation.hyp_runner import (
    seisan_hyp, VelocityModel)
from rt_eqcorrscan.plugins.relocation.growclust_runner import (
    write_stlist, phase_to_evlist, run_growclust)

Logger = logging.getLogger(__name__)

####################### WAVEFORM DB HELPERS #############################


def _within_times(
    required: Tuple[UTCDateTime, UTCDateTime],
    available: Tuple[UTCDateTime, UTCDateTime]
) -> bool:
    req_start, req_end = required
    avail_start, avail_end = available

    if req_start <= avail_start < req_end:
        # Available data start within bounds
        return True
    if req_start < avail_end <= req_end:
        # Available data end within bound
        return True
    if avail_start <= req_start < avail_end:
        # Available data span the required period
        return True

    return False


def _summarize_mseed(filename: str) -> dict:
    st = _read_mseed(filename, headonly=True)
    summary = dict()
    for tr in st:
        starttime, endtime = tr.stats.starttime, tr.stats.endtime
        if tr.id in summary.keys():
            starttime = min(starttime, summary[tr.id]['starttime'])
            endtime = max(endtime, summary[tr.id]['endtime'])
        summary.update({tr.id: {'starttime': starttime, 'endtime': endtime}})
    return summary


class LocalMSEEDDB:
    __database = dict()
    def __init__(self, path_to_db: str, scan_on_init: bool = True):
        self.path_to_db = path_to_db
        if scan_on_init:
            self.scan()

    def __repr__(self):
        return f"LocalMSEEDDB(path_to_db='{self.path_to_db}')"

    @property
    def database(self) -> dict:
        return self.__database

    @database.setter
    def database(self, database: dict):
        self.__database = database

    def scan(self, only_new: bool = True):
        """ Scan for new files """
        contents = glob.glob(f"{self.path_to_db}/**", recursive=True)
        if only_new:
            contents = [c for c in contents if c not in self.database.keys()]

        contents = [c for c in contents if os.path.isfile(c)]
        Logger.info(f"Scanning {len(contents)} files")
        db = dict()
        with ThreadPoolExecutor() as executor:
            for key, value in zip(contents, executor.map(
                    _summarize_mseed, contents,
                    chunksize=len(contents) // cpu_count())):
                db.update({key: value})
        db.update(self.database or dict())
        self.database = db

    def get_waveforms(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime
    ) -> Stream:
        nslc = ".".join([network, station, location, channel])
        files = []
        for f, value in self.database.items():
            for _nslc, times in value.items():
                if not fnmatch.fnmatch(nslc, _nslc):
                    continue
                _start, _end = times['starttime'], times['endtime']
                if _within_times(
                        required=(starttime, endtime),
                        available=(_start, _end)):
                    files.append(f)
        st = Stream()
        for f in files:
            st += _read_mseed(f)

        st.merge()
        st.trim(starttime=starttime, endtime=endtime)
        return st


def bulk_for_event(
    event: Event, pre_pick: float, length: float, inventory: Inventory = None,
) -> List[Tuple[str, str, str, str, UTCDateTime, UTCDateTime]]:
    bulk = []
    if inventory:
        nslc_filter = [".".join(
            [n.code or "*", s.code or "*",
             c.location_code or "*", c.code or "*"])
                       for n in inventory for s in n for c in s]
    else:
        nslc_filter = ["*.*.*.*"]
    for pick in event.picks:
        n = pick.waveform_id.network_code or "*"
        s = pick.waveform_id.station_code or "*"
        l = pick.waveform_id.location_code or "*"
        c = pick.waveform_id.channel_code or "*"
        nslc = ".".join([n, s, l, c])
        if not sum(fnmatch.fnmatch(nslc, _nslc) for _nslc in nslc_filter):
            continue
        bulk.append(
            (n, s, l, c,
             pick.time - pre_pick,
             (pick.time - pre_pick) + length
            ))
    return bulk

######################### RUNNERS #############################

def family_from_event(
    event: Event,
    tribe: Tribe,
    threshold_type: str,
    threshold_input: float
):
    """ Make a Detection from an event detected by RTEQcorrscan """
    template_name = [c.text.split("Template: ")[-1]
                     for c in event.comments if "Template:" in c.text]
    assert len(template_name) == 1, "Template not found in event comments"
    template_name = template_name[0]
    template = tribe.select(template_name)
    threshold = [c.text.split("threshold=")[-1]
                 for c in event.comments if "threshold=" in c.text]
    assert len(threshold) == 1, "Threshold not found in event comments"
    threshold = float(threshold[0])
    detect_val = [c.text.split("detect_val=")[-1]
                  for c in event.comments if "detect_val=" in c.text]
    assert len(detect_val) == 1, "Detect val not found in event comments"
    detect_val = float(detect_val[0])
    channels = [c.text.split("channels used: ")[-1]
                for c in event.comments if "channels used: " in c.text]
    assert len(channels) == 1, "channels not found in event comments"
    # Fairly ugly manipulation from string to list of tuples
    channels = channels[0].lstrip("(").rstrip(")").split(") (")
    channels = [c.split(', ') for c in channels]
    channels = [(c[0].replace("'", ""), c[1].replace("'", ""))
                for c in channels]

    rid = event.resource_id.id.split('/')[-1]
    det_time = UTCDateTime.strptime(rid.split("_")[-1], "%Y%m%dT%H%M%S.%f")

    detection = Detection(
        template_name=template_name,
        detect_time=det_time,
        no_chans=len(channels),
        detect_val=detect_val,
        threshold=threshold,
        typeofdet="corr",
        threshold_type=threshold_type,
        threshold_input=threshold_input,
        chans=channels,
        event=event,
        id=rid)

    return Family(template=template, detections=[detection])


def relocator(
    detect_dir: str,
    config_file: str,
    inventory_file: str,
    working_dir: str,
    velocity_file: str = "vmodel.txt",
):
    config = read_config(config_file=config_file)
    config.setup_logging(filename="rt_eqcorrscan_relocator.log")

    inv = read_inventory(inventory_file)
    vmodel = VelocityModel.read(velocity_file)

    tribe = Tribe()
    t_files_read = set(glob.glob(f"{working_dir}/running_templates/*.pkl"))
    for t_file in t_files_read:
        with open(t_file, "rb") as f:
            t = pickle.load(f)
        tribe += t

    waveform_db = LocalMSEEDDB(f"{working_dir}/streaming_wavebank")

    # Detections are written to YYYY/JJJ directories
    watcher = Watcher(watch_pattern=f"{detect_dir}/????/???/*.xml")

    _remodel = True
    event_id_mapper, stream_dict = None, dict()
    while True:
        watcher.check_for_updates()

        if len(watcher.new) == 0:
            continue  # Carry on waiting!

        # Read in detections
        cat_dict = dict()
        for ev_file in watcher.new:
            Logger.debug(f"Reading {ev_file}")
            cat = read_events(ev_file)
            for event in cat:
                cat_dict.update({ev.resource_id.id.split('/')[-1]: ev
                                 for ev in cat})

        # Construct party
        party = Party()
        for ev in cat_dict.values():
            party += family_from_event(
                event=ev, tribe=tribe,
                threshold_type=config.rt_match_filter.threshold_type,
                threshold_input=config.rt_match_filter.threshold)

        # Check for new waveform data
        waveform_db.scan(only_new=True)
        # Read in waveforms
        bulk = dict()
        for family in party:
            for detection in family:
                Logger.info(f"Making bulk for detection {detection.id}")
                event_bulk = bulk_for_event(
                    detection.event,
                    pre_pick=config.template.process_len / 2.5,
                    length=config.template.process_len,
                    inventory=inv)
                for _event_bulk in event_bulk:
                    nslc = ".".join(_event_bulk[0:4])
                    starttime, endtime = _event_bulk[4:]
                    if nslc in bulk.keys():
                        bulk[nslc][0] = min(starttime, bulk[nslc][0])
                        bulk[nslc][1] = max(endtime, bulk[nslc][1])
                    else:
                        bulk.update({nslc: [starttime, endtime]})
        st = Stream()
        for key, value in bulk.items():
            n, s, l, c = key.split('.')
            starttime, endtime = value
            st += waveform_db.get_waveforms(n, s, l, c, starttime, endtime)
        st = st.merge()

        # Fill gaps to make processing easier
        st = st.split().detrend().taper(
            max_length=0.5, max_percentage=0.5).merge(fill_value=0)

        # Run lag-calc
        # TODO: these params should be user definable
        post_cat = party.lag_calc(
            stream=st,
            pre_processed=False,
            shift_len=0.2,
            min_cc=0.15,
            ignore_length=True,
            ignore_bad_data=True
        )

        # Run location code (hyp)
        for i in range(len(post_cat)):
            # TODO: in offline runs we remove picks that do not fit well.
            post_cat[i] = seisan_hyp(
                event=post_cat[i], inventory=inv, velocities=vmodel.velocities,
                vpvs=vmodel.vpvs, remodel=_remodel, clean=False)
            # Only remodel the traveltime tables the first run.
            _remodel = False

        # TODO: More configurable params
        extract_len, pre_pick, shift_len = 20., 5., 0.5
        # TODO: Add to waveform dict
        for ev in post_cat:
            ev_st = Stream()
            for pick in ev.picks:
                tr_id = pick.waveform_id.get_seed_string()
                tr = st.select(id=tr_id).slice(
                    pick.time - (pre_pick + shift_len),
                    (pick.time - pre_pick) + extract_len + 2 * shift_len)
                ev_st += tr.copy()
            stream_dict.update({ev.resource_id.id: ev_st})
        # We are done with the full stream now
        del(st)

        # Compute correlations using _compute_dt_correlations
        # TODO: To do this efficiently we should have prepped waveforms in memory?
        # TODO: Ideally we would just correlate the new events each time...
        # if event_id_mapper is None:  # First run, do it all
        event_id_mapper = write_correlations(
                catalog=post_cat,
                stream_dict=stream_dict,
                extract_len=extract_len,
                pre_pick=pre_pick,
                shift_len=shift_len,
                lowcut=1.0,
                highcut=10.0,
                max_sep=8,
                min_link=8,
                min_cc=0.0,
                interpolate=False,
                all_horiz=False,
                max_workers=None,
                parallel_process=False,
                weight_by_square=True)

        # TODO: Everything from here to EOF should be in growcluster_runner

        # Write event.dat (append to already existing file)
        if os.path.isfile("event.dat"):
            with open("event.dat", "r") as f:
                original_events = f.read().splitlines()
            os.remove("event.dat")
        else:
            original_events = []
        write_event(post_cat, event_id_mapper)
        with open("event.dat", "r") as f:
            new_events = f.read().splitlines()
        original_events.extend(new_events)
        with open("event.dat", "w") as f:
            f.write("\n".join(original_events))

        # Write phase.dat (then convert to phaselist)
        if os.path.isfile("phase.dat"):
            with open("phase.dat", "r") as f:
                original_phases = f.read().splitlines()
        else:
            original_phases = []
        write_phase(post_cat, event_id_mapper)
        with open("phase.dat", "r") as f:
            new_phases = f.read().splitlines()
        original_phases.extend(new_phases)
        with open("phase.dat", "w") as f:
            f.write("\n".join(original_phases))
        phase_to_evlist("phase.dat")

        write_stlist(inv)
        # TODO: Write growclust.inp file. This should be written as a func in growclust_runner
        # Run relocation code (growclust)
        run_growclust()

        # Read back in growclust results

        # TODO: EOF

        # Write out current state of catalogue.
        # # TODO: try to not maintain too much in memory!

        # Update old events
        watcher.processed(watcher.new)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RT_EQcorrscan relocation plugin.")

    parser.add_argument(
        "-d", "--detect-dir", type=str, required=True,
        help="Detection directory to watch for new detections.")
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="Configuration file path.")
    parser.add_argument(
        "-i", "--inventory", type=str, required=True,
        help="Inventory file path.")

    args = parser.parse_args()

    relocator(detect_dir=args.detect_dir, config_file=args.config,
              inventory_file=args.inventory)
