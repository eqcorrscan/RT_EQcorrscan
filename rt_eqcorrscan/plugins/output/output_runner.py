"""
Runner for handling outputs

Defaults to output the "best" catalogue - e.g. will include
all template events and detections with NLL locations and
growclust locations.
"""
import datetime
import logging
import glob
import os
import time
import shutil
import pickle
import numpy as np

from typing import List, Union, Set, Tuple
from collections import OrderedDict

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from obspy import read_events, Catalog, UTCDateTime
from obspy.core.event import Event

from eqcorrscan.utils.findpeaks import decluster
from eqcorrscan.utils.clustering import dist_mat_km

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)
from rt_eqcorrscan.helpers.sparse_event import (
    sparsify_catalog, get_origin_attr, get_magnitude_attr, SparseEvent,
    SparseComment)

Logger = logging.getLogger(__name__)


def cluster_sparse_catalog(
    sparse_catalog: List[SparseEvent],
    thresh: float
) -> Tuple[List[SparseEvent], np.ndarray]:
    """
    Cluster a catalog by distance only. Adapted from EQcorrscan

    Will compute the matrix of physical distances between events and utilize
    the :mod:`scipy.clustering.hierarchy` module to perform the clustering.

    sparse_catalog:
        Sparse Catalog of events to clustered
    thresh:
        Maximum separation in km between centre of clusters

    returns:
        Catalog with comments of event id, cluster IDs ordered as input catalog
    """
    # Compute the distance matrix and linkage
    dist_mat = dist_mat_km(sparse_catalog)
    dist_vec = squareform(dist_mat)
    Z = linkage(dist_vec, method='average')

    # Cluster the linkage using the given threshold as the cutoff
    indices = fcluster(Z, t=thresh, criterion='distance')
    Logger.info(f"Clustered catalog into {len(set(indices))} clusters")

    # Put cluster IDs into events as a comment
    for cluster_id, ev in zip(indices, sparse_catalog):
        # Remove old cluster id
        ev.comments = [c for c in ev.comments if "ClusterID" not in c.text]
        ev.comments.append(SparseComment(text=f"ClusterID: {cluster_id}"))

    return sparse_catalog, indices


def similar_picks(
    event1: Union[Event, SparseEvent],
    event2: Union[Event, SparseEvent],
    pick_tolerance: float
) -> int:
    """
    Check if picks are similar.

    Only checks picks that have matching waveform ids and phase hints. Checks time
    if within pick_tolerance.


    Parameters
    ----------
    event1:
        First event to compare
    event2:
        Second event to compare
    pick_tolerance:
        Maximum time difference in seconds to consider picks similar

    Returns
    -------
    Number of similar picks
    """
    matched_picks = 0
    for pick1 in event1.picks:
        matched_wids = [
            p for p in event2.picks
            if p.waveform_id.get_seed_string() == pick1.waveform_id.get_seed_string()]
        matched_picks += len([p for p in matched_wids
                              if abs(p.time - pick1.time) <= pick_tolerance])
    return matched_picks


def template_possible_self_dets(
    template_event: Union[Event, SparseEvent],
    catalog: Union[List[Event], Catalog, List[SparseEvent]],
    origin_tolerence: float = 10.0,
    pick_tolerance: float = 1.0,
) -> List:
    """
    Find possible template self detections - match based on origin time first,
    then pick time. If origins do not have times they will be considered similar
    and picks will be checked.

    Parameters
    ----------
    template_event:
        Template to look for self detections of
    catalog:
        Catalog of events to check to see if they are self-detections
    origin_tolerence:
        Time difference in seconds to consider origins similar
    pick_tolerance:
        Time difference in seconds to consider picks similar

    Returns
    -------
    List of possible self detections
    """
    t_ori_time = get_origin_attr(template_event, "time")
    similar_origins = [ev for ev in catalog
                       if abs(t_ori_time - (get_origin_attr(ev, "time")
                       or t_ori_time)) <= origin_tolerence]
    self_dets = [ev for ev in similar_origins
                 if similar_picks(ev, template_event,
                                  pick_tolerance=pick_tolerance)]

    return self_dets


def _get_comment_val(value_name: str, event: Event) -> Union[float, None]:
    value = None
    for comment in event.comments:
        if value_name in comment.text:
            if "=" in comment.text:
                # Should be a number
                try:
                    value = float(comment.text.split('=')[-1])
                except ValueError:
                    # Leave as a string
                    break
                except Exception as e:
                    Logger.exception(
                        f"Could not get {value_name} {comment.text} due to {e}")
                else:
                    break
            elif ":" in comment.text:
                # Should be a string
                try:
                    value = comment.text.split(": ")[-1]
                except Exception as e:
                    Logger.exception(
                        f"Could not get {value_name} from {comment.text} due to {e}")
                else:
                    break
    return value


def get_threshold(event: Event) -> Union[float, None]:
    return _get_comment_val(value_name="threshold", event=event)


def get_det_val(event: Event) -> Union[float, None]:
    return _get_comment_val(value_name="detect_val", event=event)


def get_det_time(event: Event) -> UTCDateTime:
    try:
        rid = event.resource_id.id.split('/')[-1]
    except IndexError:
        Logger.error("RID poorly formed.")
        return get_origin_attr(event, "time")
    template = _get_comment_val("Template", event)
    if template not in rid:
        # Not an EQcorrscan detection RID, return origin time
        Logger.warning(f"{rid} is not an EQcorrscan detection, returning origin time")
        return get_origin_attr(event, "time")
    try:
        det_time = rid.split("_")[-1]
    except IndexError:
        Logger.error("RID poorly formed")
        return get_origin_attr(event, "time")
    return UTCDateTime(det_time)


def decluster_catalog(
    catalog: Union[Catalog, List[SparseEvent]],
    trig_int: float,
) -> Union[Catalog, List[SparseEvent]]:
    """
    Decluster catalogue based on detection time.

    Parameters
    ----------
    catalog
        Catalog of events to decluster
    trig_int
        Minimum inter-event time in seconds.

    Returns
    -------
    Declustered catalog
    """
    detect_vals = np.array([get_det_val(ev) for ev in catalog])
    detect_times = [get_det_time(ev) for ev in catalog]
    min_det_time = min(detect_times).datetime
    # Convert to microseconds
    detect_times = np.array([(d_t.datetime - min_det_time).total_seconds()
                             for d_t in detect_times])
    detect_times *= 10 ** 6
    detect_times = detect_times.astype(int)
    peaks_out = decluster(
                peaks=detect_vals, index=detect_times,
                trig_int=trig_int * 10 ** 6)
    # Peaks out is tuples of (detect_val, detect_time)
    declustered_catalog = []
    for ind in peaks_out:
        matching_time_indices = np.where(detect_times == ind[-1])[0]
        matches = matching_time_indices[
            np.where(detect_vals[matching_time_indices] == ind[0])[0][0]]
        declustered_catalog.append(catalog[matches])
    return declustered_catalog


def catalog_to_csv(
    catalog: Union[Catalog, List[SparseEvent]],
    csv_filename: str,
    cluster_ids: List | None = None,
) -> None:
    """
    Write catalog to a csv file.

    Parameters
    ----------
    catalog:
        Catalog of events to write
    csv_filename:
        Filename to write csv to. Will overwrite existing files
    """
    # Columns should be column_name: thing to evaluate to get thing from
    # variable named "event"
    columns = OrderedDict({
        "Resource ID": "event.resource_id.id",
        "Origin Time (UTC)": "get_origin_attr(event, 'time')",
        "Latitude": "get_origin_attr(event, 'latitude')",
        "Longitude": "get_origin_attr(event, 'longitude')",
        "Depth (km)": "(get_origin_attr(event, 'depth') or 0.0) / 1000.0",
        "Magnitude": "get_magnitude_attr(event, 'mag')",
        "Location Method ID": "get_origin_attr(event, 'method_id')",
        "N Picks": "len(event.picks)",
        "Detection threshold": "get_threshold(event)",
        "Detection value": "get_det_val(event)",
        "Self Detection ID": "event._self_det_id",
        "Cluster ID": "cluster_id",
    })

    lines = [",".join(columns.keys())]
    if isinstance(catalog, Catalog):
        catalog = catalog.events
    
    if cluster_ids is None:
        cluster_ids = [0 for _ in catalog]
    if len(cluster_ids) != len(catalog):
        Logger.warning("Cluster IDs not the same length as catalog, not outputting")
        cluster_ids = [0 for _ in catalog]
    
    # Sort in increasing time, with any events without an origin time coming last
    origin_times = np.array(
        [(get_origin_attr(ev, "time") or UTCDateTime(9999, 1, 1)).datetime
         for ev in catalog])
    order = np.argsort(origin_times)
    
    for index in order:
        event, cluster_id = catalog[index], cluster_ids[index]
        l = []
        # NB: Needs to be in loop rather than listcomp to get "event" defined
        for method in columns.values():
            l.append(str(eval(method)))
        lines.append(",".join(l))

    lines = "\n".join(lines)
    with open(csv_filename, "w") as f:
        f.write(lines)

    return


class OutputConfig(_PluginConfig):
    """
    Configuration for the output plugin
    """
    defaults = _PluginConfig.defaults.copy()
    defaults.update({
        "sleep_interval": 20,
        "output_templates": True,
        "retain_history": False,
        "mainshock_id": None,
        "trig_int": 2.0,
        "cluster": True,
        "search_radius": 100.0,
    })
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"output": OutputConfig})


class Outputter(_Plugin):
    name = "Outputter"
    template_dict = {}  # Dict of template SparseEvents keyed by filename
    output_events = {}  # Dict of output (filename, SparseEvent) tuples keyed by event-id
    _read_files = []  # List of files that we have already read. Used to avoid re-reading events
    _skipped_templates = set()  # Set of template files to not output

    def _read_config(self, config_file: str):
        return OutputConfig.read(config_file=config_file)

    def input_filenames(self) -> Set:
        """ Get the input filenames used for events output thusfar. """
        return {v[0] for v in self.output_events.values()}

    def output_catalog(self) -> List[SparseEvent]:
        """ Get the output events. """
        return [v[1] for v in self.output_events.values()]

    def decluster(self):
        original_cat_len = len(self.output_events)
        if original_cat_len <= 1:
            Logger.info(f"Only {original_cat_len} events, not declustering")
            return
        declustered_cat = decluster_catalog(
            self.output_catalog(), trig_int=self.config.trig_int)
        declustered_dict = {
            evid: (f, ev) for evid, (f, ev) in self.output_events.items()
            if ev in declustered_cat}
        Logger.info(f"Declustering at {self.config.trig_int} s removed "
                    f"{original_cat_len - len(declustered_dict)} events")
        self.output_events = declustered_dict
        return

    # @property
    # def _mainshock_time(self) -> UTCDateTime:
    #     if self.config.mainshock_id is None:
    #         return UTCDateTime(0)
    #     mainshock = [ev for ev in self.template_dict.values()
    #                  if self.config.mainshock_id in ev.resource_id.id]
    #     if len(mainshock) == 0:
    #         Logger.error(f"Did not find mainshock ({self.config.mainshock_id}) "
    #                      f"in templates")
    #         return UTCDateTime(0)
    #     elif len(mainshock) > 1:
    #         Logger.error(f"Found multiple matches for mainshock "
    #                      f"({self.config.mainshock_id}) in templates")
    #     mainshock.sort(key=lambda ev: get_origin_attr(ev, "time"))
    #     return get_origin_attr(mainshock[0], "time")

    def core(self, new_files: List[str], cleanup: bool) -> List:
        internal_config = self.config.copy()
        retain_history = internal_config.get("retain_history", False)
        if not isinstance(internal_config.in_dir, list):
            internal_config.in_dir = [internal_config.in_dir]
        out_dir = internal_config.pop("out_dir")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if not os.path.isdir(f"{out_dir}/history"):
            os.makedirs(f"{out_dir}/history")
        Logger.debug(f"Writing out to {out_dir}")

        # Get all templates
        tic = time.perf_counter()
        if internal_config.output_templates:
            self.get_template_events()
        Logger.debug(f"We have run {len(self.template_dict)} templates")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f} s to read templates")

        tic = time.perf_counter()
        for file in new_files:
            if file in self._read_files:
                Logger.debug(f"Already read from {file}, skipping.")
                continue
            # Note: reading events is the slow part of this loop.
            event = read_events(file)
            assert len(event) == 1, f"Multiple events in {file} - not supported"
            # We don't want to output events from before our mainshock
            if get_origin_attr(event[0], "time") < self.config.zero_time:
                # Don't output template events before our trigger event
                Logger.info(f"Skipping {event.resource_id.id}: before trigger")
                self._read_files.append(file)  # Don't re-read this file.
                continue
            event = sparsify_catalog(event, include_picks=True)
            event = event[0]
            # Add in self_det_id which will be overloaded later if it is a template self-detection
            event._self_det_id = None
            self._read_files.append(file)  # Keep track and don't read again
            if event.resource_id.id in self.output_events.keys():
                # Check the input directory. If the "new" event has an in-dir
                # with higher priority, overload the old event
                old_in_dir = os.path.dirname(
                    self.output_events[event.resource_id.id][0])
                new_in_dir = os.path.dirname(file)
                # NB: dirs could be nested
                old_index, new_index = 0, 0
                for i, _dir in enumerate(internal_config.in_dir):
                    if _dir in old_in_dir:
                        old_index = i
                    if _dir in new_in_dir:
                        new_index = i
                if old_index <= new_index:
                    Logger.info(f"Event {event.resource_id.id} already in output. Updating original from "
                                f"{old_in_dir.split('/')[-1]} to one from {new_in_dir}.")
                else:
                    Logger.info(f"New file read, but with lower priority, not updating {event.resource_id.id}")
                    continue
            # If we got to here, we either have a new event id, or an update
            self.output_events.update({event.resource_id.id: (file, event)})
        toc = time.perf_counter()
        Logger.info(f"Reading {len(self.output_events)} new events took {toc - tic:.2f} s")

        tic = time.perf_counter()
        # Add in templates as needed
        template_outputs = dict()
        if internal_config.output_templates:
            for t_file, t_event in self.template_dict.items():
                if t_file in self._skipped_templates or get_origin_attr(t_event, "time") < self.config.zero_Time:
                    # Don't output template events before our trigger event
                    Logger.debug(f"Skipping template {t_event.resource_id.id}: before trigger")
                    self._skipped_templates.update(t_file)
                    continue
                # If we have read in a relocated version of the template then we
                # should use that as the original template
                if t_event.resource_id.id in self.output_events.keys():
                    # Get the event from the relocated version and remove it from
                    # the output events so that we don't output it and its self detection.
                    _, t_event = self.output_events.pop(t_event.resource_id.id)
                t_name = t_event.resource_id.id.split('/')[-1]
                # Look for a template detections
                t_events = [ev[1] for rid, ev in self.output_events.items()
                            if rid.lstrip("smi:local/").startswith(t_name)]
                if len(t_events) == 0:
                    # No detections, so we need to output the template
                    Logger.info(f"No detections for {t_name}, "
                                f"adding template to output")
                    template_outputs.update({t_name: (t_file, t_event)})
                    continue
                # Look for template self detections - slop in origin time? Then match picks?
                self_dets = template_possible_self_dets(template_event=t_event, catalog=t_events)
                if len(self_dets):
                    Logger.debug(f"Found likely self detections for template "
                                 f"{t_name}, not including template to output")
                    for ev in self_dets:
                        ev._self_det_id = t_event.resource_id.id
                    continue
                Logger.info(f"No self detections for {t_name}, "
                            f"adding template to output")
                template_outputs.update({t_name: (t_file, t_event)})
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f} s to check for self detections")

        # Output csv and QMLs
        tic = time.perf_counter()
        # Move old output
        if retain_history and os.path.isfile(f"{out_dir}/catalog.csv"):
            try:
                shutil.move(
                    f"{out_dir}/catalog.csv",
                    f"{out_dir}/history/catalog_{UTCDateTime.now().strftime('%Y%m%dT%H%M%S')}.csv")
            except FileNotFoundError as e:
                Logger.exception(f"Could not write catalog history due to {e}")
        os.makedirs(f"{out_dir}/.catalog")

        # Link events
        output_events = []
        Logger.info(f"We have a total of {len(self.output_events)} detections "
                    f"from {len(self.template_dict)} templates.")
        Logger.info(f"Of these templates, {len(template_outputs)} have no "
                    f"self-detections")
        # Decluster
        if self.config.trig_int:
            self.decluster()

        for value in self.output_events.values():
            ev_file, ev = value
            output_events.append(ev)
            ev_file_fname = os.path.basename(ev_file)
            shutil.copyfile(ev_file, f"{out_dir}/.catalog/{ev_file_fname}")
        if internal_config.output_templates:
            for value in template_outputs.values():
                ev_file, ev = value
                output_events.append(ev)
                ev_file_fname = os.path.basename(ev_file)
                Logger.info(f"Working on {ev_file_fname}")
                if os.path.splitext(ev_file)[-1] in ['.pkl']:
                    Logger.info(
                        f"Reading template and writing event for {ev_file}")
                    # We need to read and spit those out as events
                    with open(ev_file, "rb") as f:
                        t = pickle.load(f)
                    t.event.write(
                        f"{out_dir}/.catalog/"
                        f"{os.path.splitext(ev_file_fname)[0]}.xml",
                        format="QUAKEML")
                else:
                    shutil.copyfile(
                        ev_file, f"{out_dir}/.catalog/{ev_file_fname}")
        # Overwrite old output - wait until we can put the *new* output 
        # out to remove the old, otherwise the next process is likely 
        # to run into missing files.
        if os.path.isdir(f"{out_dir}/catalog"):
            shutil.rmtree(f"{out_dir}/catalog")
        shutil.move(f"{out_dir}/.catalog", f"{out_dir}/catalog")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f}s to write catalog output")
        tic = time.perf_counter()

        # Do the clustering
        cluster_ids = np.zeros(len(output_events))
        if self.config.cluster and len(output_events) > 1:
            output_events, cluster_ids = cluster_sparse_catalog(
                sparse_catalog=output_events, thresh=self.config.search_radius)

        # Output pkl of full catalog - used by plotting for faster IO
        if len(output_events):
            with open(f"{out_dir}/catalog.pkl", "wb") as f:
                pickle.dump(output_events, f)

        catalog_to_csv(
            catalog=output_events,
            cluster_ids=list(cluster_ids),
            csv_filename=f"{out_dir}/catalog.csv")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f}s to write csv output")

        return []  # We need to process everything every time...

    def extras(self, *args, **kwargs):
        """ Do some extra work on the catalogue """
        return
