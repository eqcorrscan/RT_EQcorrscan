"""
Runner for handling outputs

Defaults to output the "best" catalogue - e.g. will include
all template events and detections with NLL locations and
growclust locations.
"""

import logging
import glob
import os
import time
import shutil
import pickle

from typing import List, Union, Set
from collections import OrderedDict

from obspy import read_events, Catalog, UTCDateTime
from obspy.core.event import Event

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)
from rt_eqcorrscan.helpers.sparse_event import (
    sparsify_catalog, get_origin_attr, get_magnitude_attr, SparseEvent)

Logger = logging.getLogger(__name__)

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


def catalog_to_csv(
    catalog: Union[Catalog, List[SparseEvent]],
    csv_filename: str
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
        "Depth (km)": "get_origin_attr(event, 'depth') / 1000.0",
        "Magnitude": "get_magnitude_attr(event, 'mag')",
        "Location Method ID": "get_origin_attr(event, 'method_id')",
    })

    lines = [", ".join(columns.keys())]
    # Sort in increasing time, with any events without an origin time coming last
    for event in sorted(catalog, key=lambda e: get_origin_attr(e, "time") or UTCDateTime(9999, 1, 1)):
        l = []
        # NB: Needs to be in loop rather than listcomp to get "event" defined
        for method in columns.values():
            l.append(str(eval(method)))
        lines.append(", ".join(l))

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

    def _read_config(self, config_file: str):
        return OutputConfig.read(config_file=config_file)

    def input_filenames(self) -> Set:
        """ Get the input filenames used for events output thusfar. """
        return {v[0] for v in self.output_events.values()}

    def output_catalog(self) -> List[SparseEvent]:
        """ Get the output events. """
        return [v[1] for v in self.output_events.values()]

    @property
    def _mainshock_time(self) -> UTCDateTime:
        if self.config.mainshock_id is None:
            return UTCDateTime(0)
        mainshock = [ev for ev in self.template_dict.values()
                     if self.config.mainshock_id in ev.resource_id.id]
        if len(mainshock) == 0:
            Logger.error(f"Did not find mainshock ({self.config.mainshock_id}) "
                         f"in templates")
            return UTCDateTime(0)
        elif len(mainshock) > 1:
            Logger.error(f"Found multiple matches for mainshock "
                         f"({self.config.mainshock_id}) in templates")
        mainshock.sort(key=lambda ev: get_origin_attr(ev, "time"))
        return get_origin_attr(mainshock[0], "time")

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
            # We don't want to output events from before our mainshock
            if get_origin_attr(event, "time") < self._mainshock_time:
                # Don't output template events before our trigger event
                self._read_files.append(file)  # Don't re-read this file.
                continue
            event = sparsify_catalog(event, include_picks=True)
            assert len(event) == 1, f"Multiple events in {file} - not supported"
            event = event[0]
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
                                f"{old_in_dir.split('/')[-1]} to one from {new_in_dir.split('/')[-1]}.")
                else:
                    Logger.info(f"New file read, but with lower priority, not updating {event.resource_id.id}")
                    continue
            # If we got to hear, we either have a new event id, or an update
            self.output_events.update({event.resource_id.id: (file, event)})
        toc = time.perf_counter()
        Logger.info(f"Reading new events took {toc - tic:.2f} s")

        tic = time.perf_counter()
        # Add in templates as needed
        template_outputs = dict()
        if internal_config.output_templates:
            for t_file, t_event in self.template_dict.items():
                if get_origin_attr(t_event, "time") < self._mainshock_time:
                    # Don't output template events before our trigger event
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
                if len(template_possible_self_dets(template_event=t_event, catalog=t_events)):
                    Logger.debug(f"Found likely self detections for template "
                                 f"{t_name}, not including template to output")
                    continue
                Logger.info(f"No self detections for {t_name}, "
                            f"adding template to output")
                template_outputs.update({t_name: (t_file, t_event)})
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f} s to check for self detections")

        # Output csv and QMLs
        tic = time.perf_counter()
        # Clear old output
        if os.path.isdir(f"{out_dir}/catalog"):
            if retain_history:
                shutil.move(
                    f"{out_dir}/catalog.csv",
                    f"{out_dir}/history/catalog_{UTCDateTime.now().strftime('%Y%m%dT%H%M%S')}.csv")
            shutil.rmtree(f"{out_dir}/catalog")
        os.makedirs(f"{out_dir}/catalog")

        # Link events
        output_events = []
        Logger.info(f"We have a total of {len(self.output_events)} detections "
                    f"from {len(self.template_dict)} templates.")
        Logger.info(f"Of these templates, {len(template_outputs)} have no "
                    f"self-detections")
        for value in self.output_events.values():
            ev_file, ev = value
            output_events.append(ev)
            ev_file_fname = os.path.basename(ev_file)
            shutil.copyfile(ev_file, f"{out_dir}/catalog/{ev_file_fname}")
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
                        f"{out_dir}/catalog/"
                        f"{os.path.splitext(ev_file_fname)[0]}.xml",
                        format="QUAKEML")
                else:
                    shutil.copyfile(
                        ev_file, f"{out_dir}/catalog/{ev_file_fname}")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f}s to write catalog output")
        tic = time.perf_counter()
        catalog_to_csv(catalog=output_events,
                       csv_filename=f"{out_dir}/catalog.csv")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f}s to write csv output")

        return []  # We need to process everything every time...

    def extras(self, *args, **kwargs):
        """ Do some extra work on the catalogue """
        return