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

from typing import List, Union
from collections import OrderedDict

from obspy import read_events, Catalog, UTCDateTime
from obspy.core.event import Event

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)
from rt_eqcorrscan.plugins.plotter.helpers import sparsify_catalog, get_magnitude_attr, get_origin_attr


Logger = logging.getLogger(__name__)

def similar_picks(event1: Event, event2: Event, pick_tolerance: float) -> int:
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
    template_event: Event,
    catalog: Union[List[Event], Catalog],
    origin_tolerence: float = 10.0,
    pick_tolerance: float = 1.0,
) -> List[Event]:
    """
    Find possible template self detections - match based on origin time first,
    then pick time. If origins do not have times they will be considered similar
    and picks will be checked.

    Parameters
    ----------
    template:
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


def catalog_to_csv(catalog: Catalog, csv_filename: str) -> None:
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
    for event in sorted(catalog.events, key=lambda e: get_origin_attr(e, "time") or UTCDateTime(9999, 1, 1)):
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
    defaults = {
        "sleep_interval": 20,
        "output_templates": True,
        "template_dir": None,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"output": OutputConfig})


class Outputter(_Plugin):
    name = "Outputter"
    template_dict = {}  # Dict of templates keyed by filename
    full_catalog = []

    def _read_config(self, config_file: str):
        return OutputConfig.read(config_file=config_file)

    # TODO: Looks like we are being fed the same files on repeat.
    def core(self, new_files: List[str], cleanup: bool) -> List:
        internal_config = self.config.copy()
        if not isinstance(internal_config.in_dir, list):
            internal_config.in_dir = [internal_config.in_dir]
        out_dir = internal_config.pop("out_dir")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if not os.path.isdir(f"{out_dir}/catalog"):
            os.makedirs(f"{out_dir}/catalog")
        Logger.debug(f"Writing out to {out_dir}")

        # Get all templates
        tic = time.perf_counter()
        if internal_config.output_templates:
            if internal_config.template_dir is None:
                Logger.error("Output templates requested, but template dir not set")
            else:
                t_files = glob.glob(f"{internal_config.template_dir}/*.xml")
                _tkeys = self.template_dict.keys()
                for t_file in t_files:
                    if t_file in _tkeys:
                        continue
                    template = read_events(t_file)
                    assert len(template) == 1, f"Multiple templates found in {t_file}"
                    self.template_dict.update({t_file: template[0]})
        Logger.debug(f"We have run {len(self.template_dict)} templates")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f} s to read templates")

        # Get all events - group by directory (cope with events coming from multiple steps)
        # TODO: We want to avoid re-reading events every time if we can? This is really slow
        tic = time.perf_counter()
        event_groups = {in_dir: [] for in_dir in internal_config.in_dir}
        Logger.info(f"Reading from {len(new_files)} event files")
        for file in new_files:
            dirname = os.path.dirname(file)
            event = read_events(file)
            for key, events in event_groups.items():
                if key in dirname:
                    events.extend(event)
                    break
            else:
                Logger.error(f"Event from {file} is not from an in-dir!?")
        Logger.debug(f"Collected events: {event_groups.keys()}")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f} s to read events")

        # Summarise - keep all templates unless they have located options -
        # keep events in order of in_dirs - events will be overwritten in the
        # dict by events from in_dirs with higher priority (appear later in
        # the list of in_dirs).
        event_dict = {}
        tic = time.perf_counter()
        for in_dir in internal_config.in_dir:
            Logger.info(f"Checking {in_dir}")
            events = event_groups[in_dir]
            for event in events:
                if event.resource_id.id in event_dict.keys():
                    Logger.debug(f"Overloading {event.resource_id.id} "
                                 f"with updated event from {in_dir}")
                event_dict.update({event.resource_id.id: event})
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f} s to sort events")

        """
        Event IDS are named as below in rt_match_filer _handle_detections
        
        d.event.resource_id = ResourceIdentifier(
            id=d.template_name + '_' + d.time,
            prefix='smi:local')
            
        Templates are named (see database_manager):
        
        template.name = event.resource_id.id.split('/')[-1]
        """

        tic = time.perf_counter()
        # Add in templates as needed
        if internal_config.output_templates:
            template_outputs = dict()
            for t_file, t_event in self.template_dict.items():
                t_name = t_event.resource_id.id.split('/')[-1]
                # Look for a template detections
                t_events = [ev for rid, ev in event_dict.items()
                            if rid.lstrip("smi:local/").startswith(t_name)]
                if len(t_events) == 0:
                    # No detections, so we need to output the template
                    Logger.info(f"No detections for {t_name}, "
                                f"adding template to output")
                    template_outputs.update({t_name: t_event})
                    continue
                # Look for template self detections - slop in origin time? Then match picks?
                if len(template_possible_self_dets(template_event=t_event, catalog=t_events)):
                    Logger.debug(f"Found likely self detections for template "
                                 f"{t_name}, not including template to output")
                    continue
                Logger.info(f"No self detections for {t_name}, "
                            f"adding template to output")
                template_outputs.update({t_name: t_event})
            # Merge into main event dict
            event_dict.update(template_outputs)
        toc = time.perf_counter()
        Logger.info(f"Took {toc-tic:.2f} s to check for self detections")

        # Output csv and QML
        tic = time.perf_counter()
        output_catalog = Catalog([ev for ev in event_dict.values()])
        for event in output_catalog:
            event.write(
                f"{out_dir}/catalog/{event.resource_id.id.split('/')[-1]}.xml",
                format="QUAKEML")
        # output_catalog.write(f"{out_dir}/catalog.xml", format="QUAKEML")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f}s to write catalog output")
        tic = time.perf_counter()
        catalog_to_csv(catalog=output_catalog,
                       csv_filename=f"{out_dir}/catalog.csv")
        toc = time.perf_counter()
        Logger.info(f"Took {toc - tic:.2f}s to write csv output")


        return []  # We need to process everything every time...

    def extras(self, *args, **kwargs):
        """ Do some extra work on the catalogue """
        return