"""
Runner for handling outputs

Defaults to output the "best" catalogue - e.g. will include
all template events and detections with NLL locations and
growclust locations.
"""

import logging
import glob

from typing import List

from obspy import read_events

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)
from rt_eqcorrscan.plugins.plotter.helpers import sparsify_catalog


Logger = logging.getLogger(__name__)


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

    def core(self, new_files: List[str], cleanup: bool) -> List:
        internal_config = self.config.copy()
        if not isinstance(internal_config.in_dir, list):
            internal_config.in_dir = [internal_config.in_dir]
        out_dir = internal_config.pop("out_dir")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # Get all templates
        if internal_config.output_templates:
            if internal_config.template_dir is None:
                Logger.error("Output templates requested, but template dir not set")
            else:
                t_files = glob.glob(f"{internal_config.template_dir}/*.xml")
                _tkeys = self.template_dict.keys()
                self.template_dict.update(
                    {t_file: sparsify_catalog(read_events(t_file))
                     for t_file in t_files if t_file not in _tkeys})

        # Get all events - group by directory
        event_groups = {in_dir: [] for in_dir in internal_config.in_dir}
        for file in new_files:
            dirname = os.path.dirname(file)
            event = read_events(file)
            for key, events in event_groups.items():
                if key in dirname:
                    events.append(sparsify_catalog(event))
                    break
            else:
                Logger.error(f"Event from {file} is not from an in-dir!?")

        # Summarise - keep all templates unless they have located options -
        # keep events in order of in_dirs
        output_catalog = {}
        for in_dir in internal_config.in_dir[::-1]:
            events = event_groups[in_dir]
            for event in events:
                if event.resource_id.id not in output_catalog.keys():
                    output_catalog.update({event.resource_id.id: event})
        output_catalog = [ev for v in output_catalog.values() for ev in v]

        # Add in templates as needed
        for template in templates:
            pass

        # Output csv and QML


        return

    def extras(self, *args, **kwargs):
        """ Do some extra work on the catalogue """
        return