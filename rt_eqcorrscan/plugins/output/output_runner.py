"""
Runner for handling outputs

Defaults to output the "best" catalogue - e.g. will include
all template events and detections with NLL locations and
growclust locations.
"""

import logging

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)


Logger = logging.getLogger(__name__)


class OutputConfig(_PluginConfig):
    """
    Configuration for the output plugin
    """
    defaults = {
        "sleep_interval": 20,
        "output_templates": True,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"output": OutputConfig})


class Outputter(_Plugin):
    name = "Outputter"

    def _read_config(self, config_file: str):
        return OutputConfig.read(config_file=config_file)

    def core(self):
        pass