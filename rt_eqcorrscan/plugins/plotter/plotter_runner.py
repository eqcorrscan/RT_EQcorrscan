"""
Runner for the plotting funcs.
"""

import os
import logging
from typing import Iterable, List

from obspy import read_events, UTCDateTime

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)
from rt_eqcorrscan.helpers.sparse_event import sparsify_catalog
from rt_eqcorrscan.plugins.plotter.rcet_plots import (
    aftershock_map, check_catalog, mainshock_mags
)


Logger = logging.getLogger(__name__)


class PlotConfig(_PluginConfig):
    """
    Configuration for the plotter plugin.
    """
    defaults = {
        "sleep_interval": 600,
        "mainshock_id": None,
        "inventory": None,
        "png_dpi": 300,
        "eps_dpi": 300,
        "scaled_mag_relation": 1, # What is this? Why is it an integer?
        "faulting_type": None, # Should be a string ["NN", "SS", "RV", "DS", "SI"]
        "rupture_area": "ellipse", # or "rectangle"
        "Mw": None,
        "Mw_unc": None,
        "MT_NP1": None, # [strike, dip, rake]
        "MT_NP2": None, # [strike, dip, rake]
        "fabric_angle": 55,  # What is this?
        "ellipse_std": 2,  # what is this?
        "IQR_k": 1.5,
        "lowess": True,
        "lowess_f": 0.5,
        "magcut": 3.0,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"plotter": PlotConfig})


class Plotter(_Plugin):
    name = "Plotter"
    detected_events = {}  # dict of sparse events keyed by event id

    def _read_config(self, config_file: str):
        return PlotConfig.read(config_file=config_file)

    def core(self, new_files: Iterable, cleanup: bool = True) -> List:
        """ Run the plotter. """
        internal_config = self.config.copy()
        out_dir = internal_config.pop("out_dir")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        """
        Things we need for the plots:
        1. The real-time catalogue - ideally we could split this into different origins
            This will come from config.in_dir - ideally identify our version of the mainshock in this.
        2. The template catalogue
            This will come from config.template_dir
        3. The mainshock/trigger event
            This needs to be set by rt_match_filter configure_plugins
        4. The inventory
            This needs to be set by rt_match_filter configure_plugins
        5. Search radius
            This needs to be set by rt_match_filter configure_plugins
        
        We want to be able to get the moment tensor from the config file, and 
        the faulting type and rupture area type - these should be update-able 
        in the config file.
        """
        # Load template events
        self.get_template_events()

        # Read detected events
        processed_files = []
        for f in new_files:
            cat = sparsify_catalog(read_events(f), include_picks=True)
            for ev in cat:
                self.detected_events.update({ev.resource_id.id: ev})
            processed_files.append(f)

        return processed_files


if __name__ == "__main__":
    import doctest

    doctest.testmod()
