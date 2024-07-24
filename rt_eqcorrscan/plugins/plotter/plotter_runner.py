"""
Runner for the plotting funcs.
"""

import os
import logging
import shutil
from typing import Iterable, List

import pandas as pd
from obspy import read_events, UTCDateTime

from obsplus.events.pd import events_to_df

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)
from rt_eqcorrscan.plugins.plotter.helpers import sparsify_catalog
from rt_eqcorrscan.plugins.plotter.map_plots import PYGMT_INSTALLED, plot_map
from rt_eqcorrscan.plugins.plotter.time_series_plots import inter_event_plot


Logger = logging.getLogger(__name__)


class PlotConfig(_PluginConfig):
    """
    Configuration for the hyp plugin.
    """
    defaults = {
        "sleep_interval": 600,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"plotter": PlotConfig})


class Plotter(_Plugin):
    name = "Plotter"
    full_catalog = []
    cat_df = None

    def _read_config(self, config_file: str):
        return PlotConfig.read(config_file=config_file)

    def core(self, new_files: Iterable) -> List:
        """ Run the plotter. """
        internal_config = self.config.copy()
        out_dir = internal_config.pop("out_dir")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        processed_files = []
        for i, infile in enumerate(new_files):
            Logger.info(
                f"Working on event-file {i}:\t{infile}")
            try:
                _cat = read_events(infile)
            except Exception as e:
                Logger.error(f"Could not read {infile} due to {e}")
                continue
            # Retain sparse events rather than full catalog
            self.full_catalog.extend(sparsify_catalog(_cat))
            _cat_df = events_to_df(_cat)
            if self.cat_df is not None:
                self.cat_df = pd.concat([self.cat_df, _cat_df])
            else:
                self.cat_df = _cat_df

        Logger.info(f"Making plots for {len(self.full_catalog)} events")
        inter_fig = inter_event_plot(catalog=self.full_catalog)
        Logger.info("Made time-series plot")
        if PYGMT_INSTALLED:
            map_fig = plot_map(catalog=self.full_catalog)
            Logger.info("Made map plot")
        time_stamp = UTCDateTime().strftime("%Y-%m-%dT%H-%M-%S")
        for _format in ("png", "eps"):
            inter_fig.savefig(
                f"{out_dir}/detection_time_series_{time_stamp}.{_format}")
            shutil.copyfile(
                f"{out_dir}/detection_time_series_{time_stamp}.{_format}",
                f"{out_dir}/detection_time_series_latest.{_format}")
            if PYGMT_INSTALLED:
                map_fig.savefig(
                    f"{out_dir}/detection_map_{time_stamp}.{_format}")
                shutil.copyfile(
                    f"{out_dir}/detection_map_{time_stamp}.{_format}",
                    f"{out_dir}/detection_map_latest.{_format}")
        return processed_files


if __name__ == "__main__":
    import doctest

    doctest.testmod()
