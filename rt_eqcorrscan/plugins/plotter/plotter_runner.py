"""
Runner for the plotting funcs.
"""

import os
import logging
import time
import shutil

import pandas as pd
from obspy import read_events, UTCDateTime

from obsplus.events.pd import events_to_df

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    Watcher, PLUGIN_CONFIG_MAPPER)
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


def main(
    config_file: str,
    loop: bool = True,
):
    """
    Main hyp-plugin runner.

    Parameters
    ----------
    config_file
        Path to configuration file for hyp runner
    loop
        Whether to keep watching for updates (default) or not.
    """
    config = PlotConfig.read(config_file=config_file)
    in_dir = config.pop("in_dir")
    out_dir = config.pop("out_dir")

    watcher = Watcher(
        top_directory=in_dir, watch_pattern="*.xml", history=None)
    kill_watcher = Watcher(
        top_directory=out_dir, watch_pattern="poison", history=None)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    full_catalog, cat_df = [], None
    while True:
        tic = time.time()

        # Check for changed config
        new_config = PlotConfig.read(config_file=config_file)
        new_in_dir = new_config.pop("in_dir")
        new_out_dir = new_config.pop("out_dir")
        if new_config != config:
            Logger.info(f"Updated configuration found: {new_config}")
            config = new_config
        if new_in_dir != in_dir:
            Logger.info(f"Looking for events in a new in dir: {new_in_dir}")
            in_dir = new_in_dir
            watcher = Watcher(
                top_directory=in_dir, watch_pattern=watcher.watch_pattern,
                history=watcher.history)
        if new_out_dir != out_dir:
            Logger.info(f"Using a new out dir: {new_out_dir}")
            out_dir = new_out_dir
            kill_watcher = Watcher(
                top_directory=out_dir,
                watch_pattern=kill_watcher.watch_pattern,
                history=None)

        kill_watcher.check_for_updates()
        if len(kill_watcher):
            Logger.critical("Plotter plugin killed")
            Logger.critical(f"Found files {kill_watcher}")
            break

        watcher.check_for_updates()
        if not len(watcher):
            if not loop:
                Logger.error("Found no events to plot")
                return
            Logger.debug(
                f"Found no new events, sleeping for {config.sleep_interval}")
            time.sleep(config.sleep_interval)
            continue

        new_files, processed_files = watcher.new.copy(), []
        for i, infile in enumerate(new_files):
            Logger.info(
                f"Working on event-file {i} of {len(new_files)}:\t{infile}")
            try:
                _cat = read_events(infile)
            except Exception as e:
                Logger.error(f"Could not read {infile} due to {e}")
                continue
            # Retain sparse events rather than full catalog
            full_catalog.extend(sparsify_catalog(_cat))
            _cat_df = events_to_df(_cat)
            if cat_df is not None:
                cat_df = pd.concat([cat_df, _cat_df])
            else:
                cat_df = _cat_df

        Logger.info(f"Making plots for {len(full_catalog)} events")
        inter_fig = inter_event_plot(catalog=full_catalog)
        Logger.info("Made time-series plot")
        if PYGMT_INSTALLED:
            map_fig = plot_map(catalog=full_catalog)
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

        watcher.processed(processed_files)

        # Check for poison again before sleeping
        kill_watcher.check_for_updates()
        if len(kill_watcher):
            Logger.error("Plotter plugin killed")
        # Sleep and repeat
        toc = time.time()
        elapsed = toc - tic
        Logger.info(f"Plotter loop took {elapsed:.2f} s")
        if not loop:
            # Get outahere once we have made the plots
            break
        if elapsed < config.sleep_interval:
            time.sleep(config.sleep_interval - elapsed)
        continue
    return


if __name__ == "__main__":
    import doctest

    doctest.testmod()