"""
Default handling of rt_eqcorrscan plugins.
"""

import fnmatch
import logging
import subprocess
import os
import time
import shutil

from abc import ABC, abstractmethod

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:  # pragma: no cover
    from yaml import Loader, Dumper

from typing import Iterable, List

# Dict of registered plugins - no other plugins will be callable
# entry point must point to script to run the plugin. Plugin should run as a
# continuously running while loop until killed.

# Entry point defined by script name in plugins.console_scripts and looked up in setup.py
REGISTERED_PLUGINS = {
    "picker": "rteqcorrscan-plugin-picker",
    "hyp": "rteqcorrscan-plugin-hyp",
    "plotter": "rteqcorrscan-plugin-plotter",
    "growclust": "rteqcorrscan-plugin-growclust",
}
PLUGIN_CONFIG_MAPPER = dict()
# Control order of plugins, outdir of previous plugin is input to plugin
ORDERED_PLUGINS = ["picker", "hyp", "growclust", "mag_calc", "plotter"]

Logger = logging.getLogger(__name__)


# TODO: This could have a threaded watch method, but it seems like more effort
#  than needed
class Watcher:
    def __init__(self, top_directory: str, watch_pattern: str, history: set = None):
        if history is None:
            history = set()
        self.top_directory = top_directory
        self.watch_pattern = watch_pattern  # Pattern to glob for
        self.history = history  # Container for old, processed events
        self.new = set()  # Container for new, unprocessed events

    def __repr__(self):
        return (f"Watcher(watch_pattern={self.watch_pattern}, "
                f"history={self.history}, new={self.new}")

    def __len__(self):
        return len(self.new)

    def processed(self, events: Iterable):
        """ Move events into the history """
        for event in events:
            if event in self.new:
                self.new.discard(event)
            else:
                Logger.warning(f"Putting {event} into history, but {event} was"
                               f" not in unprocessed set")
            self.history.add(event)

    def check_for_updates(self):
        files = []
        for head, dirs, _files in os.walk(self.top_directory):
            if len(_files):
                Logger.debug(f"Files: {_files}")
            _files = fnmatch.filter(_files, self.watch_pattern)
            if len(_files):
                files.extend([os.path.join(head, f) for f in _files])
        new = {f for f in files if f not in self.history}
        Logger.debug(f"Found {len(new)} new events to process in "
                    f"{self.top_directory}[...]{self.watch_pattern}")
        self.new = new


class _Plugin(ABC):
    """
    Abstract Base Class for plugins with commonly used methods.
    """
    watch_pattern = "*.xml"
    name = "Plugin"

    def __init__(self, config_file: str, name: str = None):
        self.config = self._read_config(config_file=config_file)
        self._config_file = config_file
        self.watcher = Watcher(
            top_directory=self.config.in_dir,
            watch_pattern=self.watch_pattern,
            history=None)
        self.kill_watcher = Watcher(
            top_directory=self.config.out_dir,
            watch_pattern="poison",
            history=None)
        if name:
            self.name = name

    @abstractmethod
    def _read_config(self, config_file: str):
        """ Us the appropriate config """

    @abstractmethod
    def core(self, new_files: Iterable) -> List:
        """ The internal plugin code to actually run the plugin! """

    def _cleanup(self):
        """ Anything that needs to be done at the end of a run. """
        pass

    def run(self, loop: bool = True):
        """

        Parameters
        ----------
        loop

        Returns
        -------

        """
        if not os.path.isdir(self.config.out_dir):
            os.makedirs(self.config.out_dir)
        while True:
            tic = time.time()

            # Check for changed config
            new_config = self._read_config(config_file=self._config_file)
            new_in_dir = new_config.get("in_dir")
            new_out_dir = new_config.get("out_dir")

            if new_config.in_dir != self.config.in_dir:
                Logger.info(f"Looking for events in a new in dir: "
                            f"{new_config.in_dir}")
                in_dir = new_in_dir
                self.watcher = Watcher(
                    top_directory=in_dir,
                    watch_pattern=self.watcher.watch_pattern,
                    history=self.watcher.history)
            if new_config.out_dir != self.config.out_dir:
                Logger.info(f"Using a new out dir: {new_config.out_dir}")
                out_dir = new_out_dir
                self.kill_watcher = Watcher(
                    top_directory=out_dir,
                    watch_pattern=self.kill_watcher.watch_pattern,
                    history=None)
            if new_config != self.config:
                Logger.info(f"Updated configuration found: {new_config}")
                self.config = new_config

            self.kill_watcher.check_for_updates()
            if len(self.kill_watcher):
                Logger.critical(f"{self.name} plugin killed")
                Logger.critical(f"Found files: {self.kill_watcher}")
                break

            self.watcher.check_for_updates()
            if not len(self.watcher):
                if loop:
                    Logger.debug(
                        f"No new events found, sleeping for "
                        f"{self.config.sleep_interval}")
                    time.sleep(self.config.sleep_interval)
                    continue
                else:
                    Logger.info("No new events found, returning")
                    break

            # We have some events to process!
            new_files = self.watcher.new.copy()
            processed_files = self.core(new_files=new_files)

            self.watcher.processed(processed_files)
            if loop:
                # Check for poison again before sleeping
                self.kill_watcher.check_for_updates()
                if len(self.kill_watcher):
                    Logger.error(f"{self.name} plugin killed")
                # Sleep and repeat
                toc = time.time()
                elapsed = toc - tic
                Logger.info(f"{self.name} loop took {elapsed:.2f} s")
                if elapsed < self.config.sleep_interval:
                    time.sleep(self.config.sleep_interval - elapsed)
                continue
            else:
                break
        self._cleanup()
        return


def run_plugin(
    plugin: str,
    plugin_args: list,
):
    plugin_path = REGISTERED_PLUGINS.get(plugin)
    executable_path = shutil.which(plugin_path)
    if executable_path is None:
        raise FileNotFoundError(f"plugin: {plugin} is not registered")
    else:
        Logger.info(f"Running {plugin} at {executable_path}")
    # Start plugin subprocess
    _call = [plugin_path]
    _call.extend(plugin_args)

    Logger.info("Running `{call}`".format(call=" ".join(_call)))
    proc = subprocess.Popen(_call)

    return proc
