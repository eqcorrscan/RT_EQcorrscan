"""
Default handling of rt_eqcorrscan plugins.
"""

import fnmatch
import logging
import subprocess
import os
import glob
import shutil

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:  # pragma: no cover
    from yaml import Loader, Dumper

from typing import List, Set, Iterable

# Dict of registered plugins - no other plugins will be callable
# entry point must point to script to run the plugin. Plugin should run as a
# continuously running while loop until killed.

# Entry point defined by script name in plugins.console_scripts and looked up in setup.py
REGISTERED_PLUGINS = {
    "lag_calc": "rteqcorrscan-plugin-lag-calc",
    "hyp": "rteqcorrscan-plugin-hyp",
}
PLUGIN_CONFIG_MAPPER = dict()
# Control order of plugins, outdir of previous plugin is input to plugin
ORDERED_PLUGINS = ["lag_calc", "hyp", "growclust", "mag_calc"]

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
