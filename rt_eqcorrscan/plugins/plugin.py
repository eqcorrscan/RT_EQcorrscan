"""
Default handling of rt_eqcorrscan plugins.
"""

import logging
import subprocess
import os
import glob

from typing import List, Set, Iterable

# Dict of registered plugins - no other plugins will be callable
# entry point must point to script to run the plugin. Plugin should run as a
# continuously running while loop until killed.

# Plugin death is communicated by a kill_file

REGISTERED_PLUGINS = {
    "reloc": "relocation/relocator.py"
}

Logger = logging.getLogger(__name__)


# TODO: This could have a threaded watch method, but it seems like more effort
#  than needed
class Watcher:
    def __init__(self, watch_pattern: str, history: set = None):
        if history is None:
            history = set()
        self.watch_pattern = watch_pattern  # Pattern to glob for
        self.history = history  # Container for old, processed events
        self.new = set()  # Container for new, unprocessed events

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
        files = glob.glob(self.watch_pattern)
        new = {f for f in files if f not in self.history}
        self.new = new


def run_plugin(
    plugin: str,
    plugin_args: list,
):
    plugin_path = REGISTERED_PLUGINS.get(plugin)
    if plugin_path is None:
        raise FileNotFoundError(f"plugin: {plugin} is not registered")
    # Start plugin subprocess
    plugin_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), plugin_path)
    _call = [
        "python", plugin_path,
    ]
    _call.extend(plugin_args)

    Logger.info("Running `{call}`".format(call=" ".join(_call)))
    proc = subprocess.Popen(_call)

    return proc
