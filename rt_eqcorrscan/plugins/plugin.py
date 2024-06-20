"""
Default handling of rt_eqcorrscan plugins.
"""

import logging
import subprocess
import os
import glob

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:  # pragma: no cover
    from yaml import Loader, Dumper

from typing import List, Set, Iterable

from rt_eqcorrscan.config.config import _ConfigAttribDict

# Dict of registered plugins - no other plugins will be callable
# entry point must point to script to run the plugin. Plugin should run as a
# continuously running while loop until killed.

# Plugin death is communicated by a kill_file

REGISTERED_PLUGINS = {
    "reloc": "relocation/relocator.py",
    "hyp": "relocation/hyp_runner.py",
    "mag-calc": "magnitudes/local_magnitudes.py",
    "lag-calc": "lag-calc/lag_calc_runner.py",
}

Logger = logging.getLogger(__name__)


class _PluginConfig(_ConfigAttribDict):
    """ Base configuration for plugins. """
    defaults = dict()
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write(self, config_file: str) -> None:
        """ Write to a yaml formatted file. """
        with open(config_file, "w") as f:
            f.write(dump(self.to_yaml_dict(), Dumper=Dumper))

    @classmethod
    def read(cls, config_file: str):
        with open(config_file, "rb") as f:
            config = load(f, Loader=Loader)
            # Convert spaces to underscores
            config = {key.replace(" ", "_"): value
                      for key, value in config.items()}
        return cls(**config)


# TODO: This could have a threaded watch method, but it seems like more effort
#  than needed
class Watcher:
    def __init__(self, watch_pattern: str, history: set = None):
        if history is None:
            history = set()
        self.watch_pattern = watch_pattern  # Pattern to glob for
        self.history = history  # Container for old, processed events
        self.new = set()  # Container for new, unprocessed events

    def __repr__(self):
        return f"Watcher(watch_pattern={self.watch_pattern}, history={self.history}, new={self.new}"

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
        files = glob.glob(self.watch_pattern)
        new = {f for f in files if f not in self.history}
        Logger.info(f"Found {len(new)} new events to process")
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
