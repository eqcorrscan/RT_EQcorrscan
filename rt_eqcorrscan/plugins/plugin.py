"""
Default handling of rt_eqcorrscan plugins.
"""

import fnmatch
import glob
import logging
import subprocess
import os
import time
import shutil
import itertools
import tarfile

from abc import ABC, abstractmethod

from obspy import UTCDateTime, Catalog, read_events

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
    "nll": "rteqcorrscan-plugin-nll",
    "plotter": "rteqcorrscan-plugin-plotter",
    "growclust": "rteqcorrscan-plugin-growclust",
    "outputter": "rteqcorrscan-plugin-outputter",
}
PLUGIN_CONFIG_MAPPER = dict()
# Control order of plugins, outdir of previous plugin is input to plugin
ORDERED_PLUGINS = ["picker", "hyp", "growclust", "mag_calc", "outputter", "plotter"]

Logger = logging.getLogger(__name__)


class Watcher:
    def __init__(self, top_directory: str, watch_pattern: str, history: dict = None):
        if history is None:
            history = dict()
        self.top_directory = top_directory
        self.watch_pattern = watch_pattern  # Pattern to glob for
        self.history = history  # Container for old, processed events
        self._new = dict()  # Container for new, unprocessed events

    def __repr__(self):
        return (f"Watcher(watch_pattern={self.watch_pattern}, "
                f"history={self.history}, new={self.new}")

    def __len__(self):
        return len(self.new)

    @property
    def new(self):
        return set(self._new.keys())

    @new.setter
    def new(self, value):
        if not isinstance(value, dict):
            raise TypeError(f"{value} must be dict")
        self._new = value

    def processed(self, events: Iterable[str]):
        """ Move events into the history """
        for event in events:
            if event in self.new:
                mtime = self._new.pop(event, None)
            else:
                Logger.warning(f"Putting {event} into history, but {event} was"
                               f" not in unprocessed set")
                mtime = None
            self.history.update({event: mtime})

    def check_for_updates(self):
        files = _scan_dir(top_dir=self.top_directory,
                          watch_pattern=self.watch_pattern)
        new = dict()
        for f in files:
            if f not in self.history.keys():
                # File is totally new to us, add it to new
                new.update({f: os.path.getmtime(f)})
            elif self.history[f] is None or os.path.getmtime(f) > self.history[f]:
                # File has been updated since we last looked - reprocess
                new.update({os.path.abspath(f): os.path.getmtime(f)})

        Logger.debug(f"Found {len(new)} new events to process in "
                    f"{self.top_directory}[...]{self.watch_pattern}")
        self._new = new


def _scan_dir(top_dir, watch_pattern):
    files = []
    for head, dirs, _files in os.walk(top_dir):
        if len(_files):
            Logger.debug(f"Files: {_files}")
        _files = fnmatch.filter(_files, watch_pattern)
        if len(_files):
            files.extend([os.path.join(head, f) for f in _files])
    return files



class _Plugin(ABC):
    """
    Abstract Base Class for plugins with commonly used methods.
    """
    watch_pattern = "*.xml"
    name = "Plugin"
    _write_sim_catalogues = False  # Flag to write time-stamped simulation cats

    def __init__(self, config_file: str, name: str = None):
        self.config = self._read_config(config_file=config_file)
        self._config_file = config_file
        self.watchers = {}  # Dict keyed by in_dir
        if not isinstance(self.config.in_dir, list):
            in_dirs = [self.config.in_dir]
        else:
            in_dirs = self.config.in_dir
        for in_dir in in_dirs:
            self.watchers.update({in_dir: Watcher(
                top_directory=in_dir,
                watch_pattern=self.watch_pattern,
                history=None)})
        self.kill_watcher = Watcher(
            top_directory=self.config.out_dir,
            watch_pattern="poison",
            history=None)
        if name:
            self.name = name
        Logger.info(f"Initialised {self.name} plugin")

    @abstractmethod
    def _read_config(self, config_file: str):
        """ Us the appropriate config """

    @abstractmethod
    def core(self, new_files: Iterable, cleanup: bool) -> List:
        """ The internal plugin code to actually run the plugin! """

    def setup(self):
        """ Run any setup here. """
        pass

    def _cleanup(self):
        """ Anything that needs to be done at the end of a run. """
        pass

    def _summarise_state(self):
        """ Summarise the events in the outdir and write a time-stamped tgz """
        now = UTCDateTime.now()
        out_files = _scan_dir(top_dir=self.config.out_dir,
                              watch_pattern=self.watch_pattern)
        if len(out_files) == 0:
            Logger.info(f"No output at {now} for {self.name}, "
                        f"not writing snapshot")
            return

        out_archive = f"{self.config.out_dir}/sim_{self.name}_at_{now}.tgz"
        with tarfile.open(out_archive, "w:gz") as tar:
            for f in out_files:
                tar.add(f)
        Logger.info(f"Written snapshot at {now} to {out_archive}")
        return

    def run(self, loop: bool = True, cleanup: bool = True):
        """

        Parameters
        ----------
        loop
        cleanup

        Returns
        -------

        """
        if not os.path.isdir(self.config.out_dir):
            os.makedirs(self.config.out_dir)
        # Run pre-looping setup - this could be making files that just need to
        # be made once or making output dirs etc.
        self.setup()
        while True:
            tic = time.time()

            # Check for changed config
            new_config = self._read_config(config_file=self._config_file)
            new_in_dir = new_config.get("in_dir")
            new_out_dir = new_config.get("out_dir")

            if new_config.in_dir != self.config.in_dir:
                # Get rid of old watchers
                self.watchers = dict()
                Logger.info(f"Looking for events in a new in dir: "
                            f"{new_config.in_dir}")
                if not isinstance(new_in_dir, list):
                    in_dir = [new_in_dir]
                else:
                    in_dir = new_in_dir
                for _in_dir in in_dir:
                    self.watchers.update({_in_dir: Watcher(
                        top_directory=_in_dir,
                        watch_pattern=self.watcher.watch_pattern,
                        history=self.watcher.history)})
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
                Logger.warning("Not re-running setup!")

            self.kill_watcher.check_for_updates()
            if len(self.kill_watcher):
                Logger.critical(f"{self.name} plugin killed")
                Logger.critical(f"Found files: {self.kill_watcher}")
                break

            new_file_dict = dict()
            for _in_dir, watcher in self.watchers.items():
                watcher.check_for_updates()
                if len(watcher):
                    new_file_dict.update({_in_dir: watcher.new.copy()})

            if not len(new_file_dict):
                if loop:
                    Logger.debug(
                        f"No new events found, sleeping for "
                        f"{self.config.sleep_interval}")
                    time.sleep(self.config.sleep_interval)
                    continue
                else:
                    Logger.info("No new events found, returning")
                    break

            # TODO: If input files are removed, remove the associated output
            #  files as well - needs a database of input file linked to output file.
            new_files = list(itertools.chain.from_iterable(new_file_dict.values()))
            processed_files = self.core(new_files=new_files, cleanup=cleanup)

            # Associate file with correct watcher
            processed_files = {
                _in_dir: [f for f in processed_files
                          if f in new_file_dict[_in_dir]]
                for _in_dir in self.watchers.keys()}
            for _in_dir, files in processed_files.items():
                self.watchers[_in_dir].processed(files)

            if loop:
                # Check for poison again before sleeping
                self.kill_watcher.check_for_updates()
                if len(self.kill_watcher):
                    Logger.error(f"{self.name} plugin killed")
                # Sleep and repeat
                toc = time.time()
                elapsed = toc - tic
                Logger.info(f"{self.name} loop took {elapsed:.2f} s")
                if self._write_sim_catalogues and len(processed_files):
                    Logger.info("Summarising state")
                    self._summarise_state()
                if elapsed < self.config.sleep_interval:
                    Logger.info(
                        f"Sleeping for "
                        f"{self.config.sleep_interval - elapsed:.2f} s")
                    time.sleep(self.config.sleep_interval - elapsed)

                continue
            else:
                break

        if cleanup:
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
