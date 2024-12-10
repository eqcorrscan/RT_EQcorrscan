"""
Classes for real-time matched-filter detection of earthquakes.
"""
import shutil
import time
import traceback
import os
import pickle
import logging
import copy
import numpy
import gc
import glob
import subprocess
import numpy as np

# Memory tracking for debugging
# import psutil
# from pympler import summary, muppy

from typing import Union, List, Iterable

from obspy import Stream, UTCDateTime, Inventory, Trace
from obsplus import WaveBank
from matplotlib.figure import Figure
from multiprocessing import Lock
from eqcorrscan import Tribe, Template, Party, Detection
from eqcorrscan.utils.pre_processing import _prep_data_for_correlation

from rt_eqcorrscan.streaming.streaming import _StreamingClient
from rt_eqcorrscan.event_trigger.triggers import average_rate
from rt_eqcorrscan.config.mailer import Notifier
from rt_eqcorrscan.plugins import (
    REGISTERED_PLUGINS, run_plugin, ORDERED_PLUGINS)
from sqlalchemy.sql import True_

Logger = logging.getLogger(__name__)


class RealTimeTribe(Tribe):
    sleep_step = 1.0
    plotter = None
    plotting_exclude_channels = [
        "EHE", "EHN", "EH1", "EH2", "HHE", "HHN", "HH1", "HH2"]
    """
    Real-Time tribe for real-time matched-filter detection.

    Parameters
    ----------
    name
        Tribe identifier - used to define save path.
    tribe
        Tribe of templates to use for detection.
    inventory
        Inventory of stations used for detection.
    rt_client
        Real-Time Client for streaming data.
    detect_interval
        Frequency to conduct detection. Must be less than buffer_capacity.
    plot
        Whether to generate the real-time bokeh plot
    plot_options
        Plotting options parsed to `rt_eqcorrscan.plotting.plot_buffer`
    wavebank
        WaveBank to save data to. Used for backfilling by RealTimeTribe.
        Set to `None` to not use a WaveBank.
    
    sleep_step
        Default sleep-step in seconds while waiting for data. Defaults to 1.0
    plotting_exclude_channels
        Channels to exclude from plotting
    """
    # Management of detection multi-processing
    process_cores = 2
    _parallel_processing = True  # This seems unstable for subprocessing.
    max_correlation_cores = None

    # Thread management
    lock = Lock()  # Lock for access to internals
    _running = False
    _detecting_thread = None
    _backfillers = dict()  # Backfill subprocesses
    _backfill_tribe = Tribe()  # Tribe of as-yet unused templates for backfilling
    _last_backfill_start = UTCDateTime.now()  # Time of last backfill run - update on run
    _number_of_backfillers = 0  # Book-keeping of backfiller processes.
    _clean_backfillers = True  # If false will leave temporary backfiller dirs

    _plugins = dict()  # Plugin subprocesses

    busy = False

    _simulation = False  # Flag to get extra output for simulations
    _speed_up = 1.0  # For simulated runs - do not change for real-time!
    _stream_end = UTCDateTime(1970, 1, 1)  # End of real-time data - will be
    # updated in first loop. Used for keeping track of when templates are relative
    # to the data received.
    _spoilers = True  # If the reactor gets ahead of the rt_match_filter (in
    # simulations) then templates from the future might be read in. Set this to
    # False to disallow templates from the future
    _max_wait_length = 60.
    _fig = None  # Cache figures to save memory
    _template_dir = "new_templates"  # Where new templates should be.
    _min_run_length = 24 * 3600  # Minimum run-length in seconds.
    # Usurped by max_run_length, used to set a threshold for rate calculation.
    __killfile = None  # Killfile pathname

    # WaveBank management
    wavebank_lock = Lock()
    has_wavebank = False

    # Plugin management
    plugins = []

    def __init__(
        self,
        name: str = None,
        tribe: Tribe = None,
        inventory: Inventory = None,
        rt_client: _StreamingClient = None,
        detect_interval: float = 60.,
        backfill_interval: float = 600.,
        plot: bool = True,
        plot_options: dict = None,
        wavebank: Union[str, WaveBank] = WaveBank("Streaming_WaveBank"),
        notifier: Notifier = Notifier(),
        plugin_config: dict = None,
    ) -> None:
        super().__init__(templates=tribe.templates)
        self.rt_client = rt_client
        assert (self.rt_client.buffer_capacity >= max(
            [template.process_length for template in self.templates]))
        assert (self.rt_client.buffer_capacity >= detect_interval)
        self.name = name or "RealTimeTribe"
        self.inventory = inventory
        self.party = Party()
        self.detect_interval = detect_interval
        self.backfill_interval = backfill_interval
        self.plot = plot
        self.notifier = notifier
        self.plot_options = {}
        self.plugin_config = plugin_config
        if plot_options is not None:
            self.plot_length = plot_options.get("plot_length", 300)
            self.plot_options.update({
                key: value for key, value in plot_options.items()
                if key != "plot_length"})
        self.detections = []
        self._killfile = f"kill_{self.name}_{id(self)}"

        # Wavebank status to avoid accessing the underlying, lockable, wavebank
        if isinstance(wavebank, str):
            wavebank = WaveBank(wavebank)
        self.__wavebank = wavebank
        if wavebank:
            self.has_wavebank = True
        self._wavebank_warned = False  # Reduce duplicate warnings

    def __repr__(self):
        """
        Print information about the tribe.

        .. rubric:: Example

        >>> from rt_eqcorrscan.streaming.clients.seedlink import RealTimeClient
        >>> rt_client = RealTimeClient(server_url="geofon.gfz-potsdam.de")
        >>> tribe = RealTimeTribe(
        ...     tribe=Tribe([Template(name='a', process_length=60)]),
        ...     rt_client=rt_client)
        >>> print(tribe) # doctest: +NORMALIZE_WHITESPACE
        Real-Time Tribe of 1 templates on client:
        Seed-link client at geofon.gfz-potsdam.de, status: Stopped, \
        buffer capacity: 600.0s
            Current Buffer:
        Buffer(0 traces, maxlen=600.0)
        """
        return 'Real-Time Tribe of {0} templates on client:\n{1}'.format(
            self.__len__(), self.rt_client)

    @property
    def _latitude(self):
        lats = []
        for network in self.inventory:
            for station in network:
                if station.latitude is not None:
                    lats.append(station.latitude)
        for template in self.templates:
            if template.event is None:
                continue
            if len(template.event.origins) == 0:
                continue
            ori = (template.event.preferred_origin() or
                   template.event.origins[-1])
            if ori.latitude is not None:
                lats.append(ori.latitude)
        return lats

    @property
    def _longitude(self):
        lons = []
        for network in self.inventory:
            for station in network:
                if station.longitude is not None:
                    lons.append(station.longitude)
        for template in self.templates:
            if template.event is None:
                continue
            if len(template.event.origins) == 0:
                continue
            ori = (template.event.preferred_origin() or
                   template.event.origins[-1])
            if ori.longitude is not None:
                lons.append(ori.longitude)
        return lons

    @property
    def minlat(self):
        return min(self._latitude)

    @property
    def minlon(self):
        return min(self._longitude)

    @property
    def maxlat(self):
        return max(self._latitude)

    @property
    def maxlon(self):
        return max(self._longitude)

    @property
    def _killfile(self):
        return self.__killfile

    @_killfile.setter
    def _killfile(self, killfile: str):
        self.__killfile = killfile
        Logger.info(f"To stop this RTTribe, run: touch "
                    f"{os.path.abspath(os.curdir)}/{killfile}")

    @property
    def running_template_dir(self) -> str:
        return f"{self.name}_running_templates"

    @property
    def template_seed_ids(self) -> set:
        """ Channel-ids used in the templates. """
        return set(tr.id for template in self.templates for tr in template.st)

    @property
    def used_seed_ids(self) -> set:
        """ Channel-ids in the inventory. """
        if self.inventory is None:
            return set()
        return set("{net}.{sta}.{loc}.{chan}".format(
            net=net.code, sta=sta.code, loc=chan.location_code, chan=chan.code)
                   for net in self.inventory for sta in net for chan in sta)

    @property
    def expected_seed_ids(self) -> set:
        """ ids of channels to be used for detection. """
        if self.inventory is None or len(self.inventory) == 0:
            return self.template_seed_ids
        return self.template_seed_ids.intersection(self.used_seed_ids)

    @property
    def used_stations(self) -> set:
        """ Set of station names used in detection. """
        if self.inventory is None:
            return set()
        return {sta.code for net in self.inventory for sta in net}

    @property
    def minimum_data_for_detection(self) -> float:
        """ Get the minimum required data length (in seconds) for detection. """
        return max(template.process_length for template in self.templates)

    @property
    def running_templates(self) -> set:
        """
        Get a set of the names of the running templates.

        Note that names are not guaranteed to be unique.
        """
        return {t.name for t in self.templates}

    @property
    def wavebank(self):
        return self.__wavebank

    @wavebank.setter
    def wavebank(self, wavebank: WaveBank):
        self.__wavebank = wavebank
        if wavebank:
            self.has_wavebank = True
        else:
            self.has_wavebank = False

    def _ensure_templates_have_enough_stations(self, min_stations):
        """ Remove templates that don't have enough stations. """
        self.templates = [
            t for t in self.templates
            if len({tr.stats.station for tr in t.st}.intersection(
                self.used_stations)) >= min_stations]

    def _remove_old_detections(self, endtime: UTCDateTime) -> None:
        """ Remove detections older than keep duration. Works in-place. """
        # Use a copy to avoid changing list while iterating
        for d in copy.copy(self.detections):
            if d.detect_time <= endtime:
                self.detections.remove(d)

    def _remove_unused_backfillers(
        self,
        trig_int: float,
        hypocentral_separation: float,
        earliest_detection_time: UTCDateTime,
        detect_directory: str,
        save_waveforms: bool,
        plot_detections: bool,
        **kwargs
    ):
        """ Expire unused back fill processes to release resources. """
        active_backfillers = dict()
        for backfiller_name, backfill_process in self._backfillers.items():
            if backfill_process.poll() is not None:
                Logger.info(f"Handling detections from {backfiller_name}")
                self._backfiller_return(
                    backfiller_name=backfiller_name,
                    trig_int=trig_int,
                    hypocentral_separation=hypocentral_separation,
                    earliest_detection_time=earliest_detection_time,
                    detect_directory=detect_directory,
                    save_waveforms=save_waveforms,
                    plot_detections=plot_detections)
                Logger.info(f"Cleaning backfiller {backfiller_name}")
                if os.path.isdir(backfiller_name):
                    if self._clean_backfillers:
                        shutil.rmtree(backfiller_name)
                else:
                    Logger.warning(f"Did not find backfiller temp dir {backfiller_name}")
            else:
                active_backfillers.update({backfiller_name: backfill_process})
        Logger.debug(f"There are {len(active_backfillers)} backfillers currently active")
        self._backfillers = active_backfillers

    def _access_wavebank(
        self,
        method: str,
        timeout: float = None,
        *args, **kwargs
    ):
        """
        Thread and process safe access to the wavebank.

        Multiple processes cannot access the underlying HDF5 file at the same
        time.  This method waits until access to the HDF5 file is available and

        Parameters
        ----------
        method
            Method of wavebank to call
        timeout
            Maximum time to try to get access to the file
        args
            Arguments passed to method
        kwargs
            Keyword arguments passed to method

        Returns
        -------
        Whatever should be returned by the method.
        """
        if not self.has_wavebank:
            if not self._wavebank_warned:
                Logger.error("No wavebank attached to streamer")
            return None
        timer, wait_step = 0.0, 0.5
        Logger.debug("Getting wavebank lock")
        with self.wavebank_lock:
            try:
                func = self.wavebank.__getattribute__(method)
            except AttributeError:
                Logger.error(f"No wavebank method named {method}")
                return None
            # Attempt to access the underlying wavebank
            out = None
            while timer < timeout:
                Logger.debug(f"Trying to call {method} on wavebank")
                tic = time.time()
                try:
                    out = func(*args, **kwargs)
                    break
                except (IOError, OSError) as e:
                    Logger.warning(f"Call to {method} failed due to {e}",
                                   exc_info=True)
                    time.sleep(wait_step)
                toc = time.time()
                timer += toc - tic
            else:
                Logger.error(
                    f"Waited {timer} s and could not access the wavebank "
                    f"due to {e}")
        return out

    def get_wavebank_stream(self, bulk: List[tuple]) -> Stream:
        """ processsafe get-waveforms-bulk call """
        st = self._access_wavebank(
            method="get_waveforms_bulk", timeout=120., bulk=bulk)
        return st

    def get_wavebank_files(self, bulk: List[tuple]) -> List[str]:
        """ processsafe way to get the file paths meeting bulk criteria """
        paths = []
        for _bulk in bulk:
            index = self._access_wavebank(
                method="read_index", timeout=120, network=_bulk[0],
                station=_bulk[1], location=_bulk[2], channel=_bulk[3],
                starttime=_bulk[4], endtime=_bulk[5])
            files = (str(self.wavebank.bank_path) + os.sep + index.path).unique()
            paths.extend(list(files))
        return paths

    def _backfiller_return(
        self,
        backfiller_name: str,
        trig_int: float,
        hypocentral_separation: float,
        earliest_detection_time: UTCDateTime,
        detect_directory: str,
        save_waveforms: bool,
        plot_detections: bool,
    ):
        """
        Handle finalisation of backfillers.

        Parameters
        ----------
        backfiller_name
        """
        if os.path.isfile(f"{backfiller_name}/party.tgz"):
            party = Party().read(f"{backfiller_name}/party.tgz")
        else:
            Logger.info("No party written by backfiller - no detections")
            return
        Logger.info(f"Read backfiller party of {len(party)} detections")
        if len(party) == 0:
            return
        Logger.info(f"Backfill handling: Trying to get lock - Lock status: {self.lock}")
        with self.lock:
            Logger.info(f"Backfill handling: Lock acquired - Lock status: {self.lock}")
            self._handle_detections(
                party, trig_int=trig_int,
                hypocentral_separation=hypocentral_separation,
                earliest_detection_time=earliest_detection_time,
                detect_directory=detect_directory,
                save_waveforms=save_waveforms,
                plot_detections=plot_detections, st=None, skip_existing=False,
                backfill_dir=backfiller_name)
            self._remove_old_detections(earliest_detection_time)
            Logger.info("Party now contains {0} detections".format(
                len(self.detections)))
        Logger.info(f"Backfill handling: Lock released - Lock status {self.lock}")
        return

    def _handle_detections(
        self,
        new_party: Party,
        trig_int: float,
        hypocentral_separation: float,
        earliest_detection_time: UTCDateTime,
        detect_directory: str,
        save_waveforms: bool,
        plot_detections: bool,
        st: Stream = None,
        skip_existing: bool = True,
        backfill_dir: str = None,
        **kwargs
    ) -> None:
        """
        Handle new detections - do all the additional post-detection processing

        Parameters
        ----------
        new_party
            The party of new detections
        trig_int
            Minimum inter-detection time in seconds.
        hypocentral_separation
            Maximum inter-event distance in km to consider detections as being
            duplicates.
        earliest_detection_time
            Earliest (oldest) detection-time to be kept.
        detect_directory
            The head directory to write to - will create
            "{detect_directory}/{year}/{julian day}" directories
        save_waveforms
            Whether to save the waveform for the detected event or not
        plot_detections
            Whether to plot the detection waveform or not
        st
            The stream the detection was made in - required for save_waveform
            and plot_detection.
        skip_existing
            Whether to skip detections already written to disk.
        backfill_dir
            Location of backfiller if these detections have come from a backfiller.
        """
        _detected_templates = [f.template.name for f in self.party]
        for family in new_party:
            if family is None:
                continue
            for d in family:
                d._calculate_event(template=family.template)
                Logger.debug(f"New detection at {d.detect_time}")
            # Cope with no picks and hence no origins - these events have to be removed
            family.detections = [d for d in family if len(d.event.origins)]
            if family.template.name not in _detected_templates:
                self.party.families.append(family)
            else:
                self.party.select(family.template.name).detections.extend(
                    family.detections)

        Logger.info("Removing duplicate detections")
        Logger.info(f"Party contained {len(self.party)} before decluster")
        if len(self.party) > 0:
            # TODO: Need to remove detections from disk that are removed in decluster
            self.party.decluster(
                trig_int=trig_int, timing="origin", metric="cor_sum",
                hypocentral_separation=hypocentral_separation)
        Logger.info("Completed decluster")
        Logger.info(f"Party contains {len(self.party)} after decluster")
        Logger.info("Writing detections to disk")

        # Cope with not being given a stream
        read_st = False
        if st is None and backfill_dir is None:
            read_st = True

        # TODO: Need a better way to keep track of written detections - unique keys for detections?
        # TODO: This is slow, and for Kaikoura, this is what stops it from running in real time
        for family in self.party:
            for detection in family:
                # TODO: this check doesn't necassarily work well - detections may be the same physical detection, but different Detection objects
                if detection in self.detections:
                    continue
                detect_file_base = _detection_filename(
                    detection=detection, detect_directory=detect_directory)
                _filename = f"{detect_file_base}.xml"
                if os.path.isfile(f"{detect_file_base}.xml") and skip_existing:
                    Logger.info(f"{_filename} exists, skipping")
                    continue
                Logger.debug(f"Writing detection at {detection.detect_time}")
                # TODO: Do not do this, let some other process work on making the waveforms.
                if read_st:
                    max_shift = (
                        max(tr.stats.endtime for tr in family.template.st) -
                        min(tr.stats.starttime for tr in family.template.st))
                    bulk = [
                        (tr.stats.network,
                         tr.stats.station,
                         tr.stats.location,
                         tr.stats.channel,
                         (detection.detect_time - 5),
                         (detection.detect_time + max_shift + 5))
                        for tr in family.template.st]
                    st = self.wavebank.get_waveforms_bulk(bulk)
                    st_read = True
                self._fig = _write_detection(
                    detection=detection,
                    detect_file_base=detect_file_base,
                    save_waveform=save_waveforms,
                    plot_detection=plot_detections, stream=st,
                    fig=self._fig, backfill_dir=backfill_dir,
                    detect_dir=detect_directory)
        Logger.info("Expiring old detections")
        # Empty self.detections
        self.detections.clear()
        for family in self.party:
            Logger.debug(f"Checking for {family.template.name}")
            family.detections = [
                d for d in family.detections
                if d.detect_time >= earliest_detection_time]
            Logger.debug(f"Appending {len(family)} detections")
            for detection in family:
                # Need to append rather than create a new object
                self.detections.append(detection)
        return

    def _plot(self) -> None:  # pragma: no cover
        """ Plot the data as it comes in. """
        from rt_eqcorrscan.plotting.plot_buffer import EQcorrscanPlot

        self._wait()
        plot_options = copy.deepcopy(self.plot_options)
        update_interval = plot_options.pop("update_interval", 100.)
        plot_height = plot_options.pop("plot_height", 800)
        plot_width = plot_options.pop("plot_width", 1500)
        offline = plot_options.pop("offline", False)
        self.plotter = EQcorrscanPlot(
            rt_client=self.rt_client, plot_length=self.plot_length,
            tribe=self, inventory=self.inventory,
            detections=self.detections,
            exclude_channels=self.plotting_exclude_channels,
            update_interval=update_interval, plot_height=plot_height,
            plot_width=plot_width, offline=offline,
            **plot_options)
        self.plotter.background_run()

    def _check_for_killfile(self):
        """ Check to see if the killfile for this RTTribe exists """
        if self._killfile is None:
            return False
        if os.path.isfile(self._killfile):
            Logger.info(f"Found killfile: {self._killfile}, stopping")
            return True
        return False

    def _wait(self, wait: float = None, detection_kwargs: dict = None) -> bool:
        """ Wait for `wait` seconds, or until all channels are available. """
        if self._check_for_killfile():
            self.stop()
            return False
        if wait is not None and wait <= 0:
            Logger.info(f"No fucking about - get back! (wait: {wait}")
            return True
        Logger.info("Waiting for data.")
        max_wait = min(self._max_wait_length, self.rt_client.buffer_capacity)
        wait_length = 0.
        while True:
            if self._check_for_killfile():
                self.stop()
                return False
            tic = time.time()
            if detection_kwargs:
                # Check on backfillers
                self._remove_unused_backfillers(**detection_kwargs)
            if UTCDateTime.now() - self._last_backfill_start >= (self.backfill_interval / self._speed_up) and \
                    len(self._backfill_tribe) and detection_kwargs:
                Logger.info(f"Starting backfilling with {len(self._backfill_tribe)} templates")
                self.backfill(templates=self._backfill_tribe.copy(), **detection_kwargs)
                # Empty the tribe
                self._backfill_tribe.templates = []
                self._last_backfill_start = UTCDateTime.now()
            # Wait until we have some data
            Logger.debug(
                "Waiting for data, currently have {0} channels of {1} "
                "expected channels".format(
                    len(self.rt_client.buffer_ids),
                    len(self.expected_seed_ids)))
            if detection_kwargs:
                self._add_templates_from_disk(
                    min_stations=detection_kwargs.get("min_station", None))
            else:
                new_tribe = self._read_templates_from_disk()
                if len(new_tribe) > 0:
                    self.templates.extend(new_tribe.templates)
            # Only sleep if this ran faster than sleep step
            iter_time = time.time() - tic
            sleep_step = self.sleep_step - iter_time
            Logger.debug(f"Iteration of wait took {iter_time}s. Sleeping for {sleep_step}s")
            if sleep_step > 0:
                time.sleep(sleep_step)
            toc_sleep = time.time()
            wait_length += (toc_sleep - tic)
            if wait is None:
                if len(self.rt_client.buffer_ids) >= len(self.expected_seed_ids):
                    break
                if wait_length >= max_wait:
                    Logger.warning(
                        "Starting operation without the full dataset")
                    break
            elif wait_length >= wait:
                break
            pass
        return True

    def _start_plugins(self):
        """ Start up any registered plugins. """
        if self.plugin_config is None:
            Logger.debug("No plugins configured")
            return
        for key, value in self.plugin_config.items():
            if value is None:
                continue
            if key not in REGISTERED_PLUGINS.keys() and key != "order":
                Logger.error(f"Plugin {key} is not known to RT-EQcorrscan. "
                             f"Known plugins: {REGISTERED_PLUGINS.keys()}")
                continue
            config_name = f"{key}-config-{self.name}.yml"
            Logger.info(f"Writing config file for {key} to {config_name}")
            value.write(config_name)
            plugin_args = ["-c", config_name]
            if self._simulation:
                plugin_args.append("--simulation")
            plugin_proc = run_plugin(key, plugin_args)
            self._plugins.update({key: plugin_proc})
        return

    def _stop_plugins(self):
        if self.plugin_config is None:
            Logger.debug("No plugins configured")
            return
        for key, proc in self._plugins.items():
            Logger.info(f"Stopping subprocess for plugin {key}")
            config = self.plugin_config[key]
            outdir = config.out_dir
            with open(f"{outdir}/poison", "w") as f:
                f.write(f"Poisoned at {time.time()}")
            # proc.terminate()
            Logger.info(f"{key} poisoned")

        return

    def _configure_plugins(self, in_dir: str):
        # Nuance of attribdict means that even key lookup doesn't work as
        # expected, so need to do try/except
        try:
            # Need to pop this from configs so that order doesn't get
            # run as a plugin
            order = self.plugin_config.pop("order")
        except KeyError:
            order = ORDERED_PLUGINS
        for plugin_name in order:
            # If plugin name is plot then out_dir should be in_dir
            config = self.plugin_config.get(plugin_name, None)
            if config is None:
                continue
            config.in_dir = in_dir
            if plugin_name in ["plotter"]:
                config.out_dir = in_dir
            else:
                config.out_dir = f"{self.name}/{plugin_name}_out"
            if plugin_name == "nll":
                # We need to set the bounds to be useful
                config.maxlat = (
                        self.maxlat + (0.1 * (self.maxlat - self.minlat)))
                config.minlat = (
                        self.minlat - (0.1 * (self.maxlat - self.minlat)))
                config.maxlon = (
                        self.maxlon - (0.1 * (self.maxlon - self.minlon)))
                config.minlon = (
                        self.minlon - (0.1 * (self.maxlon - self.minlon)))
            if plugin_name == "growclust" and "nll" in order:
                # If we are running both growclust and nonlinloc we can use the nll 3D grids
                if config.ttabsrc == "nllgrid":
                    nll_config = self.plugin_config.get('nll')
                    # Point to the location of the to-be-generated NonLinLoc config file.
                    # The growclust plugin should read from this and set the appropriate values.
                    config.nll_config_file = os.path.join(
                        nll_config.working_dir, os.path.basename(nll_config.infile))
            config.wavebank_dir = os.path.abspath(self.wavebank.bank_path)
            config.template_dir = os.path.abspath(
                self.running_template_dir)
            # Output of previous plugin as input to next
            in_dir = config.out_dir
        return in_dir

    def _start_streaming(self):
        if not self.rt_client.started:
            self.rt_client.start()
        if self.rt_client.can_add_streams:
            for tr_id in self.expected_seed_ids:
                self.rt_client.select_stream(
                    net=tr_id.split('.')[0], station=tr_id.split('.')[1],
                    selector=tr_id.split('.')[3])
        else:
            Logger.warning("Client already in streaming mode,"
                           " cannot add channels")
        if not self.rt_client.streaming:
            self.rt_client.background_run()
            Logger.info("Started real-time streaming")
        else:
            Logger.info("Real-time streaming already running")

    def _runtime_check(self, run_start, max_run_length):
        run_time = UTCDateTime.now() - run_start
        if max_run_length is None:
            Logger.info(
                f"Run time: {run_time:.2f}s, no maximum run length")
        else:
            Logger.info(
                f"Run time: {run_time:.2f}s, max_run_length: {max_run_length:.2f}s")
            if run_time > max_run_length:
                Logger.critical("Hit maximum run time, stopping.")
                self.stop()
                return False
        return True

    def run(
        self,
        threshold: float,
        threshold_type: str,
        trig_int: float,
        hypocentral_separation: float = None,
        min_stations: int = None,
        keep_detections: float = 86400,
        detect_directory: str = "{name}/detections",
        plot_detections: bool = True,
        save_waveforms: bool = True,
        max_run_length: float = None,
        minimum_rate: float = None,
        backfill_to: UTCDateTime = None,
        backfill_client = None,
        **kwargs
    ) -> Party:
        """
        Run the RealTimeTribe detection.

        Detections will be added to a party and returned when the detection
        is done. Detections will be stored in memory for up to `keep_detections`
        seconds.  Detections will also be written to individual files in the
        `detect_directory`.

        Parameters
        ----------
        threshold
            Threshold for detection
        threshold_type
            Type of threshold to use. See
            `eqcorrscan.core.match_filter.Tribe.detect` for options.
        trig_int
            Minimum inter-detection time in seconds.
        hypocentral_separation
            Maximum inter-event distance in km to consider detections as being
            duplicates.
        min_stations
            Minimum number of stations required to make a detection.
        keep_detections
            Duration to store detection in memory for in seconds.
        detect_directory
            Relative path to directory for detections. This directory will be
            created if it doesn't exist - tribe name will be appended to this
            string to give the directory name.
        plot_detections
            Whether to plot detections or not - plots will be saved to the
            `detect_directory` as png images.
        save_waveforms
            Whether to save waveforms of detections or not - waveforms
            will be saved in the `detect_directory` as miniseed files.
        max_run_length
            Maximum detection run time in seconds. Default is to run
            indefinitely.
        minimum_rate
            Stopping criteria: if the detection rate drops below this the
            detector will stop. If set to None, then the detector will run
            until `max_run_length`. Units: events per day
        backfill_to
            Time to backfill the data buffer to.
        backfill_client
            Client to use to backfill the data buffer.

        Returns
        -------
        The party created - will not contain detections expired by
        `keep_detections` threshold.
        """
        # Update backfill start time
        self._last_backfill_start = UTCDateTime.now()
        restart_interval = 600.0
        # Squash duplicate channels to avoid excessive channels
        self.templates = [squash_duplicates(t) for t in self.templates]
        # Reshape the templates first
        if len(self.templates) > 0:
            self.templates = reshape_templates(
                templates=self.templates, used_seed_ids=self.expected_seed_ids)
        else:
            Logger.error("No templates, will not run")
            return Party()
        # Remove templates that do not have enough stations in common with the
        # inventory.
        min_stations = min_stations or 0
        n = len(self.templates)
        self._ensure_templates_have_enough_stations(min_stations=min_stations)
        n -= len(self.templates)
        Logger.info(
            f"{n} templates were removed because they did not share enough "
            f"stations with the inventory. {len(self.templates)} will be used")
        if len(self.templates) == 0:
            Logger.critical("No templates remain, not running")
            return Party()
        # Fix unsupported args
        try:
            if kwargs.pop("plot"):
                Logger.info("EQcorrscan plotting disabled")
        except KeyError:
            pass
        _cores = kwargs.get("cores", None)
        if _cores:
            self.max_correlation_cores = min(
                kwargs.pop("cores"), self.max_correlation_cores)
        run_start = UTCDateTime.now()
        detection_iteration = 0  # Counter for number of detection loops run
        if not self.busy:
            self.busy = True
        detect_directory = detect_directory.format(name=self.name)
        if not os.path.isdir(detect_directory):
            os.makedirs(detect_directory)

        # dump templates to the record of templates running
        if not os.path.isfile(self.running_template_dir):
            os.makedirs(self.running_template_dir)
        for template in self.templates:
            _tout = f"{self.running_template_dir}/{template.name}.pkl"
            Logger.info(f"Writing template to {_tout}")
            with open(_tout, "wb") as f:
                pickle.dump(template, f)
        # Get this locally before streaming starts
        buffer_capacity = self.rt_client.buffer_capacity  
        # Start the streamer
        self._start_streaming()

        # Add config options for plugins as needed
        in_dir = detect_directory
        if self.plugin_config:
            plugin_out_dir = self._configure_plugins(in_dir=in_dir)
            # Start any plugins
            self._start_plugins()

        Logger.info("Detection will use the following data: {0}".format(
            self.expected_seed_ids))
        if backfill_client and backfill_to:
            backfill = Stream()
            # Wait for the streamer to have some data
            _continue = self._wait()
            if not _continue:
                return Party()
            _buffer = self.rt_client.stream
            for tr_id in self.expected_seed_ids:
                try:
                    tr_in_buffer = _buffer.select(id=tr_id).merge()[0]
                except IndexError:
                    continue
                # Overlap - request more data than we are likely to get
                endtime = tr_in_buffer.stats.endtime + 120
                Logger.info(f"Buffer for {tr_in_buffer.id} between "
                            f"{tr_in_buffer.stats.starttime} and "
                            f"{tr_in_buffer.stats.endtime}")
                if endtime - backfill_to > buffer_capacity:
                    starttime = endtime - buffer_capacity
                    Logger.info(
                        f"Truncating backfill to buffer length: "
                        f"{buffer_capacity} (getting data between {starttime} "
                        f"and {endtime}")
                else:
                    starttime = backfill_to
                if starttime > endtime:
                    continue
                try:
                    tr = backfill_client.get_waveforms(
                        *tr_id.split('.'), starttime=starttime,
                        endtime=endtime).merge(method=1)[0]
                except Exception as e:
                    Logger.error("Could not back fill due to: {0}".format(e),
                                 exc_info=True)
                    Logger.error(f"The request was: {tr_id.split('.')},"
                                 f" starttime={starttime}, endtime={endtime}")
                    continue
                Logger.debug("Downloaded backfill: {0}".format(tr))
                backfill += tr
            # for tr in backfill:
            #     # Get the lock!
            #     Logger.info(f"Adding {tr.id} {tr.stats.starttime} -- "
            #                 f"{tr.stats.endtime} to the buffer")
            #     self.rt_client.on_data(tr)
            # Logger.info("Stream in buffer is now: {0}".format(
            #     self.rt_client.stream))
            # Use this to get around adding to rt client from another thread
            past_st = backfill
        else:
            past_st = Stream()
        if self.plot:  # pragma: no cover
            # Set up plotting thread
            self._plot()
            Logger.info("Plotting thread started")
        if not self.rt_client.buffer_length >= self.minimum_data_for_detection:
            sleep_step = (
                self.minimum_data_for_detection -
                self.rt_client.buffer_length + 5) / self._speed_up
            Logger.info("Sleeping for {0:.2f}s while accumulating data".format(
                sleep_step))
            _continue = self._wait(sleep_step)
            if not _continue:
                return Party()
        first_data = min([tr.stats.starttime
                          for tr in self.rt_client.stream.merge(method=1)])
        detection_kwargs = dict(
            threshold=threshold, threshold_type=threshold_type,
            trig_int=trig_int, hypocentral_separation=hypocentral_separation,
            keep_detections=keep_detections, detect_directory=detect_directory,
            plot_detections=plot_detections, save_waveforms=save_waveforms,
            maximum_backfill=first_data, endtime=None,
            min_stations=min_stations, earliest_detection_time=None)

        long_runs, long_run_time = 0, 0  # Keep track of over-running loops
        try:
            while self.busy:
                try:
                    self._running = True  # Lock tribe
                    start_time = UTCDateTime.now()
                    st = self.rt_client.stream.split().merge(method=1)
                    Logger.info(f"RTTribe received this stream from client:\n{st.__str__(extended=True)}")
                    # Add in past data if needed - will be trimmed later
                    Logger.info(f"Past stream is:\n{past_st.__str__(extended=True)}")
                    st = (st + past_st).merge(method=1)  # Keep overlapping data
                    Logger.info(f"Adding st to past_st results in:\n{st.__str__(extended=True)}")
                    # Warn if data are gappy
                    gappy = False
                    for tr in st:
                        if np.ma.is_masked(tr.data):
                            gappy = True
                            gaps = tr.split().get_gaps()
                            Logger.warning(
                                f"Masked data found on {tr.id}. Gaps: {gaps}")
                    if gappy:
                        st = st.merge(method=1)  # Re-merge after gap checking
                    if self.has_wavebank:
                        st = _check_stream_is_int(st)
                        wb_retries, wb_max_retries, e = 0, 10, None
                        while wb_retries <= wb_max_retries:
                            try:
                                Logger.info(f"Putting stream into wavebank: "
                                            f"{st.__str__(extended=True)}")
                                self._access_wavebank(
                                    method="put_waveforms", timeout=10.,
                                    stream=st)
                                break
                            except Exception as e:
                                Logger.info(f"Retry {wb_retries}/{wb_max_retries}")
                                wb_retries += 1
                                # time.sleep(0.1)
                                continue
                        else:
                            Logger.error(
                                f"Could not write to wavebank due to {e}",
                                exc_info=True)
                    last_data_received = self.rt_client.last_data
                    # Split to remove trailing mask
                    if len(st) == 0:
                        Logger.warning("No data")
                        continue
                    elif last_data_received is None:
                        Logger.warning(
                            "Streamer incorrectly reported None for last "
                            "data received, setting to stream end")
                        last_data_received = max(tr.stats.endtime for tr in st)
                    Logger.info(
                        f"Streaming Client last received data at "
                        f"{last_data_received}")
                    self._stream_end = max(tr.stats.endtime for tr in st)
                    min_stream_end = min(tr.stats.endtime for tr in st)
                    # Update detection kwargs endtime to end of current data -
                    # no need to backfill beyond that
                    detection_kwargs["endtime"] = self._stream_end
                    Logger.info(
                        "Real-time client provided data: \n"
                        f"{st.__str__(extended=True)}")
                    # Cope with data that doesn't come
                    if start_time - last_data_received > restart_interval:
                        Logger.warning(
                            "The streaming client has not given any new "
                            f"data for {restart_interval} seconds. Restarting"
                            " Streaming client")
                        Logger.info(
                            f"start_time: {start_time}, last_data_received: "
                            f"{last_data_received}, "
                            f"stream_end: {self._stream_end}")
                        Logger.info("Stopping streamer")
                        self.rt_client.background_stop()
                        self.rt_client.stop()
                        # Get a clean instance just in case
                        Logger.info("Starting streamer")
                        self._start_streaming()
                        Logger.info("Streamer started")
                        st = self.rt_client.stream.split().merge(method=1)  # Get data again.
                    Logger.info("Streaming client seems healthy")
                    # Remove any data that shouldn't be there - sometimes GeoNet's
                    # Seedlink client gives old data.
                    Logger.info(
                        f"Trimming between {self._stream_end - (buffer_capacity + 20.0)} "
                        f"and {self._stream_end}")
                    st.trim(
                        starttime=self._stream_end - (buffer_capacity + 20.0),
                        endtime=self._stream_end)
                    if detection_iteration > 0:
                        # For the first run we want to detect in everything we have.
                        # Otherwise trim so that all channels have at-least minimum data for detection
                        st.trim(
                            starttime=min_stream_end - self.minimum_data_for_detection,
                            endtime=self._stream_end)
                    Logger.info("Trimmed data")
                    if len(st) == 0:
                        Logger.warning("No data")
                        continue
                    # Remove short channels
                    st.traces = [
                        tr for tr in st
                        if _numpy_len(tr.data) >= (
                                .8 * self.minimum_data_for_detection)]
                    if len(st) == 0:
                        Logger.error("Insufficient data after trimming, accumulating")
                        continue
                    Logger.info("Starting detection run")
                    # merge again - checking length can split the data?
                    st = st.merge(method=1)
                    Logger.info("Using data: \n{0}".format(
                        st.__str__(extended=True)))
                    try:
                        Logger.debug("Currently have {0} templates in tribe".format(
                            len(self)))
                        new_party = self.detect(
                            stream=st, plot=False, threshold=threshold,
                            threshold_type=threshold_type, trig_int=trig_int,
                            xcorr_func="fftw", concurrency="concurrent",
                            cores=self.max_correlation_cores,
                            process_cores=self.process_cores,
                            parallel_process=self._parallel_processing,
                            ignore_bad_data=True, copy_data=False,
                            concurrent_processing=False,
                            **kwargs)
                        Logger.info("Completed detection")
                    except Exception as e:  # pragma: no cover
                        Logger.error(e, exc_info=True)
                        if "Cannot allocate memory" in str(e):
                            Logger.error(
                                "Out of memory, stopping this detector")
                            self.stop()
                            break
                        if not self._runtime_check(
                                run_start=run_start,
                                max_run_length=max_run_length):
                            break
                        Logger.info(
                            "Waiting for {0:.2f}s and hoping this gets "
                            "better".format(self.detect_interval))
                        time.sleep(self.detect_interval)
                        continue
                    Logger.info(
                        f"Trying to get lock - Lock status: {self.lock}")
                    detection_kwargs.update(
                        dict(earliest_detection_time=self._stream_end - keep_detections))
                    with self.lock:
                        Logger.info(f"Lock acquired - Lock status: {self.lock}")
                        if len(new_party) > 0:
                            self._handle_detections(
                                new_party, st=st,
                                **detection_kwargs)
                        self._remove_old_detections(
                            self._stream_end - keep_detections)
                        Logger.info("Party now contains {0} detections".format(
                            len(self.detections)))
                    Logger.info(f"Lock released - Lock status {self.lock}")
                    self._running = False  # Release lock
                    run_time = (UTCDateTime.now() - start_time) * self._speed_up  # Work in fake time
                    Logger.info("Detection took {0:.2f}s".format(run_time))
                    if self.detect_interval <= run_time:
                        long_run_time += run_time
                        long_runs += 1
                        if long_runs > 10:
                            long_run_time /= long_runs
                            # NEVER EXCEED THE BUFFER LENGTH!!!!
                            new_detect_interval = min(self.rt_client.buffer_length - 10, long_run_time + 10)
                            Logger.warning(
                                "detect_interval {0:.2f} shorter than run-time for 10 occasions"
                                "{1:.2f}, increasing detect_interval to {2:.2f}".format(
                                    self.detect_interval, run_time, new_detect_interval))
                            self.detect_interval = new_detect_interval
                            long_runs, long_run_time = 0, 0  # Reset counters
                    Logger.info("Iteration {0} took {1:.2f}s total".format(
                        detection_iteration, run_time))
                    Logger.info("Waiting {0:.2f}s until next run".format(
                        self.detect_interval - run_time))
                    detection_iteration += 1
                    _continue = self._wait(
                        wait=(self.detect_interval - run_time) / self._speed_up,  # Convert to real-time
                        detection_kwargs=detection_kwargs)
                    _runtime_continue = self._runtime_check(
                        run_start=run_start, max_run_length=max_run_length)
                    if not _continue or not _runtime_continue:
                        self.stop()
                        break
                    if minimum_rate and UTCDateTime.now() > run_start + self._min_run_length:
                        _rate = average_rate(
                            self.detections,
                            starttime=max(
                                self._stream_end - keep_detections, first_data),
                            endtime=self._stream_end)
                        Logger.info(f"Average rate:\t{_rate}, "
                                    f"minimum rate:\t{minimum_rate}")
                        if _rate < minimum_rate:
                            Logger.critical(
                                "Rate ({0:.2f}) has dropped below minimum rate, "
                                "stopping.".format(_rate))
                            self.stop()
                            break
                    # Re-use this stream
                    past_st = st
                    # Logger.info("Enforcing garbage collection")
                    # gc.collect()
                    # Memory output
                    # Logger.info("Working out memory use")
                    # sum1 = summary.summarize(muppy.get_objects())
                    # for line in summary.format_(sum1):
                    #     Logger.info(line)
                    # Total memory used for process according to psutil
                    # total_memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                    # Logger.info(f"Total memory used by {os.getpid()}: {total_memory_mb:.4f} MB")
                except Exception as e:
                    # Error locally and mail this in!
                    Logger.critical(f"Uncaught error: {e}", exc_info=True)

                    message = f"""\
                    Uncaught error: {e}
                    
                    Traceback:
                    {traceback.format_exc()}
                    """
                    self.notifier.notify(content=message)

                    if not self._runtime_check(
                            run_start=run_start, max_run_length=max_run_length):
                        break
        finally:
            Logger.critical("Stopping")
            self.stop()
        return self.party

    def _read_templates_from_disk(self):
        template_files = glob.glob(f"{self._template_dir}/*")
        if len(template_files) == 0:
            return []
        Logger.info(f"Checking for events in {self._template_dir}")
        new_tribe = Tribe()
        for template_file in template_files:
            if os.path.isdir(template_file):
                # Can happen if the archive hasn't finished being created
                continue
            Logger.info(f"Reading from {template_file}")
            try:
                template = Template().read(template_file)
            except Exception as e:
                Logger.error(f"Could not read {template_file} due to {e}",
                             exc_info=True)
                continue
            template_endtime = max(tr.stats.endtime for tr in template.st)
            if template_endtime > self._stream_end:
                msg = (f"Template {template_file} ends at {template_endtime}, "
                       f"after now: {self._stream_end}")
                if self._spoilers:
                    Logger.warning(msg)
                else:
                    Logger.debug(msg)
                    continue  # Skip and do not remove template, we will get it later
            # If we got to here we can add the template to the tribe and remove the file.
            new_tribe += template
            if os.path.isfile(template_file):
                os.remove(template_file)  # Remove file once done with it.
        new_tribe.templates = [t for t in new_tribe
                               if t.name not in self.running_templates]
        if len(new_tribe):
            Logger.info(f"Read in {len(new_tribe)} new templates from disk")

        # dump them to the record of templates running
        if not os.path.isdir(self.running_template_dir):
            os.makedirs(self.running_template_dir)
        for template in new_tribe:
            _tout = f"{self.running_template_dir}/{template.name}.pkl"
            Logger.info(f"Writing template to {_tout}")
            with open(_tout, "wb") as f:
                pickle.dump(template, f)
        return new_tribe

    def _add_templates_from_disk(
        self,
        min_stations: int,
    ):
        new_tribe = self._read_templates_from_disk()
        if len(new_tribe) == 0:
            return
        Logger.info(
            f"Adding {len(new_tribe)} templates to already running tribe.")
        self.add_templates(new_tribe, min_stations=min_stations)

    def add_templates(
        self,
        templates: Union[List[Template], Tribe],
        min_stations: int = None,
    ) -> set:
        """
        Add templates to the tribe.

        This method will run the new templates back in time, then append the
        templates to the already running tribe.

        Parameters
        ----------
        templates
            New templates to add to the tribe.
        min_stations:
            Minimum number of stations required to make a detection.

        Returns
        -------
            Complete set of template names after addition
        """
        while self._running:
            Logger.info("Waiting for access to tribe in add_templates")
            time.sleep(1)  # Wait until the lock is released
        if isinstance(templates, list):
            new_tribe = Tribe(templates)
        else:
            new_tribe = templates
        # Squash duplicate channels to avoid excessive channels
        new_tribe.templates = [squash_duplicates(t) for t in new_tribe.templates]
        # Reshape
        new_tribe.templates = reshape_templates(
            templates=new_tribe.templates, used_seed_ids=self.expected_seed_ids)
        # Remove templates that do not have enough stations.
        min_stations = min_stations or 0
        n = len(new_tribe)
        new_tribe.templates = [
            t for t in new_tribe.templates
            if len({tr.stats.station for tr in t.st}.intersection(
                self.used_stations)) >= min_stations]
        n -= len(new_tribe)
        Logger.info(
            f"{n} templates were removed because they did not have enough "
            f"stations. {len(new_tribe)} will be added to the running tribe.")
        self.templates.extend(new_tribe.templates)
        # Add templates to backfill set.
        self._backfill_tribe.templates.extend(new_tribe.templates)
        return set(t.name for t in self.templates)

    def backfill(
        self,
        templates: Union[List[Template], Tribe],
        threshold: float,
        threshold_type: str,
        trig_int: float,
        maximum_backfill: Union[float, UTCDateTime] = None,
        endtime: UTCDateTime = None,
        plot_detections: bool = False,
        **kwargs
    ) -> None:
        """
        Backfill using data from rt_client's wavebank.

        This method will run the new templates through old data and record
        detections in the real-time-tribe.

        Parameters
        ----------
        templates
            New templates to add to the tribe.
        threshold
            Threshold for detection
        threshold_type
            Type of threshold to use. See
            `eqcorrscan.core.match_filter.Tribe.detect` for options.
        trig_int
            Minimum inter-detection time in seconds.
        maximum_backfill
            Time in seconds to backfill to - if this is larger than the
            difference between the time now and the time that the tribe
            started, then it will backfill to when the tribe started.
        endtime
            Time to stop the backfill, if None will run to now.
        """
        # Get the stream - Only let the main process get the Stream
        Logger.info("Acquiring stream from wavebank")
        endtime = endtime or UTCDateTime.now()
        if maximum_backfill is not None:
            if isinstance(maximum_backfill, (float, int)):
                starttime = endtime - maximum_backfill
            elif isinstance(maximum_backfill, UTCDateTime):
                starttime = maximum_backfill
            else:
                Logger.warning(
                    f"maximum_backfill is {type(maximum_backfill)}, not float "
                    "or UTCDateTime, starting from 0")
                starttime = UTCDateTime(0)
        else:
            starttime = UTCDateTime(0)
        Logger.info(f"Backfilling between {starttime} and {endtime}")
        if starttime >= endtime or not self.has_wavebank:
            Logger.info("No data meets backfill needs. Returning")
            return
        if self.expected_seed_ids and len(self.expected_seed_ids) > 0:
            bulk = []
            for chan in self.expected_seed_ids:
                query = chan.split('.')
                query.extend([starttime, endtime])
                bulk.append(tuple(query))
        else:
            Logger.warning("No expected seed ids")
            return
        if len(bulk) == 0:
            Logger.warning("No bulk")
            return
        Logger.debug(f"Getting stations for backfill: {bulk}")
        # st = self.get_wavebank_stream(bulk)
        st_files = self.get_wavebank_files(bulk)
        Logger.info(f"Concatenating {len(st_files)} stream files for backfill")
        
        self._number_of_backfillers += 1

        backfiller_name = f"Backfiller_{self._number_of_backfillers}"

        # Make working directory and write files.
        if not os.path.isdir(backfiller_name):
            os.makedirs(backfiller_name, exist_ok=True)

        # Just copy all the files to a streams folder and use a LocalClient for backfiller
        os.makedirs(f"{backfiller_name}/streams")
        for st_file in st_files:
            st_file_new_path = st_file.split(str(self.wavebank.bank_path))[-1]
            st_file_new_path = f"{backfiller_name}/streams/{st_file_new_path}"
            os.makedirs(os.path.split(st_file_new_path)[0], exist_ok=True)
            # shutil.copyfile(st_file, st_file_new_path)
            os.link(st_file, st_file_new_path)  # Link, rather than copy

        # st.write(f"{backfiller_name}/stream.ms", format="MSEED")
        # with open(f"{backfiller_name}/stream.ms", "wb") as fout:
        #     for st_file in st_files:
        #         with open(st_file, "rb") as fin:
        #             fout.write(fin.read())

        if isinstance(templates, Tribe):
            tribe = templates
        else:
            tribe = Tribe(templates)
        tribe.write(f"{backfiller_name}/tribe.tgz")

        del st_files
        # Force garbage collection before creating new process
        gc.collect()

        working_dir = os.path.abspath(backfiller_name)

        # Start backfiller subprocess
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reactor", "backfill.py")
        _call = [
            "python", script_path,
            "-w", working_dir,
            "-m", str(self.minimum_data_for_detection),
            "-t", str(threshold),
            "-T", threshold_type,
            "-i", str(trig_int),
            "-c", str(self.max_correlation_cores),
            "--starttime", str(starttime),
            "--endtime", str(endtime),
            "-P",  # Enable parallel processing
        ]
        if plot_detections:
            _call.append("--plot")
        _call.append("-s") # Add on the list of expected seed ids
        _call.extend(self.expected_seed_ids)

        Logger.info("Running `{call}`".format(call=" ".join(_call)))
        proc = subprocess.Popen(_call)
        self._backfillers.update({backfiller_name: proc})
        Logger.info("Backfill process started, returning")
        return

    def stop(self, write_stopfile: bool = False) -> None:
        """
        Stop the real-time system.
       
        Parameters
        ----------
        write_stopfile:
            Used to write a one-line file telling listening systems that
            this has stopped. Used by the Reactor.
        """
        if self.plotter is not None:  # pragma: no cover
            self.plotter.background_stop()
        self.rt_client.background_stop()
        self.busy = False
        self._running = False
        if self._detecting_thread is not None:
            self._detecting_thread.join()
        # Kill all the backfillers
        for backfiller in self._backfillers.values():
            backfiller.kill()
        if self._clean_backfillers:
            for backfiller_name in self._backfillers.keys():
                if os.path.isdir(backfiller_name):
                    shutil.rmtree(backfiller_name)
        if write_stopfile:
            with open(".stopfile", "a") as f:
                f.write(f"{self.name}\n")
        # Stop plugins
        self._stop_plugins()


def reshape_templates(
    templates: List[Template],
    used_seed_ids: Iterable[str]
) -> List[Template]:
    """
    Reshape templates to have the full set of required channels (and no more).

    This is done within matched-filter as well as here - we do it here so that
    the templates are not reshaped every iteration.

    Parameters
    ----------
    templates
        Templates to be reshaped - the templates are changed in-place
    used_seed_ids
        Seed-ids used for detection (network.station.location.channel)

    Returns
    -------
    Templates filled for detection.
    """
    template_streams = [t.st for t in templates]
    template_names = [t.name for t in templates]

    samp_rate = template_streams[0][0].stats.sampling_rate
    process_len = max(t.process_length for t in templates)
    # Make a dummy stream with all the used seed ids
    stream = Stream()
    for seed_id in used_seed_ids:
        net, sta, loc, chan = seed_id.split('.')
        tr = Trace(header=dict(
            network=net, station=sta, location=loc, channel=chan,
            sampling_rate=samp_rate),
            data=numpy.empty(int(process_len * samp_rate)))
        stream += tr

    _, template_streams, template_names = _prep_data_for_correlation(
        stream=stream, templates=template_streams,
        template_names=template_names, force_stream_epoch=False)
    templates_back = []
    for template_st, template_name in zip(template_streams, template_names):
        original = [t for t in templates if t.name == template_name]
        assert len(original) == 1
        original = original[0]
        original.st = template_st
        templates_back.append(original)
    return templates_back


def squash_duplicates(template: Template):
    """
    Remove duplicate channels in templates. 

    This happens when there are duplicate picks, and it fucks shit up.
    More explicitly (less?): when there are duplicate picks, duplicate
    channels appear in the template, which are then extended to all
    templates meaning that many copies of a template are run, resulting
    in excessive detection bias on one channel, and very expensive templates
    without good reason.
    """
    from collections import Counter
    # Check if there are duplicate channels first
    seed_ids = Counter(tr.id for tr in template.st)
    if seed_ids.most_common(1)[0][1] == 1:
        # Do nothing, no duplicates
        return template
    unique_template_st = Stream()
    unique_event_picks = []
    for seed_id, repeats in seed_ids.most_common():
        # TODO: this restricts to only picks on that seed id - but we could just have matched picks on station
        seed_id_picks = [p for p in template.event.picks 
                         if p.waveform_id.get_seed_string() == seed_id 
                         and p.phase_hint[0] in "PS"]
        if repeats == 1:
            unique_template_st += template.st.select(id=seed_id)
            unique_event_picks.append(seed_id_picks[0])
            continue
        # Now we get to doing something - get the stream and the picks
        repeat_stream = template.st.select(id=seed_id)
        # Get the unique start-times
        unique_traces = {tr.stats.starttime.datetime: tr 
                         for tr in repeat_stream.traces}
        if len(unique_traces) == 1:
            unique_template_st += unique_traces.popitem()[1]
            unique_event_picks.append(seed_id_picks[0])
            continue
        # If there are more than one unique start-time then we need to 
        # find the appropriate pick for each
        unique_picks = {f"{p.phase_hint}_{p.time}": p for p in seed_id_picks}
        for pick in unique_picks.values():
            expected_starttime = pick.time - template.prepick
            tr = unique_traces.get(expected_starttime.datetime, None)
            if tr:
                unique_event_picks.append(pick)
                unique_template_st += tr
            else:
                Logger.debug(f"No trace for pick at {pick.time}")
    template.st = unique_template_st
    template.event.picks = unique_event_picks
    return template


def _detection_filename(
        detection: Detection,
        detect_directory: str,
) -> str:
    _path = os.path.join(
        detect_directory, detection.detect_time.strftime("%Y"),
        detection.detect_time.strftime("%j"))
    if not os.path.isdir(_path):
        os.makedirs(_path)
    _filename = os.path.join(
        _path, detection.detect_time.strftime("%Y%m%dT%H%M%S"))
    return _filename


def _write_detection(
    detection: Detection,
    detect_file_base: str,
    save_waveform: bool,
    plot_detection: bool,
    stream: Stream,
    fig=None,
    backfill_dir: str = None,
    detect_dir: str = None
) -> Figure:
    """
    Handle detection writing including writing streams and figures.

    Parameters
    ----------
    detection
        The Detection to write
    detect_file_base
        File to write to (without extension)
    save_waveform
        Whether to save the waveform for the detected event or not
    plot_detection
        Whether to plot the detection waveform or not
    stream
        The stream the detection was made in - required for save_waveform and
        plot_detection.
    fig
        A figure object to reuse.
    backfill_dir:
        Backfill directory - set if the detections have already been written
        to this dir and just need to be copied.
    detect_dir
        Detection directory - only used to manipulate backfillers.

    Returns
    -------
    An empty figure object to be reused if a figure was created, or the figure
    passed to it.
    """
    from rt_eqcorrscan.plotting.plot_event import plot_event

    if backfill_dir:
        backfill_file_base = (
            f"{backfill_dir}/detections/{detect_file_base.split(detect_dir)[-1]}")
        Logger.info(f"Looking for detections in {backfill_file_base}.*")
        backfill_dets = glob.glob(f"{backfill_file_base}.*")
        Logger.info(f"Copying {len(backfill_dets)} to main detections")
        for f in backfill_dets:
            ext = os.path.splitext(f)[-1]
            shutil.copyfile(f, f"{detect_file_base}{ext}")
            Logger.info(f"Copied {f} to {detect_file_base}{ext}")
        return fig

    try:
        detection.event.write(f"{detect_file_base}.xml", format="QUAKEML")
    except Exception as e:
        Logger.error(f"Could not write event file due to {e}", exc_info=True)
    else:
        Logger.info(f"Written detection at {detection.detect_time} to "
                    f"{detect_file_base}")
    detection.event.picks.sort(key=lambda p: p.time)
    st = stream.slice(
        detection.event.picks[0].time - 10,
        detection.event.picks[-1].time + 20).copy()
    if plot_detection:
        # Make plot
        fig = plot_event(fig=fig, event=detection.event, st=st,
                         length=90, show=False)
        try:
            fig.savefig(f"{detect_file_base}.png")
        except Exception as e:
            Logger.error(f"Could not write plot due to {e}", exc_info=True)
        fig.clf()
    if save_waveform:
        st = _check_stream_is_int(st)
        try:
            st.write(f"{detect_file_base}.ms", format="MSEED")
        except Exception as e:
            Logger.error(f"Could not write stream due to {e}", exc_info=True)
    return fig


def _check_stream_is_int(st):
    st = st.split()
    for tr in st:
        # Ensure data are int32, see https://github.com/obspy/obspy/issues/2683
        if tr.data.dtype == numpy.int32 and \
                tr.data.dtype.type != numpy.int32:
            tr.data = tr.data.astype(numpy.int32, subok=False)
        if tr.data.dtype.type == numpy.intc:
            tr.data = tr.data.astype(numpy.int32, subok=False)
    return st


def _numpy_len(arr: Union[numpy.ndarray, numpy.ma.MaskedArray]) -> int:
    """
    Convenience function to return the length of a numpy array.

    If arr is a masked array will return the count of the non-masked elements.

    Parameters
    ----------
    arr
        Array to get the length of - must be 1D

    Returns
    -------
        Length of non-masked elements
    """
    assert arr.ndim == 1, "Only supports 1D arrays."
    if numpy.ma.is_masked(arr):
        return arr.count()
    return arr.shape[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
