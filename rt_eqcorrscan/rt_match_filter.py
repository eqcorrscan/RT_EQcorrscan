"""
Classes for real-time matched-filter detection of earthquakes.
"""
import time
import traceback
import os
import logging
import copy
import numpy
import gc
import glob

# from pympler import summary, muppy

from typing import Union, List, Iterable

from obspy import Stream, UTCDateTime, Inventory, Trace
from matplotlib.figure import Figure
from multiprocessing import Process, Lock
from eqcorrscan import Tribe, Template, Party, Detection
from eqcorrscan.utils.pre_processing import _prep_data_for_correlation

from rt_eqcorrscan.streaming.streaming import _StreamingClient
from rt_eqcorrscan.event_trigger.triggers import average_rate

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
    _backfillers = []  # Backfill processes
    _number_of_backfillers = 0  # Book-keeping of backfiller processes.
    busy = False

    _speed_up = 1.0  # For simulated runs - do not change for real-time!
    _max_wait_length = 60.
    _fig = None  # Cache figures to save memory
    _template_dir = "new_templates"  # Where new templates should be.
    _min_run_length = 24 * 3600  # Minimum run-length in seconds.
    # Usurped by max_run_length, used to set a threshold for rate calculation.

    def __init__(
        self,
        name: str = None,
        tribe: Tribe = None,
        inventory: Inventory = None,
        rt_client: _StreamingClient = None,
        detect_interval: float = 60.,
        plot: bool = True,
        plot_options: dict = None,
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
        self.plot = plot
        self.plot_options = {}
        if plot_options is not None:
            self.plot_length = plot_options.get("plot_length", 300)
            self.plot_options.update({
                key: value for key, value in plot_options.items()
                if key != "plot_length"})
        self.detections = []

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

    def _remove_unused_backfillers(self):
        """ Expire unused back fill processes to release resources. """
        active_backfillers = []
        for backfill_process in self._backfillers:
            if not backfill_process.is_alive():
                backfill_process.join()
                backfill_process.close()
            else:
                active_backfillers.append(backfill_process)
        self._backfillers = active_backfillers

    def _handle_detections(
        self,
        new_party: Party,
        trig_int: float,
        hypocentral_separation: float,
        endtime: UTCDateTime,
        detect_directory: str,
        save_waveforms: bool,
        plot_detections: bool,
        st: Stream,
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
        endtime
            Last detection-time to be kept.
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
        """
        for family in new_party:
            if family is None:
                continue
            for d in family:
                d._calculate_event(template=family.template)
            self.party += family
        Logger.info("Removing duplicate detections")
        if len(self.party) > 0:
            self.party.decluster(
                trig_int=trig_int, timing="origin", metric="cor_sum",
                hypocentral_separation=hypocentral_separation)
        Logger.info("Completed decluster")
        Logger.info("Writing detections to disk")
        for family in self.party:
            for detection in family:
                if detection in self.detections:
                    continue
                Logger.info(f"Writing detection: {detection.detect_time}")
                self._fig = _write_detection(
                    detection=detection,
                    detect_directory=detect_directory,
                    save_waveform=save_waveforms,
                    plot_detection=plot_detections, stream=st,
                    fig=self._fig)
        Logger.info("Expiring old detections")
        for family in self.party:
            Logger.debug(f"Checking for {family.template.name}")
            family.detections = [
                d for d in family.detections if d.detect_time >= endtime]
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

    def _wait(self, wait: float = None, detection_kwargs: dict = None) -> None:
        """ Wait for `wait` seconds, or until all channels are available. """
        Logger.info("Waiting for data.")
        max_wait = min(self._max_wait_length, self.rt_client.buffer_capacity)
        wait_length = 0.
        while True:
            tic = time.time()
            # Check on backfillers
            self._remove_unused_backfillers()
            # Wait until we have some data
            Logger.debug(
                "Waiting for data, currently have {0} channels of {1} "
                "expected channels".format(
                    len(self.rt_client.buffer_ids),
                    len(self.expected_seed_ids)))
            if detection_kwargs:
                self._add_templates_from_disk(**detection_kwargs)
            else:
                new_tribe = self._read_templates_from_disk()
                if len(new_tribe) > 0:
                    self.templates.extend(new_tribe.templates)
            time.sleep(self.sleep_step)
            toc = time.time()
            wait_length += (toc - tic)
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
        return

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
        backfill_client=None,
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
        # Reshape the templates first
        if len(self.templates) > 0:
            self.templates = reshape_templates(
                templates=self.templates, used_seed_ids=self.expected_seed_ids)
        else:
            Logger.error("No templates, will not run")
            return
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
        if not self.rt_client.busy:
            self.rt_client.background_run()
            Logger.info("Started real-time streaming")
        else:
            Logger.info("Real-time streaming already running")
        Logger.info("Detection will use the following data: {0}".format(
            self.expected_seed_ids))
        if backfill_client and backfill_to:
            backfill = Stream()
            self._wait()
            _buffer = self.rt_client.stream
            for tr_id in self.expected_seed_ids:
                try:
                    tr_in_buffer = _buffer.select(id=tr_id)[0]
                except IndexError:
                    continue
                endtime = tr_in_buffer.stats.starttime
                if endtime - backfill_to > self.rt_client.buffer_capacity:
                    Logger.info("Truncating backfill to buffer length")
                    starttime = endtime - self.rt_client.buffer_capacity
                else:
                    starttime = backfill_to
                try:
                    tr = backfill_client.get_waveforms(
                        *tr_id.split('.'), starttime=starttime,
                        endtime=endtime).merge()[0]
                except Exception as e:
                    Logger.error("Could not back fill due to: {0}".format(e))
                    continue
                Logger.debug("Downloaded backfill: {0}".format(tr))
                backfill += tr
            for tr in backfill:
                # Get the lock!
                Logger.debug(f"Adding {tr.id} to the buffer")
                self.rt_client.on_data(tr)
            Logger.debug("Stream in buffer is now: {0}".format(
                self.rt_client.stream))
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
            self._wait(sleep_step)
        first_data = min([tr.stats.starttime
                          for tr in self.rt_client.stream.merge()])
        detection_kwargs = dict(
            threshold=threshold, threshold_type=threshold_type,
            trig_int=trig_int, hypocentral_separation=hypocentral_separation,
            keep_detections=keep_detections, detect_directory=detect_directory,
            plot_detections=plot_detections, save_waveforms=save_waveforms,
            maximum_backfill=first_data, endtime=None,
            min_stations=min_stations)
        try:
            while self.busy:
                self._running = True  # Lock tribe
                start_time = UTCDateTime.now()
                st = self.rt_client.stream.split().merge()
                # Split to remove trailing mask
                if len(st) == 0:
                    Logger.warning("No data")
                    continue
                # Cope with data that doesn't come
                last_data = max(tr.stats.endtime for tr in st)
                # Remove any data that shouldn't be there - sometimes GeoNet's
                # Seedlink client gives old data.
                st.trim(
                    starttime=last_data - (self.rt_client.buffer_capacity + 20.0),
                    endtime=last_data)
                if detection_iteration > 0:
                    # For the first run we want to detect in everything we have.
                    st.trim(
                        starttime=last_data - self.minimum_data_for_detection,
                        endtime=last_data)
                if len(st) == 0:
                    Logger.warning("No data")
                    continue
                # Remove short channels
                st.traces = [
                    tr for tr in st
                    if _numpy_len(tr.data) >= (
                            .8 * self.minimum_data_for_detection)]
                Logger.info("Starting detection run")
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
                        ignore_bad_data=True, **kwargs)
                    Logger.info("Completed detection")
                except Exception as e:  # pragma: no cover
                    Logger.error(e)
                    Logger.error(traceback.format_exc())
                    if "Cannot allocate memory" in str(e):
                        Logger.error("Out of memory, stopping this detector")
                        self.stop()
                        break
                    Logger.info(
                        "Waiting for {0:.2f}s and hoping this gets "
                        "better".format(self.detect_interval))
                    time.sleep(self.detect_interval)
                    continue
                Logger.info(f"Trying to get lock - lock status: {self.lock}")
                with self.lock:
                    if len(new_party) > 0:
                        self._handle_detections(
                            new_party, trig_int=trig_int,
                            hypocentral_separation=hypocentral_separation,
                            endtime=last_data - keep_detections,
                            detect_directory=detect_directory,
                            save_waveforms=save_waveforms,
                            plot_detections=plot_detections, st=st)
                    self._remove_old_detections(last_data - keep_detections)
                    Logger.info("Party now contains {0} detections".format(
                        len(self.detections)))
                self._running = False  # Release lock
                run_time = UTCDateTime.now() - start_time
                Logger.info("Detection took {0:.2f}s".format(run_time))
                if self.detect_interval <= run_time:
                    Logger.warning(
                        "detect_interval {0:.2f} shorter than run-time "
                        "{1:.2f}, increasing detect_interval to {2:.2f}".format(
                            self.detect_interval, run_time, run_time + 10))
                    self.detect_interval = run_time + 10
                Logger.debug("This step took {0:.2f}s total".format(run_time))
                Logger.info("Waiting {0:.2f}s until next run".format(
                    self.detect_interval - run_time))
                detection_iteration += 1
                self._wait(
                    wait=(self.detect_interval - run_time) / self._speed_up,
                    detection_kwargs=detection_kwargs)
                if max_run_length and UTCDateTime.now() > run_start + max_run_length:
                    Logger.critical("Hit maximum run time, stopping.")
                    self.stop()
                    break
                if minimum_rate and UTCDateTime.now() > run_start + self._min_run_length:
                    _rate = average_rate(
                        self.detections,
                        starttime=max(last_data - keep_detections, first_data),
                        endtime=last_data)
                    if _rate < minimum_rate:
                        Logger.critical(
                            "Rate ({0:.2f}) has dropped below minimum rate, "
                            "stopping.".format(_rate))
                        self.stop()
                        break
                gc.collect()
                # Memory output
                # sum1 = summary.summarize(muppy.get_objects())
                # summary.print_(sum1)
        finally:
            self.stop()
        return self.party

    def _read_templates_from_disk(self):
        template_files = glob.glob(f"{self._template_dir}/*")
        if len(template_files) == 0:
            return []
        Logger.info(f"Checking for events in {self._template_dir}")
        new_tribe = Tribe()
        for template_file in template_files:
            Logger.debug(f"Reading from {template_file}")
            try:
                new_tribe += Template().read(template_file)
            except Exception as e:
                Logger.error(f"Could not read {template_file} due to {e}")
            os.remove(template_file)  # Remove file once done with it.
        new_tribe.templates = [t for t in new_tribe
                               if t.name not in self.running_templates]
        Logger.info(f"Read in {len(new_tribe)} new templates from disk")
        return new_tribe

    def _add_templates_from_disk(
        self,
        threshold: float,
        threshold_type: str,
        trig_int: float,
        min_stations: int,
        keep_detections: float = 86400,
        detect_directory: str = "{name}/detections",
        plot_detections: bool = True,
        save_waveforms: bool = True,
        maximum_backfill: Union[float, UTCDateTime] = None,
        endtime: UTCDateTime = None,
        **kwargs
    ):
        new_tribe = self._read_templates_from_disk()
        if len(new_tribe) == 0:
            return
        Logger.info(
            f"Adding {len(new_tribe)} templates to already running tribe.")
        self.add_templates(
            new_tribe, threshold=threshold, threshold_type=threshold_type,
            trig_int=trig_int, min_stations=min_stations,
            keep_detections=keep_detections, detect_directory=detect_directory,
            plot_detections=plot_detections, save_waveforms=save_waveforms,
            maximum_backfill=maximum_backfill, endtime=endtime, **kwargs)

    def add_templates(
        self,
        templates: Union[List[Template], Tribe],
        threshold: float,
        threshold_type: str,
        trig_int: float,
        min_stations: int = None,
        keep_detections: float = 86400,
        detect_directory: str = "{name}/detections",
        plot_detections: bool = True,
        save_waveforms: bool = True,
        maximum_backfill: Union[float, UTCDateTime] = None,
        endtime: UTCDateTime = None,
        **kwargs
    ) -> set:
        """
        Add templates to the tribe.

        This method will run the new templates back in time, then append the
        templates to the already running tribe.

        # TODO - make standard parameters and wrap methods to decorate them
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
        min_stations:
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
        maximum_backfill
            Time in seconds to backfill to - if this is larger than the
            difference between the time now and the time that the tribe
            started, then it will backfill to when the tribe started.
        endtime
            Time to stop the backfill, if None will run to now.

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
        self.backfill(
            templates=new_tribe, threshold=threshold,
            threshold_type=threshold_type, trig_int=trig_int,
            keep_detections=keep_detections, detect_directory=detect_directory,
            plot_detections=plot_detections, save_waveforms=save_waveforms,
            maximum_backfill=maximum_backfill, endtime=endtime, **kwargs)
        return set(t.name for t in self.templates)

    def backfill(
        self,
        templates: Union[List[Template], Tribe],
        threshold: float,
        threshold_type: str,
        trig_int: float,
        hypocentral_separation: float = None,
        keep_detections: float = 86400,
        detect_directory: str = "{name}/detections",
        plot_detections: bool = True,
        save_waveforms: bool = True,
        maximum_backfill: Union[float, UTCDateTime] = None,
        endtime: UTCDateTime = None,
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
        hypocentral_separation
            Maximum inter-event distance in km to consider detections as being
            duplicates.
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
        maximum_backfill
            Time in seconds to backfill to - if this is larger than the
            difference between the time now and the time that the tribe
            started, then it will backfill to when the tribe started.
        endtime
            Time to stop the backfill, if None will run to now.
        """
        self._number_of_backfillers += 1
        backfill_process = Process(
            target=self._backfill,
            args=(templates, threshold, threshold_type, trig_int,
                  hypocentral_separation, keep_detections, detect_directory,
                  plot_detections, save_waveforms, maximum_backfill, endtime),
            kwargs=kwargs, name=f"Backfiller_{self._number_of_backfillers}")
        backfill_process.start()
        self._backfillers.append(backfill_process)

    def _backfill(
        self,
        templates: Union[List[Template], Tribe],
        threshold: float,
        threshold_type: str,
        trig_int: float,
        hypocentral_separation: float = None,
        keep_detections: float = 86400,
        detect_directory: str = "{name}/detections",
        plot_detections: bool = True,
        save_waveforms: bool = True,
        maximum_backfill: float = None,
        endtime: UTCDateTime = None,
        **kwargs
    ) -> None:
        """ Background backfill method """
        if isinstance(templates, Tribe):
            new_tribe = templates
        else:
            new_tribe = Tribe(templates)
        Logger.info(f"Backfilling with {len(new_tribe)} templates")
        # Get the stream
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
        if starttime >= endtime or not self.rt_client.has_wavebank:
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
        Logger.info(f"Getting stations for backfill: {bulk}")
        st = self.rt_client.get_wavebank_stream(bulk)
        Logger.debug("Additional templates to be run: \n{0} "
                     "templates".format(len(new_tribe)))

        Logger.info("Starting backfill detection run with:")
        Logger.info(st.__str__(extended=True))
        new_party = new_tribe.detect(
            stream=st, plot=False, threshold=threshold,
            threshold_type=threshold_type, trig_int=trig_int,
            xcorr_func="fftw", concurrency="concurrent",
            cores=self.max_correlation_cores,
            parallel_process=self._parallel_processing,
            process_cores=self.process_cores, **kwargs)
        detect_directory = detect_directory.format(name=self.name)
        Logger.info("Backfill detection completed - handling detections")
        if len(new_party) > 0:
            Logger.info(f"Lock status: {self.lock}")
            with self.lock:  # The only time the state is altered
                Logger.info(f"Lock status: {self.lock}")
                self._handle_detections(
                    new_party=new_party, detect_directory=detect_directory,
                    endtime=endtime - keep_detections,
                    plot_detections=plot_detections,
                    save_waveforms=save_waveforms, st=st, trig_int=trig_int,
                    hypocentral_separation=hypocentral_separation)
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
        if write_stopfile:
            with open(".stopfile", "a") as f:
                f.write(f"{self.name}\n")


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


def _write_detection(
    detection: Detection,
    detect_directory: str,
    save_waveform: bool,
    plot_detection: bool,
    stream: Stream,
    fig=None,
) -> Figure:
    """
    Handle detection writing including writing streams and figures.

    Parameters
    ----------
    detection
        The Detection to write
    detect_directory
        The head directory to write to - will create
        "{detect_directory}/{year}/{julian day}" directories
    save_waveform
        Whether to save the waveform for the detected event or not
    plot_detection
        Whether to plot the detection waveform or not
    stream
        The stream the detection was made in - required for save_waveform and
        plot_detection.
    fig
        A figure object to reuse.

    Returns
    -------
    An empty figure object to be reused if a figure was created, or the figure
    passed to it.
    """
    from rt_eqcorrscan.plotting.plot_event import plot_event

    _path = os.path.join(
        detect_directory, detection.detect_time.strftime("%Y/%j"))
    if not os.path.isdir(_path):
        os.makedirs(_path)
    _filename = os.path.join(
        _path, detection.detect_time.strftime("%Y%m%dT%H%M%S"))
    detection.event.write(f"{_filename}.xml", format="QUAKEML")
    detection.event.picks.sort(key=lambda p: p.time)
    st = stream.slice(
        detection.event.picks[0].time - 10,
        detection.event.picks[-1].time + 20).copy()
    if plot_detection:
        # Make plot
        fig = plot_event(fig=fig, event=detection.event, st=st,
                         length=90, show=False)
        fig.savefig(f"{_filename}.png")
        fig.clf()
    if save_waveform:
        st.split().write(f"{_filename}.ms", format="MSEED")
    return fig


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
