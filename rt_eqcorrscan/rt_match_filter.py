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

# from pympler import summary, muppy

from typing import Union, List

from obspy import Stream, UTCDateTime, Inventory
from matplotlib.figure import Figure
from eqcorrscan import Tribe, Template, Party, Detection

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
    process_cores = 2
    _parallel_processing = True  # This seems unstable for subprocessing.
    _running = False
    _detecting_thread = None
    busy = False
    _speed_up = 1.0
    # Speed-up for simulated runs - do not change for real-time!
    _max_wait_length = 60.
    _fig = None
    _tribe_file = "tribe.tgz"
    _min_run_length = 3600  # Minimum run-length in seconds.
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
        self.buffer = Stream()
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

    def _handle_detections(
        self,
        new_party: Party,
        trig_int: float,
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
        # TODO: Decluster on pick time? Find matching picks and calc median
        #  pick time difference.
        if len(self.party) > 0:
            self.party.decluster(
                trig_int=trig_int, timing="origin", metric="cor_sum")
        Logger.info("Completed decluster")
        for family in self.party:
            family.detections = [
                d for d in family.detections if d.detect_time >= endtime]
        Logger.info("Writing detections to disk")
        for f in self.party:
            for d in f:
                if d in self.detections:
                    continue
                Logger.info(f"Writing detection: {d.detect_time}")
                self._fig = _write_detection(
                    detection=d,
                    detect_directory=detect_directory,
                    save_waveform=save_waveforms,
                    plot_detection=plot_detections, stream=st,
                    fig=self._fig)
                # Need to append rather than create a new object
                self.detections.append(d)

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
            # Wait until we have some data
            Logger.debug(
                "Waiting for data, currently have {0} channels of {1} "
                "expected channels".format(
                    len(self.rt_client.buffer), len(self.expected_seed_ids)))
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
                if len(self.rt_client.buffer) >= len(self.expected_seed_ids):
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
        # Remove templates that do not have enough stations in common with the
        # inventory.
        min_stations = min_stations or 0
        n = len(self.templates)
        self._ensure_templates_have_enough_stations(min_stations=min_stations)
        n -= len(self.templates)
        Logger.info(
            f"{n} templates were removed because they did not share enough "
            f"stations with the inventory. {len(self.templates)} will be used")
        try:
            if kwargs.pop("plot"):
                Logger.info("EQcorrscan plotting disabled")
        except KeyError:
            pass
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
            self._wait()
            for tr_id in self.expected_seed_ids:
                try:
                    tr_in_buffer = self.rt_client.buffer.select(id=tr_id)[0]
                except IndexError:
                    continue
                endtime = tr_in_buffer.stats.starttime
                try:
                    tr = backfill_client.get_waveforms(
                        *tr_id.split('.'), starttime=backfill_to,
                        endtime=endtime).merge()[0]
                except Exception as e:
                    Logger.error("Could not back fill due to: {0}".format(e))
                    continue
                Logger.debug("Downloaded backfill: {0}".format(tr))
                self.rt_client.on_data(tr)
                Logger.debug("Stream in buffer is now: {0}".format(
                    self.rt_client.buffer))
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
                          for tr in self.rt_client.get_stream().merge()])
        detection_kwargs = dict(
            threshold=threshold, threshold_type=threshold_type,
            trig_int=trig_int, keep_detections=keep_detections,
            detect_directory=detect_directory, plot_detections=plot_detections,
            save_waveforms=save_waveforms, maximum_backfill=first_data,
            endtime=None, min_stations=min_stations)
        try:
            while self.busy:
                self._running = True  # Lock tribe
                start_time = UTCDateTime.now()
                st = self.rt_client.get_stream().split().merge()
                # Split to remove trailing mask
                if len(st) == 0:
                    Logger.warning("No data")
                    continue
                # Cope with data that doesn't come
                last_data = max(tr.stats.endtime for tr in st)
                if detection_iteration > 0:
                    # For the first run we want to detect in everything we have.
                    st.trim(
                        starttime=last_data - self.minimum_data_for_detection,
                        endtime=last_data)
                # Remove short channels
                st.traces = [
                    tr for tr in st
                    if _numpy_len(tr.data) >= (.8 * self.minimum_data_for_detection)]
                Logger.info("Starting detection run")
                Logger.debug("Using data: \n{0}".format(st.__str__(extended=True)))
                try:
                    Logger.debug("Currently have {0} templates in tribe".format(
                        len(self)))
                    new_party = self.detect(
                        stream=st, plot=False, threshold=threshold,
                        threshold_type=threshold_type, trig_int=trig_int,
                        xcorr_func="fftw", concurrency="concurrent",
                        process_cores=self.process_cores,
                        parallel_process=self._parallel_processing,
                        ignore_bad_data=True, **kwargs)
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
                self._handle_detections(
                    new_party, trig_int=trig_int,
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
        if not os.path.isfile(self._tribe_file):
            return []
        Logger.info(f"Checking for events in {self._tribe_file}")
        new_tribe = Tribe().read(self._tribe_file)
        os.remove(self._tribe_file)  # Remove file once done with it.
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
        maximum_backfill: float = None,
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
        maximum_backfill: float = None,
        endtime: UTCDateTime = None,
        **kwargs
    ) -> set:
        """
        Add templates to the tribe.

        This method will run the new templates back in time, then append the
        templates to the already running tribe.

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
            time.sleep(1)  # Wait until the lock is released
        if isinstance(templates, list):
            new_tribe = Tribe(templates)
        else:
            new_tribe = templates
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
        # Get the stream
        endtime = endtime or UTCDateTime.now()
        if maximum_backfill is not None:
            starttime = endtime - maximum_backfill
        else:
            starttime = UTCDateTime(0)
        if starttime >= endtime or self.rt_client.wavebank is None:
            return
        bulk = [tuple(chan.split('.').extend([starttime, endtime]))
                for chan in self.expected_seed_ids]
        st = self.rt_client.wavebank.get_waveforms_bulk(bulk)
        Logger.debug("Additional templates to be run: \n{0} "
                     "templates".format(len(new_tribe)))
        # TODO: This could be in a non-blocking Process of it's own
        new_party = new_tribe.detect(
            stream=st, plot=False, threshold=threshold,
            threshold_type=threshold_type, trig_int=trig_int,
            xcorr_func="fftw", concurrency="concurrent",
            parallel_process=self._parallel_processing,
            process_cores=self.process_cores, **kwargs)
        while self._running:
            time.sleep(1)  # Wait until lock is released to add detections
        self._handle_detections(
            new_party=new_party, detect_directory=detect_directory,
            endtime=endtime - keep_detections, plot_detections=plot_detections,
            save_waveforms=save_waveforms, st=st, trig_int=trig_int)
        return set(t.name for t in self.templates)

    def stop(self) -> None:
        """ Stop the real-time system. """
        if self.plotter is not None:  # pragma: no cover
            self.plotter.background_stop()
        self.rt_client.background_stop()
        self.busy = False
        self._running = False
        if self._detecting_thread is not None:
            self._detecting_thread.join()


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
