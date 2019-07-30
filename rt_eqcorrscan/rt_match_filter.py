"""
Classes for real-time matched-filter detection of earthquakes.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import time
import os
import logging
import copy
import numpy
import gc

# from pympler import summary, muppy

from typing import Union

from obspy import Stream, UTCDateTime, Inventory
from matplotlib.figure import Figure
from eqcorrscan import Tribe, Template, Party, Family, Detection

from rt_eqcorrscan.streaming.streaming import _StreamingClient
from rt_eqcorrscan.config.notification import Notifier

Logger = logging.getLogger(__name__)


class RealTimeTribe(Tribe):
    sleep_step = 1.0
    plotter = None
    exclude_channels = ["EHE", "EHN", "EH1", "EH2", "HHE", "HHN", "HH1", "HH2"]
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
    exclude_channels
        Channels to exclude from plotting
    """
    notifier = Notifier()
    _running = False
    _speed_up = 1.0
    # Speed-up for simulated runs - do not change for real-time!
    _max_wait_length = 60.

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

        >>> from rt_eqcorrscan.streaming import RealTimeClient
        >>> rt_client = RealTimeClient(server_url="geofon.gfz-potsdam.de")
        >>> tribe = RealTimeTribe(
        ...     tribe=Tribe([Template(name='a', process_length=60)]),
        ...     rt_client=rt_client)
        >>> print(tribe) # doctest: +NORMALIZE_WHITESPACE
        Real-Time Tribe of 1 templates on client:
        Seed-link client at geofon.gfz-potsdam.de, status: Stopped, \
        buffer capacity: 600.0s
            Current Buffer:
        0 Trace(s) in Stream:
        <BLANKLINE>
        """
        return 'Real-Time Tribe of {0} templates on client:\n{1}'.format(
            self.__len__(), self.rt_client)

    @property
    def template_channels(self) -> set:
        """ Channel-ids used in the templates. """
        return set(tr.id for template in self.templates for tr in template.st)

    @property
    def used_stations(self) -> set:
        """ Channel-ids in the inventory. """
        if self.inventory is None:
            return set()
        return set("{net}.{sta}.{loc}.{chan}".format(
            net=net.code, sta=sta.code, loc=chan.location_code, chan=chan.code)
                   for net in self.inventory for sta in net for chan in sta)

    @property
    def expected_channels(self) -> set:
        """ ids of channels to be used for detection. """
        if self.inventory is None or len(self.inventory) == 0:
            return self.template_channels
        return self.template_channels.intersection(self.used_stations)

    @property
    def minimum_data_for_detection(self) -> float:
        """ Get the minimum required data length (in seconds) for detection. """
        return max(template.process_length for template in self.templates)

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
            detections=self.detections, exclude_channels=self.exclude_channels,
            update_interval=update_interval, plot_height=plot_height,
            plot_width=plot_width, offline=offline,
            **plot_options)
        self.plotter.background_run()

    def _wait(self) -> None:
        Logger.info("Waiting for data.")
        max_wait = min(self._max_wait_length, self.rt_client.buffer_capacity)
        wait_length = 0.
        while len(self.rt_client.buffer) < len(self.expected_channels):
            if wait_length >= max_wait:
                Logger.warning("Starting plotting without the full dataset")
                break
            # Wait until we have some data
            Logger.debug(
                "Waiting for data, currently have {0} channels of {1} "
                "expected channels".format(
                    len(self.rt_client.buffer), len(self.expected_channels)))
            wait_length += self.sleep_step
            time.sleep(self.sleep_step)
            pass
        return

    def stop(self) -> None:
        """ Stop the real-time system. """
        if self.plotter is not None:  # pragma: no cover
            self.plotter.background_stop()
        self.rt_client.background_stop()
        self._running = False

    def run(
        self,
        threshold: float,
        threshold_type: str,
        trig_int: float,
        keep_detections: float = 86400,
        detect_directory: str = "detections",
        plot_detections: bool = True,
        save_waveforms: bool = True,
        max_run_length: float = None,
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
        backfill_to
            Time to backfill the data buffer to.
        backfill_client
            Client to use to backfill the data buffer.

        Returns
        -------
        The party created - will not contain detections expired by
        `keep_detections` threshold.
        """
        run_start = UTCDateTime.now()
        detection_iteration = 0  # Counter for number of detection loops run
        self._running = True
        fig = None
        # Used for plotting - figure is reused to avoid memory leaks.

        last_possible_detection = UTCDateTime(0)  # TODO: Why is this here?
        detect_directory += self.name
        if not os.path.isdir(detect_directory):
            os.makedirs(detect_directory)
        if not self.rt_client.started:
            self.rt_client.start()
        if self.rt_client.can_add_streams:
            for tr_id in self.expected_channels:
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
            self.expected_channels))
        if backfill_client and backfill_to:
            self._wait()
            for tr_id in self.expected_channels:
                try:
                    tr_in_buffer = self.rt_client.buffer.select(id=tr_id)[0]
                except IndexError:
                    continue
                endtime = tr_in_buffer.stats.starttime
                tr = backfill_client.get_waveforms(
                    *tr_id.split('.'), starttime=backfill_to,
                    endtime=endtime).merge()[0]
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
            time.sleep(sleep_step)
        try:
            while self._running:
                start_time = UTCDateTime.now()
                st = self.rt_client.get_stream().merge()
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
                Logger.debug(st)
                Logger.info("Starting detection run")
                try:
                    new_party = self.detect(
                        stream=st, plotvar=False, threshold=threshold,
                        threshold_type=threshold_type, trig_int=trig_int,
                        xcorr_func="fftw", concurrency="concurrent",
                        **kwargs)
                except Exception as e:  # pragma: no cover
                    Logger.error(e)
                    Logger.info(
                        "Waiting for {0:.2f}s and hoping this gets "
                        "better".format(self.detect_interval))
                    time.sleep(self.detect_interval)
                    continue
                if len(new_party) > 0:
                    for family in new_party:
                        if family is None:
                            continue
                        _family = Family(
                            template=family.template, detections=[])
                        for detection in family:
                            if detection.detect_time > last_possible_detection:
                                # TODO: lag-calc and relative magnitudes?
                                _family += detection
                        self.party += _family
                    Logger.info("Removing duplicate detections")
                    self.party.decluster(trig_int=trig_int)
                    # Remove old detections here
                    for family in self.party:
                        family.detections = [
                            d for d in family.detections
                            if d.detect_time >= last_data - keep_detections]
                    for f in self.party:
                        for d in f:
                            if d not in self.detections:
                                # Need to append rather than create a new object
                                fig = _write_detection(
                                    detection=d,
                                    detect_directory=detect_directory,
                                    save_waveform=save_waveforms,
                                    plot_detection=plot_detections,
                                    stream=st, fig=fig)
                                self.detections.append(d)
                                self.notifier.notify(
                                    message="Made detection at {0}".format(
                                        d.detect_time),
                                    level=2)
                Logger.info("Party now contains {0} detections".format(
                    len(self.detections)))
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
                time.sleep((self.detect_interval - run_time) / self._speed_up)
                # TODO: Needs a min-rate stop condition.
                if max_run_length and UTCDateTime.now() > run_start + max_run_length:
                    self.stop()
                gc.collect()
                # sum1 = summary.summarize(muppy.get_objects())
                # summary.print_(sum1)
        finally:
            self.stop()
        return self.party


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
    detection.event.write("{0}.xml".format(_filename), format="QUAKEML")
    detection.event.picks.sort(key=lambda p: p.time)
    st = stream.slice(
        detection.event.picks[0].time, detection.event.picks[-1].time).copy()
    if save_waveform:
        st.split().write("{0}.ms".format(_filename), format="MSEED")
    if plot_detection:
        # Make plot
        fig = plot_event(fig=fig, event=detection.event, st=st.merge(),
                         length=90, show=False)
        fig.savefig("{0}.png".format(_filename))
        fig.clf()
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
