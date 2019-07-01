"""
Classes for real-time matched-filter detection of earthquakes.

:copyright:
    Calum Chamberlain

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import time
import os
import logging

from obspy import Stream, UTCDateTime
from eqcorrscan import Tribe, Template, Party, Family

from rt_eqcorrscan.utils.seedlink import RealTimeClient
from rt_eqcorrscan.plotting.plot_buffer import EQcorrscanPlot

Logger = logging.getLogger(__name__)


class RealTimeTribe(Tribe):
    sleep_step = 1.0
    plotter = None
    exclude_channels = ["EHE", "EHN", "EH1", "EH2", "HHE", "HHN", "HH1", "HH2"]
    """
    Real-Time tribe.

    :type tribe: `eqcorrscan.core.match_filter.Tribe
    :param tribe: Tribe of templates to use for detection.
    :type inventory: :class: `obspy.core.station.Inventory`
    :param inventory: Inventory of stations used for detection.
    :type server_url: str
    :param server_url: Address of seedlink client.
    :type buffer_capacity: float
    :param buffer_capacity:
        Length of data buffer in memory in seconds. Must be longer than the
        process_len of the Tribe.
    :type detect_interval: float
    :param detect_interval:
        Frequency to conduct detection. Must be less than buffer_capacity.
    """
    def __init__(self, tribe=None, inventory=None, server_url=None,
                 buffer_capacity=600, detect_interval=60,
                 plot=True, **plot_options):
        super().__init__(templates=tribe.templates)
        assert (buffer_capacity >= max(
            [template.process_length for template in self.templates]))
        assert (buffer_capacity >= detect_interval)
        self.buffer = Stream()
        self.inventory = inventory
        self.party = Party()
        self.busy = True
        self.detect_interval = detect_interval
        self.client = RealTimeClient(
            server_url=server_url, autoconnect=True, buffer=self.buffer,
            buffer_capacity=buffer_capacity)
        self.plot = plot
        self.plot_length = plot_options.get("plot_length", 300)
        self.plot_options = {
            key: value for key, value in plot_options.items()
            if key != "plot_length"}
        self.detections = []

    def __repr__(self):
        """
        Print information about the tribe.

        .. rubric:: Example

        >>> tribe = RealTimeTribe(
        ...     tribe=Tribe([Template(name='a', process_length=60)]),
        ...     server_url="geofon.gfz-potsdam.de")
        >>> print(tribe) # doctest: +NORMALIZE_WHITESPACE
        Real-Time Tribe of 1 templates on client:
        Seed-link client at geofon.gfz-potsdam.de, status: Stopped, \
        buffer capacity: 600s
            Current Buffer:
        0 Trace(s) in Stream:
        <BLANKLINE>
        """
        return 'Real-Time Tribe of {0} templates on client:\n{1}'.format(
            self.__len__(), self.client)

    @property
    def template_channels(self):
        return set(tr.id for template in self.templates for tr in template.st)

    @property
    def used_stations(self):
        return set("{net}.{sta}.{loc}.{chan}".format(
            net=net.code, sta=sta.code, loc=chan.location_code, chan=chan.code)
                   for net in self.inventory for sta in net for chan in sta)

    @property
    def expected_channels(self):
        return self.template_channels.intersection(self.used_stations)

    def _plot(self):
        """Plot the data as it comes in."""
        wait_length = 0
        while len(self.client.buffer) < len(self.expected_channels):
            if wait_length >= self.detect_interval:
                Logger.warning("Starting plotting without the full dataset")
                break
            # Wait until we have some data
            Logger.debug(
                "Waiting for data, currently have {0} channels of {1} "
                "expected channels".format(
                    len(self.client.buffer), len(self.expected_channels)))
            wait_length += self.sleep_step
            time.sleep(self.sleep_step)
            pass
        self.plotter = EQcorrscanPlot(
            rt_client=self.client, plot_length=self.plot_length,
            tribe=self, inventory=self.inventory,
            detections=self.detections, exclude_channels=self.exclude_channels)
        self.plotter.background_run()

    def stop(self):
        if self.plotter is not None:
            self.plotter.background_stop()
        self.client.background_stop()

    def run(self, threshold, threshold_type, trig_int,
            keep_detections=86400, detect_directory="detections",
            max_run_length=None, **kwargs):
        """
        Run the RealTimeTribe detection.

        Detections will be added to a party and returned when the detection
        is done. Detections will be stored in memory for up to `keep_detections`
        seconds.  Detections will also be written to individual files in the
        `detect_directory`.

        :type threshold: float
        :param threshold: Threshold for detection
        :type threshold_type: str
        :param threshold_type:
            Type of threshold to use. See
            `eqcorrscan.core.match_filter.Tribe.detect` for options.
        :type trig_int: float
        :param trig_int: Minimum inter-detection time in seconds.
        :type keep_detections: float
        :param keep_detections:
            Duration to store detection in memory for in seconds.
        :type detect_directory: str
        :param detect_directory:
            Relative path to directory for detections. This directory will be
            created if it doesn't exist.
        :type max_run_length: float
        :param max_run_length:
            Maximum detection run time in seconds. Default is to run
            indefinitely.
        """
        run_start = UTCDateTime.now()
        running = True

        last_possible_detection = UTCDateTime(0)
        if not os.path.isdir(detect_directory):
            os.makedirs(detect_directory)
        if not self.client.busy:
            for tr_id in self.expected_channels:
                self.client.select_stream(
                    net=tr_id.split('.')[0], station=tr_id.split('.')[1],
                    selector=tr_id.split('.')[3])
            self.client.background_run()
            Logger.info("Started real-time streaming")
        else:
            Logger.warning("Client already in streaming mode,"
                           " cannot add channels")
        if self.plot:  # pragma: no cover
            # Set up plotting thread
            self._plot()
            Logger.info("Plotting thread started")
        if not self.client.buffer_full:
            Logger.info("Sleeping for {0}s while accumulating data".format(
                self.client.buffer_capacity - self.client.buffer_length + 5))
            time.sleep(self.client.buffer_capacity -
                       self.client.buffer_length + 5)
        try:
            while running:
                start_time = UTCDateTime.now()
                st = self.client.get_stream()
                st = st.merge()
                st.trim(starttime=max([tr.stats.starttime for tr in st]),
                        endtime=min([tr.stats.endtime for tr in st]))
                Logger.info("Starting detection run")
                try:
                    new_party = self.detect(
                        stream=st, plotvar=False, threshold=threshold,
                        threshold_type=threshold_type, trig_int=trig_int,
                        **kwargs)
                except Exception as e:
                    Logger.error(e)
                    Logger.info(
                        "Waiting for {0}s and hoping this gets better".format(
                            self.detect_interval))
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
                                year_dir = os.path.join(
                                    detect_directory,
                                    str(detection.detect_time.year))
                                if not os.path.isdir(year_dir):
                                    os.makedirs(year_dir)
                                day_dir = os.path.join(
                                    year_dir, str(detection.detect_time.julday))
                                if not os.path.isdir(day_dir):
                                    os.makedirs(day_dir)
                                detection.event.write(os.path.join(
                                    day_dir, detection.detect_time.strftime(
                                        "%Y%m%dT%H%M%S.xml")), format="QUAKEML")
                                _family += detection
                        self.party += _family
                    Logger.info("Removing duplicate detections")
                    self.party.decluster(trig_int=trig_int)
                    # Remove old detections here
                    for family in self.party:
                        family.detections = [
                            d for d in family.detections
                            if d.detect_time >= UTCDateTime.now() - keep_detections]
                    for f in self.party:
                        for d in f:
                            if d not in self.detections:
                                # Need to append rather than create a new object
                                self.detections.append(d)
                Logger.info("Party now contains {0} detections".format(
                    len(self.detections)))
                run_time = UTCDateTime.now() - start_time
                Logger.info("Detection took {0:.2f}s".format(run_time))
                if self.detect_interval <= run_time:
                    Logger.warning(
                        "detect_interval {0} shorter than run-time {1}, "
                        "increasing detect_interval to {2}".format(
                            self.detect_interval, run_time, run_time + 10))
                    self.detect_interval = run_time + 10
                Logger.debug("This step took {0}s total".format(run_time))
                Logger.info("Waiting {0}s until next run".format(
                    self.detect_interval - run_time))
                time.sleep(self.detect_interval - run_time)
                if max_run_length and UTCDateTime.now() > run_start + max_run_length:
                    running = False
        finally:
            self.stop()
        return self.party


if __name__ == "__main__":
    import doctest

    doctest.testmod()
