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
from eqcorrscan import Tribe, Template, Party

from rt_eqcorrscan.utils.seedlink import RealTimeClient
from rt_eqcorrscan.plotting.plot_buffer import EQcorrscanPlot

SLEEP_STEP = 1.0
LOGGING_MAP = {
    'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING,
    'error': logging.ERROR, 'critical': logging.CRITICAL}
DEBUG_MAP = {'info': 5, 'debug': 4, 'warning': 3, 'error': 2, 'critical': 1}


class RealTimeTribe(Tribe):
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
                 buffer_capacity=600, detect_interval=60, log_level='warning',
                 plot=True, plot_length=300):
        assert (buffer_capacity >= max(
            [template.process_length for template in self.templates]))
        assert (buffer_capacity >= detect_interval)
        try:
            logging.basicConfig(
                level=LOGGING_MAP[log_level],
                format="%(asctime)s   %(threadName)s\t%(levelname)s \t"
                       "%(message)s")
        except KeyError:
            print("log_level must be in {0}".format(LOGGING_MAP.keys()))
        self.log_level = log_level
        super().__init__(templates=tribe.templates)
        self.buffer = Stream()
        self.inventory = inventory
        self.party = Party()
        self.busy = True
        self.detect_interval = detect_interval
        self.client = RealTimeClient(
            server_url=server_url, autoconnect=True, buffer=self.buffer,
            buffer_capacity=buffer_capacity)
        self.plot = plot
        self.plot_length = plot_length

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
    def expected_channels(self):
        return set(tr.id for template in self.templates for tr in template.st)

    def _plot(self):
        """Plot the data as it comes in."""
        wait_length = 0
        while len(self.client.buffer) < len(self.expected_channels):
            if wait_length >= self.detect_interval:
                logging.warning("Starting plotting without the full dataset")
                break
            # Wait until we have some data
            logging.debug(
                "Waiting for data, currently have {0} channels of {1} "
                "expected channels".format(
                    len(self.client.buffer), len(self.expected_channels)))
            wait_length += SLEEP_STEP
            time.sleep(SLEEP_STEP)
            pass
        plotter = EQcorrscanPlot(
            rt_client=self.client, plot_length=self.plot_length,
            template_catalog=[t.event for t in self.templates],
            inventory=self.inventory)
        plotter.background_run()

    # TODO: Remove old detections to save memory. Write detections to file
    #  during operation.
    def run(self, threshold, threshold_type, trig_int,
            keep_detections=86400, detect_directory="detections",
            max_run_length=None):
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
            logging.info("Started real-time streaming")
            time.sleep(self.client.buffer_capacity)
        else:
            logging.warning("Client already in streaming mode,"
                            " cannot add channels")
            if not self.client.buffer_full:
                time.sleep(self.client.buffer_capacity -
                           self.client.buffer_length + 5)
        if self.plot:  # pragma: no cover
            # Set up plotting thread
            self._plot()
            logging.info("Plotting thread started")
        while running:
            start_time = UTCDateTime.now()
            st = self.client.get_stream()
            st.trim(starttime=max([tr.stats.starttime for tr in st]),
                    endtime=min([tr.stats.endtime for tr in st]))
            # I think I need to copy this to ensure it isn't worked on in place.
            logging.info("Starting detection run")
            new_party = self.detect(
                stream=st, plotvar=False, threshold=threshold,
                threshold_type=threshold_type, trig_int=trig_int,
                debug=DEBUG_MAP[self.log_level])
            for family in new_party:
                _family = family.copy()
                _family.detections = []
                for detection in family:
                    if detection.detect_time > last_possible_detection:
                        year_dir = os.path.join(
                            detect_directory, str(detection.detect_time.year))
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
            if len(self.party) > 0:
                logging.info("Removing duplicate detections")
                # TODO: Use the declustering in EQcorrscan.
            run_time = UTCDateTime.now() - start_time
            logging.info("Detection took {0}s".format(run_time))
            time.sleep(self.detect_interval - run_time)
            if UTCDateTime.now() > run_start + max_run_length:
                running = False
        return self.party


if __name__ == "__main__":
    import doctest

    doctest.testmod()
