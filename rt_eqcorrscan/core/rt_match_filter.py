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

from obspy import Stream, UTCDateTime
from eqcorrscan import Tribe, Template, Party

from rt_eqcorrscan.utils.seedlink import RealTimeClient


class RealTimeTribe(Tribe):
    """
    Real-Time tribe.

    :type tribe: `eqcorrscan.core.match_filter.Tribe
    :param tribe: Tribe of templates to use for detection.
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
    def __init__(self, tribe=None, server_url=None, buffer_capacity=600,
                 detect_interval=60):
        super().__init__(templates=tribe.templates)
        self.buffer = Stream()
        self.party = Party()
        self.detect_interval = detect_interval
        self.client = RealTimeClient(
            server_url=server_url, autoconnect=True, buffer=self.buffer,
            buffer_capacity=buffer_capacity)
        assert(buffer_capacity >= max(
            [template.process_length for template in self.templates]))
        assert(buffer_capacity >= detect_interval)

    def __repr__(self):
        """
        Print information about the tribe.

        .. rubric:: Example

        >>> tribe = RealTimeTribe(
        ...     tribe=Tribe([Template(name='a', process_length=60)]),
        ...     server_url="geofon.gfz-potsdam.de")
        >>> print(tribe) # doctest: +NORMALIZE_WHITESPACE
        Real-Time Tribe of 1 templates on client:
        Seed-link client at geofon.gfz-potsdam.de, buffer capacity: 600s
            Current Buffer:
        0 Trace(s) in Stream:
        <BLANKLINE>
        """
        return 'Real-Time Tribe of {0} templates on client:\n{1}'.format(
            self.__len__(), self.client)

    def run(self, threshold, threshold_type, trig_int,
            keep_detections=86400, detect_directory="detections"):
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

        """
        last_possible_detection = UTCDateTime(0)
        if not os.path.isdir(detect_directory):
            os.makedirs(detect_directory)
        self.client.background_run()
        time.sleep(self.client.buffer_capacity)
        while True:
            start_time = UTCDateTime.now()
            new_party = self.detect(
                stream=self.client.buffer, plotvar=False, threshold=threshold,
                threshold_type=threshold_type, trig_int=trig_int)
            for family in new_party:
                for detection in family:
                    if detection.detect_time > last_possible_detection:
                        year_dir = os.path.join(
                            detect_directory, str(detection.detect_time.year))
                        if not os.path.isdir(year_dir):
                            os.makedirs(year_dir)
                        day_dir = os.path.join(
                            year_dir, str(detection.detect_time.jday))
                        if not os.path.isdir(day_dir):
                            os.makedirs(day_dir)
                        detection.event.write(os.path.join(
                            day_dir, detection.detect_time.strftime(
                                "%Y%m%dT%H%M%S.xml")))
                        self.party += detection  # Does this method work?
            for family in self.party:
                _detections = []
                for detection in family:
                    if detection.detect_time > start_time - keep_detections:
                        _detections.append(detection)
                family.detections = _detections
            run_time = UTCDateTime.now() - start_time
            time.sleep(self.detect_interval - run_time)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
