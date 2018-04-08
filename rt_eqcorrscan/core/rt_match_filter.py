"""
Classes for real-time matched-filter detection of earthquakes.

:copyright:
    Calum Chamberlain

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from obspy import Stream
from eqcorrscan import Tribe, Template

from rt_eqcorrscan.utils.seedlink import RealTimeClient


class RealTimeTribe(Tribe):
    """
    Real-Time tribe.

    :type tribe: `eqcorrscan.core.match_filter.Tribe
    :param tribe: Tribe of templates to use for detection.
    :type server_url: str
    :param server_url: Address of seedlink client.
    """
    def __init__(self, tribe=None, server_url=None, buffer_capacity=600):
        Tribe.templates = tribe.templates
        self.buffer = Stream()
        self.client = RealTimeClient(
            server_url=server_url, autoconnect=True, buffer=self.buffer,
            buffer_capacity=buffer_capacity)

    def __repr__(self):
        """
        Print information about the tribe.

        .. rubric:: Example

        >>> tribe = RealTimeTribe(tribe=Tribe([Template(name='a')]),
        ...                       server_url="geofon.gfz-potsdam.de")
        >>> print(tribe) # doctest: +NORMALIZE_WHITESPACE
        Real-Time Tribe of 1 templates on client:
        Seed-link client at geofon.gfz-potsdam.de, buffer capacity: 600s
            Current Buffer:
        0 Trace(s) in Stream:
        <BLANKLINE>
        """
        return 'Real-Time Tribe of {0} templates on client:\n{1}'.format(
            self.__len__(), self.client)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
