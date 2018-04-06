"""
Classes for real-time matched-filter detection of earthquakes.

:copyright:
    Calum Chamberlain

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from eqcorrscan import Tribe, Template


class RealTimeTribe(Tribe):
    """
    Real-Time tribe.
    """
    def __init__(self, tribe=None, templates=[], client=None):
        if tribe:
            Tribe.templates = tribe.templates
        else:
            super().__init__(templates=templates)
        self.client = client

    def __repr__(self):
        """
        Print information about the tribe.

        .. rubric:: Example

        >>> tribe = RealTimeTribe(templates=[Template(name='a')], client="bob")
        >>> print(tribe)
        Real-Time Tribe of 1 templates on client bob
        """
        return 'Real-Time Tribe of {0} templates on client {1}'.format(
            self.__len__(), self.client)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
