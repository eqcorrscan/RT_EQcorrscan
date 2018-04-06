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
    def __init__(self, tribe, client):
        super().__init__()