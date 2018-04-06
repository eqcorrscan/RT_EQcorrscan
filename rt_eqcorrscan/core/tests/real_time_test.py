"""
Tests for real-time matched-filtering.
"""

import unittest

from eqcorrscan import Tribe

from rt_eqcorrscan.core.rt_match_filter import RealTimeTribe


class MatchFilterTest(unittest.TestCase):
    def test_init(self):
        tribe = Tribe().construct()
        rt_tribe = RealTimeTribe(tribe=tribe, client="GEONET")


if __name__ == "__main__":
    unittest.main()