"""
Tests for the lag-calc plugin for RTEQcorrscan
"""

import os
import shutil
import unittest
import pickle

from eqcorrscan import Party

from rt_eqcorrscan.plugins.lag_calc import events_to_party


class TestLagCalcPlugin(unittest.TestCase):
    clean_up = []
    @classmethod
    def setUpClass(cls):
        cls.party = Party().read("test_data/lag_calc_test_party.tgz")

    def test_party_construction(self):
        party = self.party.copy()

        # Get the events out of the party
        events = party.get_catalog()

        # Write out into expected structure
        template_dir = ".templates"
        os.makedirs(template_dir, exist_ok=True)
        self.clean_up.append(template_dir)
        for f in party:
            t = f.template
            with open(f"{template_dir}/{t.name}.pkl", "wb") as fp:
                pickle.dump(t, fp)

        party_back = events_to_party(events=events, template_dir=template_dir)
        # self.assertEqual(party, party_back)
        # Some elements (threshold_input, threshold_type, threshold)
        # lost in translation
        for fam in party:
            fam_back = party_back.select(fam.template.name)
            self.assertEqual(len(fam), len(fam_back))
            self.assertEqual(fam.template, fam_back.template)
            dets = sorted(fam.detections, key=lambda d: d.detect_time)
            dets_back = sorted(fam_back.detections, key=lambda d: d.detect_time)
            for d, db in zip(dets, dets_back):
                for key, value in d.__dict__.items():
                    if key in ["threshold_type", "threshold_input", "threshold"]:
                        continue
                    value_back = db.__dict__[key]
                    self.assertEqual(value, value_back)

    @classmethod
    def tearDownClass(cls):
        for thing in cls.clean_up:
            if os.path.isdir(thing):
                shutil.rmtree(thing)
            else:
                os.remove(thing)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level="DEBUG")
    unittest.main()