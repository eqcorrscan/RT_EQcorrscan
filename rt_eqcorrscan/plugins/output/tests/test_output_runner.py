"""
Tests for the output plugin
"""

import unittest
import logging
import time
import os
import shutil
import pandas as pd
import glob
import pickle

from itertools import cycle
from typing import Tuple, List
from multiprocessing import Process

from obspy import Catalog, UTCDateTime, read_events

from eqcorrscan import Template

from rt_eqcorrscan.plugins.output.output_runner import Outputter, OutputConfig

Logger = logging.getLogger(__name__)


def _write_sim_info(
    templates: List[Template],
    locations: Catalog,
    relocations: Catalog,
    template_dir: str,
    location_dir: str,
    relocation_dir: str,
    poison_dir: str,
    sleep_step: float = 20.0
):
    for _dir in [template_dir, location_dir, relocation_dir, poison_dir]:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
            Logger.debug(f"Making {_dir}")
    Logger.debug("Simulation output dirs created")

    # Write all templates
    for template in templates:
        Logger.info(
            f"Writing template to {template_dir}/{template.name}.pkl")
        with open(f"{template_dir}/{template.name}.pkl", "wb") as f:
            pickle.dump(template, f)
    Logger.debug("Simulation templates written")

    # Write chunks of detections
    loc_slices = [slice(0, 1), slice(1, 5), slice(5, 50), slice(50, 125),
                  slice(125, None)]
    reloc_slices = [slice(0, 1), slice(1, 4), slice(4, 30), slice(30, 100),
                    slice(100, None)]

    for _loc_slice, _reloc_slice in zip(loc_slices, reloc_slices):
        loc_events = locations[_loc_slice]
        reloc_events = relocations[_reloc_slice]
        for event in loc_events:
            Logger.debug(f"Writing located event {event.resource_id.id.split('/')[-1]}")
            event.write(f"{location_dir}/{event.resource_id.id.split('/')[-1]}.xml",
                        format="QUAKEML")
        for event in reloc_events:
            Logger.debug(f"Writing located event {event.resource_id.id.split('/')[-1]}")
            event.write(f"{relocation_dir}/{event.resource_id.id.split('/')[-1]}.xml",
                        format="QUAKEML")
        time.sleep(sleep_step)
    Logger.info("Detection writing complete, waiting to poison")
    time.sleep(sleep_step * 4)

    Logger.info("Poisoning")
    with open(f"{poison_dir}/poison", "w") as f:
        f.write("Poisoned at the end of detection writing")
    return


def setup_testcase() -> Tuple[List[Template], Catalog, Catalog]:
    """ Download some files and chuck em in. """
    from obspy.clients.fdsn import Client

    client = Client("GEONET")

    Logger.info("Setting up testcase")
    # Kaikōura - Cape Campbell ~ 200 events
    cat = client.get_events(
        starttime=UTCDateTime(2016, 11, 13),
        endtime=UTCDateTime(2016, 11, 14),
        minlatitude=-41.9,
        maxlatitude=-41.6,
        minlongitude=174.0,
        maxlongitude=174.4,
        maxdepth=40
    )
    Logger.info(f"Downloaded {len(cat)} events")

    # Don't use all the events as templates
    templates = cat[0:20].copy()
    template_ids = [t.resource_id.id.split('/')[-1] for t in templates]
    templates = [Template(event=t, name=t.resource_id.id.split('/')[-1])
                 for t in templates]

    template_ids = cycle(template_ids)
    # Hack them to look like detections
    for event in cat:
        tid = next(template_ids)
        event.resource_id.id = f"{tid}_{event.origins[-1].time.strftime('%Y%m%dT%H%M%S')}"
        event.preferred_origin().method_id.id = "NLL"

    # Don't take all the events through to relocation
    drop_ids = [5, 25, 30, 32, 100, 105, 106, 107, 108, 109, 110, 145, 167, 188]
    reloc_cat = Catalog([ev.copy() for i, ev in enumerate(cat)
                         if i not in drop_ids])
    for ev in reloc_cat:
        ev.preferred_origin().method_id.id = "Growclust"

    return templates, cat, reloc_cat


# TODO: Test self-detection matching


class TestOutputPlugin(unittest.TestCase):
    clean_up = []
    @classmethod
    def setUpClass(cls) -> None:
        cls.template_dir = os.path.abspath("./output_test_templates")
        cls.location_dir = os.path.abspath("./output_test_locations")
        cls.relocation_dir = os.path.abspath("./output_test_relocations")

        cls.templates, cls.located, cls.relocated = setup_testcase()

        cls.clean_up.extend([cls.template_dir, cls.location_dir, cls.relocation_dir])

    def test_updating(self):
        config_file, out_dir = "test_output_config.yml", os.path.abspath("./test_output_output")
        self.clean_up.extend([config_file, out_dir])
        config = OutputConfig(
            output_templates=True, template_dir=self.template_dir,
            in_dir=[self.location_dir, self.relocation_dir],
            out_dir=out_dir)

        config.write(config_file)

        detection_writer = Process(
            target=_write_sim_info,
            args=(self.templates, self.located, self.relocated, self.template_dir,
                  self.location_dir, self.relocation_dir, out_dir, 50.0),
            name="DetectionWriter")

        Logger.info("Starting detection writer")
        detection_writer.start()

        Logger.info("Starting output runner")
        failed = False
        try:
            output_runner = Outputter(config_file=config_file)
            output_runner.run()
        except Exception as e:
            Logger.exception(f"Failed due to {e}")
            failed = True
        finally:
            detection_writer.kill()
        Logger.info("Output runner finished")
        self.assertFalse(failed)

        # Read back in the output - this should be one qml and one csv
        self.assertEqual(len(glob.glob(f"{out_dir}/catalog/*.xml")),
                         len(self.located))
        cat_back = read_events(f"{out_dir}/catalog/*.xml")
        cat_back_csv = pd.read_csv(f"{out_dir}/catalog.csv")

        self.assertEqual(len(cat_back), len(self.located))

        return

    @classmethod
    def tearDownClass(cls, clean=True):
        if clean:
            for thing in cls.clean_up:
                if os.path.isdir(thing):
                    try:
                        Logger.info(f"Removing {thing}")
                        shutil.rmtree(thing)
                    except Exception:
                        pass
                else:
                    try:
                        Logger.info(f"Deleting {thing}")
                        os.remove(thing)
                    except Exception:
                        pass


if __name__ == "__main__":
    unittest.main()
