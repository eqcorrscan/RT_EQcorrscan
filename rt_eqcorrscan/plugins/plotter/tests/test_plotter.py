"""
Tests for the output plugin
"""

import unittest
import logging
import os
import shutil
import glob
import pickle

from itertools import cycle
from typing import Tuple, List

from obspy import Catalog, UTCDateTime, Inventory

from eqcorrscan import Template

from rt_eqcorrscan.plugins.plotter.plotter_runner import Plotter, PlotConfig
from rt_eqcorrscan.helpers.sparse_event import sparsify_catalog

Logger = logging.getLogger(__name__)


def _write_sim_info(
    templates: List[Template],
    locations: Catalog,
    relocations: Catalog,
    template_dir: str,
    location_dir: str,
    relocation_dir: str,
):
    for _dir in [template_dir, location_dir, relocation_dir]:
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

    # Convert catalogs to sparse
    locations = sparsify_catalog(locations)
    relocations = sparsify_catalog(relocations)

    # Write chunks of detections
    loc_slices = [slice(0, 1), slice(1, 5), slice(5, 50), slice(50, 125),
                  slice(125, None)]
    reloc_slices = [slice(0, 1), slice(1, 4), slice(4, 30), slice(30, 100),
                    slice(100, None)]

    for _loc_slice, _reloc_slice in zip(loc_slices, reloc_slices):
        loc_events = locations[_loc_slice]
        reloc_events = relocations[_reloc_slice]
        for event in loc_events:
            Logger.info(f"Writing located event {event.resource_id.id.split('/')[-1]}")
            with open(f"{location_dir}/{event.resource_id.id.split('/')[-1]}.pkl", "wb") as f:
                pickle.dump(event, f)
        for event in reloc_events:
            Logger.info(f"Writing relocated event {event.resource_id.id.split('/')[-1]}")
            with open(f"{relocation_dir}/{event.resource_id.id.split('/')[-1]}.pkl", "wb") as f:
                pickle.dump(event, f)
    return


def setup_testcase() -> Tuple[List[Template], Catalog, Catalog, Inventory]:
    """ Download some files and chuck em in. """
    from obspy.clients.fdsn import Client

    client = Client("GEONET")

    Logger.info("Setting up testcase")
    # Kaikōura - Cape Campbell ~ 200 events
    starttime, endtime = UTCDateTime(2016, 11, 13), UTCDateTime(2016, 11, 14)

    cat = client.get_events(
        starttime=starttime,
        endtime=endtime,
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

    # Get stations
    # bulk = []
    # for ev in cat:
    #     for p in ev.picks:
    #         _bulk = (p.waveform_id.network_code,
    #                  p.waveform_id.station_code,
    #                  p.waveform_id.location_code,
    #                  p.waveform_id.channel_code,
    #                  starttime, endtime)
    #         if _bulk not in bulk:
    #             bulk.append(_bulk)
    # inv = client.get_stations_bulk(bulk)
    inv = client.get_stations(
        starttime=starttime, endtime=endtime,
        latitude=cat[0].preferred_origin().latitude,
        longitude=cat[0].preferred_origin().longitude,
        maxradius=1.0)

    return templates, cat, reloc_cat, inv


class TestOutputPlugin(unittest.TestCase):
    clean_up = []
    @classmethod
    def setUpClass(cls) -> None:
        cls.template_dir = os.path.abspath("./plotter_test_templates")
        cls.location_dir = os.path.abspath("./plotter_test_locations")
        cls.relocation_dir = os.path.abspath("./plotter_test_relocations")
        cls.out_dir = os.path.abspath("./plotter_test_plots")
        cls.config_file = os.path.abspath("./plotter_config.yml")
        cls.station_file = os.path.abspath("./plotter_stations.xml")

        cls.templates, cls.located, cls.relocated, stations = setup_testcase()
        stations.write(cls.station_file, format="STATIONXML")

        cls.clean_up.extend([
            cls.template_dir,
            cls.location_dir,
            cls.relocation_dir,
            # cls.out_dir,
            cls.config_file,
            cls.station_file,
        ])

    def test_plotting(self):
        _write_sim_info(
            templates=self.templates, locations=self.located,
            relocations=self.relocated,
            template_dir=self.template_dir,
            location_dir=self.location_dir,
            relocation_dir=self.relocation_dir)
        self.templates[0].event.preferred_magnitude().mag_errors.uncertainty = 0.5
        config = PlotConfig(
            in_dir=[self.location_dir, self.relocation_dir],
            template_dir=self.template_dir,
            out_dir=self.out_dir,
            station_file=self.station_file,
            mainshock_id=self.templates[0].event.resource_id.id.split('/')[-1],
        )
        config.write(self.config_file)
        plotter = Plotter(config_file=self.config_file)
        plotter.core(
            new_files=glob.glob(f"{self.location_dir}/*.xml") +
                      glob.glob(f"{self.relocation_dir}/*.xml")
        )
        # TODO: This test doesn't really do anything?

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