"""
Runner for the plotting funcs.
"""

import os
import logging
import shutil
import glob
import numpy as np

from typing import Iterable, List, Union, Tuple

from obspy import read_events, UTCDateTime, Inventory, read_inventory
from obspy.core.event import Event

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)
from rt_eqcorrscan.helpers.sparse_event import sparsify_catalog, SparseEvent, \
    get_origin_attr, get_magnitude_attr
from rt_eqcorrscan.plugins.plotter.rcet_plots import (
    aftershock_map, summary_files, ellipse_plots,
    ellipse_to_rectangle, focal_sphere_plots, plot_scaled_magnitudes,
    make_scaled_mag_list, mainshock_mags,
)
from rt_eqcorrscan.plugins.output.output_runner import template_possible_self_dets


Logger = logging.getLogger(__name__)


class PlotConfig(_PluginConfig):
    """
    Configuration for the plotter plugin.
    """
    defaults = _PluginConfig.defaults.copy()
    defaults.update({
        "sleep_interval": 600,
        "mainshock_id": None,
        "station_file": None,
        "png_dpi": 300,
        "eps_dpi": 300,
        "scaled_mag_relation": 1, # What is this? Why is it an integer?
        "faulting_type": None, # Should be a string ["NN", "SS", "RV", "DS", "SI"]
        "rupture_area": "ellipse", # or "rectangle"
        "Mw": None,
        "Mw_unc": None,
        "MT_NP1": None, # [strike, dip, rake]
        "MT_NP2": None, # [strike, dip, rake]
        "fabric_angle": 55,  # What is this?
        "ellipse_std": 2,  # what is this?
        "IQR_k": 1.5,
        "lowess": True,
        "lowess_f": 0.5,
        "magcut": 3.0,
        "search_radius": 100.0,
    })
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.station_file:
            self.station_file = os.path.abspath(self.station_file)


PLUGIN_CONFIG_MAPPER.update({"plotter": PlotConfig})


class Plotter(_Plugin):
    name = "Plotter"
    event_cache = {}  # Dict of SparseEvents keyed by event id
    event_files = {}  # Dict of (event-id, mtime) keyed by event-file
    inventory_cache = (None, None, None)  # Tuple of (inventory, file, mtime)
    simulation_time_offset = 0.0  # Seconds to subtract from now to get the simulated time.

    @property
    def _aftershock_templates(self):
        return [ev for ev in self.template_dict.values()
                if get_origin_attr(ev, "time") or UTCDateTime(0) >=
                get_origin_attr(self._get_mainshock(), "time")]

    @property
    def events(self) -> List:
        return [e for e in self.event_cache.values()]

    @property
    def inventory(self) -> Union[Inventory, None]:
        return self.inventory_cache[0]

    @property
    def now(self) -> UTCDateTime:
        return UTCDateTime.now() - self.simulation_time_offset

    @property
    def _now_str(self) -> str:
        return self.now.strftime("%Y-%m-%dT%H-%M-%S")

    def _read_config(self, config_file: str):
        return PlotConfig.read(config_file=config_file)

    def core(self, new_files: Iterable, cleanup: bool = True) -> List:
        """ Run the plotter. """
        now = self.now
        internal_config = self.config.copy()
        out_dir = internal_config.pop("out_dir")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if not os.path.isdir(f"{out_dir}/history"):
            os.makedirs(f"{out_dir}/history")

        """
        Things we need for the plots:
        1. The real-time catalogue - ideally we could split this into different origins
            This will come from config.in_dir - ideally identify our version of the mainshock in this.
        2. The template catalogue
            This will come from config.template_dir
        3. The mainshock/trigger event
            This needs to be set by rt_match_filter configure_plugins
        4. The inventory
            This needs to be set by rt_match_filter configure_plugins
        5. Search radius
            This needs to be set by rt_match_filter configure_plugins
        
        We want to be able to get the moment tensor from the config file, and 
        the faulting type and rupture area type - these should be update-able 
        in the config file.
        """
        # Load template events
        Logger.info("Getting template events")
        self.get_template_events()

        # Load the stations
        Logger.info("Getting inventory")
        read_inv = False
        if self.inventory_cache[2] is not None:
            # We have read the inventory file, check if we need to re-read
            if self.inventory_cache[1] != internal_config.station_file:
                # Different file, we need to read
                read_inv = True
            elif os.path.getmtime(internal_config.station_file) > self.inventory_cache[2]:
                # Updated, we need to read
                read_inv = True
        else:
            # No file has been read previously
            read_inv = True
        if read_inv:
            self.inventory_cache = (
                read_inventory(internal_config.station_file),
                internal_config.station_file,
                os.path.getmtime(internal_config.station_file))

        # Remove expired events (e.g. those in our cache that do not appear
        # in new_files)
        Logger.info("Checking expired files")
        expired_files = set(self.event_files.keys()).difference(set(new_files))
        for expired_file in expired_files:
            expired_id, _ = self.event_files.pop(expired_file)
            # Remove from event cache
            self.event_cache.pop(expired_id)

        # Read detected events - we need to re-check all events everytime to
        # cope with updates
        Logger.info("Reading events")
        for f in new_files:
            if not os.path.isfile(f):
                # File no longer exists.
                continue
            mtime = os.path.getmtime(f)
            if f in self.event_files.keys():
                # Check if the file has changed since we last read from it
                if mtime <= self.event_files[f][1]:
                    # File has not been updated. Skip reading
                    continue
            # File is either new or updated. Read
            if os.path.isfile(f):
                # Cope with files being changed while we work...
                cat = sparsify_catalog(read_events(f), include_picks=True)
            else:
                # We can ignore it.
                continue
            assert len(cat) == 1, f"More than one event in {f}"
            ev = cat[0]
            self.event_cache.update({ev.resource_id.id: ev})
            self.event_files.update(
                {f: (ev.resource_id.id, mtime)})

        Logger.info("Making earthquake maps")
        try:
            self._aftershock_maps()
        except Exception as e:
            Logger.error(f"Could not make aftershock maps due to {e}")
        Logger.info("Computing ellipse statistics and plotting")
        try:
            ellipse_stats, catalog_outliers = self._ellipse_plots()
        except Exception as e:
            Logger.error(f"Could not get ellipse stats due to {e}")
            ellipse_stats = None
        if not ellipse_stats:
            # Everything else requires the ellipse stats and/or catalog_outliers
            self._add_to_history()
            return []
        Logger.info("Plotting beachballs")
        try:
            self._beachball_plots(
                aftershock_azimuth=ellipse_stats["azimuth"],
                aftershock_dip=ellipse_stats["dip"])
        except Exception as e:
            Logger.error(f"Could not plot beachballs due to {e}")
        Logger.info("Plotting magnitude scaling relationships")
        try:
            scaled_mag = self._magnitude_plots(
                length=ellipse_stats["length"],
                length_z=ellipse_stats["length_z"],
                mainshock=self._get_mainshock()
            )
        except Exception as e:
            scaled_mag = np.nan
            Logger.error(f"Could not plot magnitude relationships due to {e}")
        Logger.info("Making summary files")
        (geonet_mainshock_mag, geonet_mainshock_mag_uncertainty,
         geonet_mainshock_depth, geonet_mainshock_depth_uncertainty,
         RT_mainshock_depth, RT_mainshock_depth_uncertainty) = mainshock_mags(
            mainshock=self._get_mainshock(), RT_mainshock=self._get_relocated_mainshock())

        output_dictionary = summary_files(
            eventid=internal_config.mainshock_id,
            current_time=now,
            elapsed_secs=now - get_origin_attr(self._get_mainshock(), "time"),
            catalog_RT=self.events,
            cat_counts=[
                len([ev for ev in self.events if len(ev.origins) == 0]),
                len([ev for ev in self.events if len(ev.magnitudes) == 0]),
                len([t for t in self.template_dict.values() if len(t.origins) == 0]),
                len([t for t in self.template_dict.values() if len(t.magnitudes) == 0])
            ],
            catalog_geonet=self._aftershock_templates,
            catalog_outliers=catalog_outliers,
            length=ellipse_stats['length'],
            azimuth=ellipse_stats['azimuth'],
            dip=ellipse_stats['dip'],
            length_z=ellipse_stats['length_z'],
            scaled_mag=scaled_mag,
            geonet_mainshock_mag=geonet_mainshock_mag,
            geonet_mainshock_mag_uncertainty=geonet_mainshock_mag_uncertainty,
            mean_depth=np.mean([get_origin_attr(ev, "depth") / 1000.0 for ev in self.events
                                if get_origin_attr(ev, "depth") is not None]),
            RT_mainshock_depth=RT_mainshock_depth,
            RT_mainshock_depth_uncertainty=RT_mainshock_depth_uncertainty,
            geonet_mainshock_depth=geonet_mainshock_depth,
            geonet_mainshock_depth_uncertainty=geonet_mainshock_depth_uncertainty,
            output_dir=out_dir)
        #
        # # TODO: pass args?
        # Logger.info("Making summary figure")
        # self._summary_figure()
        self._add_to_history()
        return []

    def _add_to_history(self):
        # Copy "latest" to history with timestamps
        now_str = self._now_str
        os.makedirs(f"{self.config.out_dir}/history/{now_str}")
        for plot in glob.glob(f"{self.config.out_dir}/*_latest.*"):
            fname = os.path.basename(plot)
            fname.replace("_latest", now_str)
            shutil.copy(plot, f"{self.config.out_dir}/history/{now_str}/{fname}")

    def _get_mainshock(self) -> Union[SparseEvent, Event, None]:
        # Get the mainshock
        mainshock = [ev for ev in self.template_dict.values()
                     if self.config.mainshock_id in ev.resource_id.id]
        if len(mainshock) == 0:
            Logger.error(
                f"Mainshock ({self.config.mainshock_id}) not found in "
                f"templates ({self.template_dict.keys()}.")
        elif len(mainshock) > 1:
            Logger.warning("Multiple possible mainshocks!")
        return mainshock[0]

    def _get_relocated_mainshock(self) -> Union[SparseEvent, Event, None]:
        """ Try to find our detection of the mainshock """
        mainshock = self._get_mainshock()
        if mainshock == None:
            return None
        t_name = mainshock.resource_id.id.split('/')[-1]
        # Look for a template detections
        t_events = [ev for ev in self.events
                    if ev.resource_id.id.lstrip("smi:local/").startswith(t_name)]
        if len(t_events) == 0:
            # No detections, so we need to output the template
            Logger.info(f"No self-detections for mainshock: {t_name}")
            return None
        # Look for template self detections
        self_dets = template_possible_self_dets(
            template_event=mainshock, catalog=t_events)
        if len(self_dets) == 0:
            Logger.info(f"No self-detections for mainshock: {t_name}")
            return None
        if len(self_dets) > 1:
            Logger.info(
                f"Multiple possible self-dets found for mainshock: {t_name}")
            # Now we need for find the "best" match - simple way would be sort by
            # origin time differences, not robust at all, but it is "something"
            deltas = [(ev, abs(get_origin_attr(mainshock, "time") -
                               get_origin_attr(ev, "time")))
                      for ev in self_dets]
            deltas.sort(key=lambda tup: tup[1])
            return deltas[0][0]
        else:
            return self_dets[0]

    #### Plotting methods
    def _summary_figure(self):
        fig = output_aftershock_map(
            catalog=catalog_origins,
            reference_catalog=catalog_geonet,
            outlier_catalog=catalog_outliers,
            mainshock=mainshock,
            RT_mainshock=relocated_mainshock[0],
            corners=corners,
            cat_counts=cat_counts,
            width=20,
            topo_res="03s",
            topo_cmap="terra",
            inventory=inv,
            hillshade=False,
            colours='depth')

        fig.savefig(
            f"{self.config.out_dir}/Aftershock_extent_depth_map_latest.png",
            dpi=self.config.png_dpi)
        fig.savefig(
            f"{self.config.out_dir}/Aftershock_extent_depth_map_latest.pdf",
            dpi=self.config.eps_dpi)
        return

    def _magnitude_plots(
        self,
        length: float,
        length_z: float,
        mainshock: Event,
    ):
        mag_list, slip_list, ref_list, scaled_mag = make_scaled_mag_list(
            length=length, width=length_z,
            rupture_area=self.config.rupture_area,
            scaled_mag_relation=self.config.scaled_mag_relation)
        # currently cuts off figure legend!
        fig = plot_scaled_magnitudes(
            mag_list=mag_list, scaled_mag=scaled_mag, slip_list=slip_list,
            ref_list=ref_list, Mw=self.config.Mw, mainshock=mainshock)
        fig.savefig(
            f"{self.config.out_dir}/Scaled_Magnitude_Comparison_latest.pdf",
            dpi=self.config.eps_dpi)
        fig.savefig(
            f"{self.config.out_dir}/Scaled_Magnitude_Comparison_latest.png",
            dpi=self.config.png_dpi)
        return scaled_mag

    def _beachball_plots(self, aftershock_azimuth, aftershock_dip):
        """ Make focal sphere plots. """
        fig = focal_sphere_plots(
            azimuth=aftershock_azimuth,
            dip=aftershock_dip,
            MT_NP1=self.config.MT_NP1,
            MT_NP2=self.config.MT_NP2)
        fig.savefig(
            f"{self.config.out_dir}/focal_sphere_latest.png",
            dpi=self.config.png_dpi)
        fig.savefig(
            f"{self.config.out_dir}/focal_sphere_latest.pdf",
            dpi=self.config.eps_dpi)
        return

    def _ellipse_plots(self) -> dict:
        """ Work out the ellipses and make plots """
        mainshock = self._get_mainshock()
        (ellipse_stats, catalog_outliers, ellipse_map,
         ellipse_xsection) = ellipse_plots(
            catalog_origins=self.events,
            mainshock=mainshock,
            relocated_mainshock=self._get_relocated_mainshock(),
            fabric_angle=self.config.fabric_angle,
            IQR_k=self.config.IQR_k,
            ellipse_std=self.config.ellipse_std,
            lowess=self.config.lowess,
            lowess_f=self.config.lowess_f,
            radius_km=self.config.search_radius)

        ellipse_map.savefig(
            f'{self.config.out_dir}/confidence_ellipsoid_latest.png',
            dpi=self.config.png_dpi)
        ellipse_map.savefig(
            f'{self.config.out_dir}/confidence_ellipsoid_latest.pdf',
            dpi=self.config.eps_dpi)

        ellipse_xsection.savefig(
            f'{self.config.out_dir}/confidence_ellipsoid_'
            f'vertical_latest.png',
            dpi=self.config.png_dpi)
        ellipse_xsection.savefig(
            f'{self.config.out_dir}/confidence_ellipsoid_'
            f'vertical_latest.pdf',
            dpi=self.config.eps_dpi)

        corners = ellipse_to_rectangle(
            latitude=(mainshock.preferred_origin() or
                      mainshock.origins[-1]).latitude,
            longitude=(mainshock.preferred_origin() or
                       mainshock.origins[-1]).longitude,
            offset_x=ellipse_stats['x_mean'],
            offset_y=ellipse_stats['y_mean'],
            length=ellipse_stats['length'],
            width=ellipse_stats['width'],
            azimuth=ellipse_stats['azimuth'])

        corners_3d = []  # TODO include depths of corners to be fed into json writeout
        # need to calculate the dip direction robustly to ensure correct
        # corners are assigned depths

        # TODO: Output corners to json?
        return ellipse_stats, catalog_outliers

    def _aftershock_maps(self):
        """ Make core aftershock maps. """
        now = UTCDateTime.now()
        mainshock = self._get_mainshock()

        template_map = aftershock_map(
            catalog=self.template_dict.values(),
            mainshock=mainshock,
            inventory=self.inventory,
            topo_res="03s",
            topo_cmap="grayC",
            hillshade=False,
            pad=5.0,
            timestamp=self.now(),
        )

        template_map.savefig(
            f"{self.config.out_dir}/catalog_templates_latest.png",
            dpi=self.config.png_dpi)
        template_map.savefig(
            f"{self.config.out_dir}/catalog_templates_latest.pdf",
            dpi=self.config.eps_dpi)
        
        detected_map = aftershock_map(
            catalog=self.events,
            mainshock=mainshock,
            inventory=self.inventory,
            topo_res="03s",
            topo_cmap="grayC",
            hillshade=False,
            pad=5.0,
            timestamp=self.now(),
        )

        detected_map.savefig(
            f"{self.config.out_dir}/catalog_RT_latest.png",
            dpi=self.config.png_dpi)
        detected_map.savefig(
            f"{self.config.out_dir}/catalog_RT_latest.pdf",
            dpi=self.config.eps_dpi)

        return


if __name__ == "__main__":
    import doctest

    doctest.testmod()
