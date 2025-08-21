"""
Code for making maps and other plots for aftershock detection.

:author: Emily Warren-Smith

(lightly) Editted by Calum Chamberlain
"""

import pygmt
import numpy as np

from typing import Union, Tuple

import datetime

from jmespath.ast import projection
from matplotlib.font_manager import FontProperties
from obspy import UTCDateTime
from obspy.geodetics import kilometer2degrees
from pyproj import CRS, Transformer

from obspy.core.event import Catalog, Event
from obspy.core.inventory import Inventory
import pyproj, math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from rt_eqcorrscan.helpers.sparse_event import get_origin_attr, \
    get_magnitude_attr

GEODESIC = pyproj.Geod(ellps="WGS84")

import statsmodels.api as sm

import os
import csv


def _eq_map(
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    mags: np.ndarray,
    middle_lon: float,
    middle_lat: float,
    search_radius_deg: float,
    times: np.ndarray,  # TODO: Don't need times, but we could use origin type and plot different styles for different types of origin (Hyp, NLL, GC, GeoNet)
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    pad: float,
    width: float,
    inset_multiplier: float,
    topo_res: Union[bool, str],
    topo_cmap: str,
    hillshade: bool,
    timestamp: Union[UTCDateTime, datetime.datetime]
) -> pygmt.Figure:
    """ """
    if station_lons.max() - station_lons.min() > 180:
        station_lons %= 360
    station_lon_range = station_lons.max() - station_lons.min()
    station_lat_range = station_lats.max() - station_lats.min()

    if lons.max() - lons.min() > 180:
        lons %= 360
    lon_range = lons.max() - lons.min()
    lat_range = lats.max() - lats.min()

    all_lons = np.concatenate([station_lons, lons])
    all_lats = np.concatenate([station_lats, lats])
    all_lon_range = all_lons.max() - all_lons.min()
    all_lat_range = all_lats.max() - all_lats.min()

    large_region = [
        all_lons.min() - (0.15 * all_lon_range),
        all_lons.max() + (all_lon_range * 0.15),
        max(-90, all_lats.min() - (all_lat_range * 0.15)),
        min(90, all_lats.max() + (all_lat_range * 0.15)),
    ]
    if middle_lon and middle_lat:
        region = [
            middle_lon - search_radius_deg,
            middle_lon + search_radius_deg,
            max(-90.0, middle_lat - search_radius_deg),
            min(90.0, middle_lat + search_radius_deg)
        ]
    else:
        region = [
            lons.min() - max(search_radius_deg, lon_range * (pad / 100)),
            lons.max() + max(search_radius_deg, lon_range * (pad / 100)),
            max(-90, lats.min() - max(search_radius_deg, lat_range * (pad / 100))),
            min(90, lats.max() + max(search_radius_deg, lat_range * (pad / 100))),
        ]
    # Work out resolution for topography
    if topo_res is True:
        min_region_dim = min(region[1] - region[0], region[3] - region[2])
        if min_region_dim > 10:
            topo_res = "01d"
        elif min_region_dim > 2:
            topo_res = "01m"
        elif min_region_dim > 0.5:
            topo_res = "15s"
        elif min_region_dim > 0.01:
            topo_res = "03s"
        else:
            topo_res = "01s"

    fig = pygmt.Figure()
    fig.basemap(region=region, projection=f"M{width}c",
                frame=["a", f"+t{len(lats)} earthquakes at "
                            f"{timestamp.strftime('%Y/%m/%d %H:%M:%S')}"])

    grid = pygmt.datasets.load_earth_relief(resolution=topo_res, region=region)
    if hillshade:
        dgrid = pygmt.grdgradient(grid=grid, radiance=[0, 80], normalize=True)
        # pygmt.makecpt(cmap=topo_cmap, series=[-1.5, 0.3, 0.01])
        fig.grdimage(grid=grid, shading=dgrid, cmap=topo_cmap)
    else:
        fig.grdimage(grid=grid, cmap=topo_cmap)
    fig.coast(shorelines="1/0.5p", water="white")

    depth_range = [depths.min(), depths.max()]
    if depth_range[0] == depth_range[1]:
        depth_range[0] -= 5
        depth_range[1] += 5

    pygmt.makecpt(cmap="plasma", series=depth_range)

    # Plot earthquakes
    fig.plot(
        x=lons,
        y=lats,
        size=0.02 * 2**mags,
        fill=depths,
        cmap=True,
        style="cc",
        pen="black",
    )
    fig.colorbar(frame='af+lDepth (km)')

    # Plot stations
    if len(station_lons) and len(station_lats):
        fig.plot(
            x=station_lons,
            y=station_lats,
            style="i0.5c",
            fill="royalblue",
            pen="black",
        )

    inset_region = [
        region[0] - (region[1] - region[0]) * inset_multiplier,
        region[1] + (region[1] - region[0]) * inset_multiplier,
        min(90, region[2] - (region[3] - region[2]) * inset_multiplier),
        max(-90, region[3] + (region[3] - region[2]) * inset_multiplier),
    ]

    inset_width = round(width * 0.3, 1)

    max_dim = max(inset_region[1] - inset_region[0], inset_region[3] - inset_region[2])
    inset_mid_lon = inset_region[0] + (inset_region[1] - inset_region[0]) / 2
    inset_mid_lat = inset_region[2] + (inset_region[3] - inset_region[2]) / 2

    inset_frame = True
    if max_dim > 15:
        # Orthographic projection
        inset_proj = f"G{inset_mid_lon}/{inset_mid_lat}/{inset_width}c"
        inset_frame = False
    elif max_dim > 3:
        # Albers
        inset_proj = (
            f"B{inset_mid_lon}/{inset_mid_lat}/{region[2]}/{region[3]}/{inset_width}c"
        )
    else:
        # Mercator
        inset_proj = f"M{inset_width}c"

    with fig.inset(
        position=f"jBL",
        region=inset_region,
        projection=inset_proj,
    ): #+o0.1c"):
        fig.coast(
            land="gray",
            water="white",
            frame=inset_frame,
        )
        rectangle = [[region[0], region[2], region[1], region[3]]]
        fig.plot(data=rectangle, style="r+s", pen="2p,red")

    # Station inset
    with fig.inset(
        position=f"jTL",
        region=large_region,
        projection=f"M{inset_width}c"
    ): #+o0.1c"):
        fig.coast(
            land="gray",
            water="white",
            frame=True,
        )
        rectangle = [[region[0], region[2], region[1], region[3]]]
        fig.plot(data=rectangle, style="r+s", pen="2p,red")
        fig.plot(
            x=station_lons,
            y=station_lats,
            style="i0.5c",
            fill="royalblue",
            pen="black",
        )

    return fig


def aftershock_map(
    catalog: Catalog,
    mainshock: Event,
    relocated_mainshock: Event,
    search_radius: float,
    inventory: Inventory = None,
    pad: float = 50.0,
    width: float = 15.0,
    mainshock_size: float = 0.7,
    inset_multiplier: float = 8,
    topo_res: Union[bool, str] = True,
    topo_cmap: str = "geo",
    hillshade: bool = False,
    timestamp: Union[UTCDateTime, datetime.datetime] = UTCDateTime.now(),
) -> pygmt.Figure:
    """
    Make a basic aftershock map.

    Arguments
    ---------
    catalog:
        Events to plots, scaled by magnitude and colored by depth
    mainshock:
        Mainshock to plot as a gold star
    pad:
        Longitude and latitude pad as percentage of range of latitude and longitude
    width:
        Figure width in cm
    mainshock_size:
        Mainshock glyph size in cm
    inset_multiplier:
        Multiplier for inset map bounds
    topo_res:
        Topography grid resolution - leave set to None to not plot topography, set to True
        to work out resolution, or provide a resolution from here:
        https://www.generic-mapping-tools.org/remote-datasets/earth-relief.html#id1
    topo_cmap:
        Colormap to use for topography - see https://docs.generic-mapping-tools.org/6.2/cookbook/cpts.html for names
    hillshade:
        Whether to plot hillshade or not.
    """
    lats = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).latitude for ev in catalog]
    )
    lons = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).longitude for ev in catalog]
    )
    depths = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).depth / 1000.0 for ev in catalog]
    )
    mags = np.array([get_magnitude_attr(ev, "mag") or 3.0 for ev in catalog])

    times = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).time.datetime for ev in catalog]
    )

    if inventory:
        station_lats = np.array([sta.latitude for net in inventory for sta in net])
        station_lons = np.array([sta.longitude for net in inventory for sta in net])
    else:
        station_lats, station_lons = np.array([]), np.array([])

    mainshock_origin = mainshock.preferred_origin() or mainshock.origins[-1]

    fig = _eq_map(
        lats=lats,
        lons=lons,
        depths=depths,
        mags=mags,
        times=times,
        middle_lon=mainshock_origin.longitude,
        middle_lat=mainshock_origin.latitude,
        search_radius_deg=kilometer2degrees(search_radius),
        station_lons=station_lons,
        station_lats=station_lats,
        width=width,
        pad=pad,
        inset_multiplier=inset_multiplier,
        topo_res=topo_res,
        topo_cmap=topo_cmap,
        hillshade=hillshade,
        timestamp=timestamp,
    )

    # Plot mainshock
    fig.plot(
        x=mainshock_origin.longitude,
        y=mainshock_origin.latitude,
        style=f"a{mainshock_size}c",
        fill="gold",
        pen="black",
    )

    if relocated_mainshock is not None:
        fig.plot(
            x=get_origin_attr(relocated_mainshock, "longitude"),
            y=get_origin_attr(relocated_mainshock, "latitude"),
            style=f"a{mainshock_size}c",
            fill="yellow",
            pen="black"
        )

    # TODO: Emily left an empty plt.text here?
    # plt.text

    return fig


def check_catalog(catalog_all, catalog_geonet):

    # Identify events without origin and/or magnitude in our catalog - better to have cat of no or yes mags?
    catalog_no_origins = Catalog()
    catalog_no_mags = Catalog()
    catalog_origins = Catalog()

    for ev in catalog_all:
        if len(ev.origins) < 1:
            catalog_no_origins.append(ev)
        else:
            catalog_origins.append(ev)
        if len(ev.magnitudes) < 1:
            catalog_no_mags.append(ev)
    no_origin_count = len(catalog_no_origins)
    no_mag_count = len(catalog_no_mags)

    # catalog_origins=[ev for ev in catalog_origins if ev.origins[-1].quality.standard_error <=1.0]

    # identify events without origin and/or magnitudes in GeoNet
    catalog_geonet_no_origins = Catalog()
    catalog_geonet_no_mags = Catalog()

    for ev in catalog_geonet:
        if len(ev.origins) < 1:
            catalog_geonet_no_origins.append(ev)
        if len(ev.magnitudes) < 1:
            catalog_geonet_no_mags.append(ev)
    geonet_no_origin_count = len(catalog_geonet_no_origins)
    geonet_no_mag_count = len(catalog_geonet_no_mags)
    cat_counts = [
        no_origin_count,
        no_mag_count,
        geonet_no_origin_count,
        geonet_no_mag_count,
    ]

    return catalog_origins, cat_counts


def mainshock_mags(mainshock, RT_mainshock):
    ## sort out GeoNet magnitude and depth values
    try:
        geonet_mainshock_mag = round(
            (mainshock.preferred_magnitude() or mainshock.magnitudes[-1]).mag, 2
        )
    except (IndexError, TypeError):
        geonet_mainshock_mag = 0.0

    try:
        geonet_mainshock_mag_uncertainty = round(
            (
                mainshock.preferred_magnitude() or mainshock.magnitudes[-1]
            ).mag_errors.uncertainty,
            2,
        )
    except (IndexError, TypeError):
        geonet_mainshock_mag_uncertainty = 0.0

    try:
        geonet_mainshock_depth = round(
            (mainshock.preferred_origin() or mainshock.origins[-1]).depth / 1000, 1
        )
    except (IndexError, TypeError):
        geonet_mainshock_depth = 0.0

    try:
        geonet_mainshock_depth_uncertainty = round(
            (
                mainshock.preferred_origin() or mainshock.origins[-1]
            ).depth_errors.uncertainty
            / 1000,
            1,
        )
    except (IndexError, TypeError):
        geonet_mainshock_depth_uncertainty = 0.0

    try:
        RT_mainshock_depth = round(
            (RT_mainshock.preferred_origin() or RT_mainshock.origins[-1]).depth / 1000,
            1,
        )
    except (IndexError, TypeError):
        RT_mainshock_depth = geonet_mainshock_depth

    try:
        RT_mainshock_depth_uncertainty = round(
            (
                RT_mainshock.preferred_origin() or RT_mainshock.origins[-1]
            ).depth_errors.uncertainty
            / 1000,
            1,
        )
    except (IndexError, TypeError):
        RT_mainshock_depth_uncertainty = geonet_mainshock_depth_uncertainty

    return (
        geonet_mainshock_mag,
        geonet_mainshock_mag_uncertainty,
        geonet_mainshock_depth,
        geonet_mainshock_depth_uncertainty,
        RT_mainshock_depth,
        RT_mainshock_depth_uncertainty,
    )


###########################################################


def _eq_map_summary(
    catalog: Catalog,
    reference_catalog: Catalog,
    outlier_catalog: Catalog,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    mags: np.ndarray,
    ref_mags: np.ndarray,
    times: np.ndarray,
    lats_o: np.ndarray,
    lons_o: np.ndarray,
    depths_o: np.ndarray,
    mags_o: np.ndarray,
    times_o: np.ndarray,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    magcut,
    pad: float,
    width: float,
    inset_multiplier: float,
    topo_res: float,
    topo_cmap: str,
    hillshade: bool,
    corners: list,
    cat_counts: list,
    colours: str,
    mainshock,
    RT_mainshock,
    search_radius_deg,
) -> pygmt.Figure:
    """ """

    # calculate dimensions:
    all_lons = np.append(lons, lons_o)
    all_lats = np.append(lats, lats_o)
    lat_range = all_lats.max() - all_lats.min()
    lon_range = all_lons.max() - all_lons.min()

    # decide which has bigger range, lats or lons:
    if lat_range > lon_range:
        total_range = lat_range
    else:
        total_range = lon_range

    # map_region = [
    #     all_lons.min() - (total_range * (pad / 100)),
    #     all_lons.max() + (total_range * (pad / 100)),
    #     min(90, all_lats.min() - (total_range * (pad / 100))),
    #     max(-90, all_lats.max() + (total_range * (pad / 100))),
    # ]
    map_region = [
        get_origin_attr(mainshock, "longitude") - search_radius_deg,
        get_origin_attr(mainshock, "longitude") + search_radius_deg,
        min(90, get_origin_attr(mainshock, "latitude") - search_radius_deg),
        max(-90, get_origin_attr(mainshock, "latitude") + search_radius_deg)
    ]

    mags_round_no = np.array([int(str(m).split(".")[0]) for m in mags])
    mags_round_o = np.array([int(str(m).split(".")[0]) for m in mags_o])
    ref_mags_round = np.array([int(str(m).split(".")[0]) for m in ref_mags])
    mags_round = np.append(mags_round_no, mags_round_o)

    # Work out resolution for topography
    plot_topo = True
    if topo_res is (None or False):
        plot_topo = False
    elif topo_res is True:
        min_region_dim = min(
            map_region[1] - map_region[0], map_region[3] - map_region[2]
        )
        if min_region_dim > 10:
            topo_res = "01d"
        elif min_region_dim > 2:
            topo_res = "01m"
        elif min_region_dim > 0.5:
            topo_res = "15s"
        elif min_region_dim > 0.01:
            topo_res = "03s"
        else:
            topo_res = "01s"

    inset_region = [
        map_region[0] - (map_region[1] - map_region[0]) * inset_multiplier / 3,
        map_region[1] + (map_region[1] - map_region[0]) * inset_multiplier / 3,
        min(90, map_region[2] - (map_region[3] - map_region[2]) * inset_multiplier / 3),
        max(
            -90, map_region[3] + (map_region[3] - map_region[2]) * inset_multiplier / 3
        ),
    ]

    inset_width = round(width * 0.2, 1)

    max_dim = max(inset_region[1] - inset_region[0], inset_region[3] - inset_region[2])
    inset_mid_lon = inset_region[0] + (inset_region[1] - inset_region[0]) / 2
    inset_mid_lat = inset_region[2] + (inset_region[3] - inset_region[2]) / 2

    if max_dim > 15:
        # Orthographic projection
        inset_proj = f"G{inset_mid_lon}/{inset_mid_lat}/{inset_width}c"
    elif max_dim > 3:
        # Albers
        inset_proj = f"B{inset_mid_lon}/{inset_mid_lat}/{map_region[2]}/{map_region[3]}/{inset_width}c"
    else:
        # Mercator
        inset_proj = f"M{inset_width}c"

    #######################
    ####               ####
    #### make the plot ####
    ####               ####
    #######################

    fig = pygmt.Figure()

    ##### Bottom Table panel
    # with fig.subplot(nrows=1, ncols=1, figsize=("15c", "5c")):
    #     fig.basemap(region=[0, 15, 0, 5], projection="X?", frame=["tblr"], panel=[0, 0])

    """
    Note, unused section as we don't calculate magnitudes yet.
    # Bottom row, single subplot with table of numbers
    with fig.subplot(nrows=1, ncols=1, figsize=("15c", "5c")):
        fig.basemap(
            region=[0, 15, 0, 5], projection="X?", frame=['tblr'], panel=[0, 0])

        #plot title
        fig.text(text="Magnitude Outputs", position='TC', offset='j0c/0.4c', font="16p,Helvetica-Bold,black")
        #plot text within table
        header_vert=3.2  # y position of table header row
        RT_vert=2.3     # y position of RT-EQcorrscan values row
        Geonet_vert=1.4 # y position of GeoNet values row
        padding=0.3
        column_positions=[1.2, 3, 4.4, 6, 7.6, 9.2, 10.8, 12.4, 14]  # x positions of columns
        column_headers=["Catalogue", "M8+", "M7-7.9", "M6-6.9", "M5-5.9", "M4-4.9", "M3-3.9", "M2-2.9", "M<2"]
        mag_list=[8, 7, 6, 5, 4, 3, 2, 1, 0]
        for i, c in enumerate(column_headers):
            fig.text(text=c, x=column_positions[i], y=header_vert, font="11p,Helvetica-Bold,black",
                      justify='BC')
        fig.text(text='RT-EQcs', x=column_positions[0], y=RT_vert, font="12p,Helvetica,black", justify='BC')
        fig.text(text='GeoNet', x=column_positions[0], y=Geonet_vert, font="12p,Helvetica,black", justify='BC')
        #plot RT mag bin counts
        for j, m in enumerate(mag_list[0:-2]):
            fig.text(text=str(np.count_nonzero(mags_round == m)), x=column_positions[j+1], y=RT_vert,
                     font="12p,Helvetica,black", justify='BC')
        fig.text(text=str(np.count_nonzero(mags_round == 0)+np.count_nonzero(mags_round == 1)),
                 x=column_positions[-1], y=RT_vert, font="12p,Helvetica,black", justify='BC')
        #plot reference mag bin counts
        for j, m in enumerate(mag_list[0:-2]):
            fig.text(text=str(np.count_nonzero(ref_mags_round == m)), x=column_positions[j+1], y=Geonet_vert,
                     font="12p,Helvetica,black", justify='BC')
        fig.text(text=str(np.count_nonzero(ref_mags_round == 0)+np.count_nonzero(ref_mags_round == 1)),
                 x=column_positions[-1], y=Geonet_vert, font="12p,Helvetica,black", justify='BC')
         #plot structure of table
        left=0.2
        right=14.7
        for v in [header_vert, RT_vert, Geonet_vert]:
            fig.plot(x=[left,right], y=[v-padding,v-padding], pen="0.5p,black", projection='X?')
        fig.plot(x=[left,right], y=[header_vert+2*padding,header_vert+2*padding],
                 pen="0.5p,black", projection='X?') #top
        for c in column_positions[2:]:
            fig.plot(x=[c-0.75,c-0.75], y=[header_vert+2*padding, Geonet_vert-padding],
                     pen="0.5p,black", projection='X?')
        fig.plot(x=[column_positions[1]-padding*2, column_positions[1]-padding*2],
                 y=[header_vert+2*padding, Geonet_vert-padding],
                     pen="0.5p,black", projection='X?') #left of value columns
        fig.plot(x=[left, left], y=[header_vert+2*padding, Geonet_vert-padding],
                 pen="0.5p,black", projection='X?') #left
        fig.plot(x=[right, right], y=[header_vert+2*padding, Geonet_vert-padding],
                 pen="0.5p,black", projection='X?') #right

        #plot footnotes
        fig.text(text='Of ' + str(len(catalog)+len(outlier_catalog)) + ' events in the RT-EQcorrscan catalog, '
                 + str(cat_counts[0]) + ' have no origin and ' + str(cat_counts[1]) + ' have no magnitude.',
                 x=0.5, y=Geonet_vert-0.8, font="9p,Helvetica,black", justify='BL')
        fig.text(text='Of ' + str(len(reference_catalog)) + ' events in the GeoNet catalog, '
                 + str(cat_counts[2]) +  ' have no origin and ' + str(cat_counts[3]) + ' have no magnitude.',
                 x=0.5, y=Geonet_vert-1.2, font="9p,Helvetica,black", justify='BL')
        """
    ####### TEXT BOX

    # TODO: Make text box full heigh of figure
    # Move plot origin by 1 cm above the height of the entire figure
    # fig.shift_origin(yshift="h+1c")
    # Top row, one subplot
    with fig.subplot(nrows=1, ncols=1, figsize=("15c", "16c")):
        fig.basemap(region=[0, 12, -1, 15], projection="X?", frame="tblr", panel=[0, 0])
        # plot key output information
        fig.text(
            text="RT-EQcorrscan Aftershock Analysis Outputs",
            position="TC",
            offset="j0c/0.4c",
            font="16p,Helvetica-Bold,black",
        )
        fig.text(textfiles=".plotting_text.txt", angle=True, font=True, justify=True)
        # Plot logos

        # space for RCET logo - need to host local file

        fig.image(
            imagefile=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logos/EQcorrscan_logo.png"),
            position="g10/0+w3c+jCM",
            box=False,
        )
        fig.image(
            imagefile=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logos/VUW Logo.png"),
            position="g10/1.5+w3c+jCM",
            box=False,
        )
        fig.image(
            imagefile=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logos/Earth_sci_NZ.jpeg"),
            position="g10/3.8+w2c+jCM",
            box=False,
        )
        fig.image(
            imagefile=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logos/RCET_logo_transparent.png"),
            position="g10/6+w3c+jCM",
            box=False,
        )

    ########## MAIN MAP

    # Move plot origin by 1 cm right of the width of the entire figure, and back down to zero height
    fig.shift_origin(xshift="w+1c") #, yshift="-6c")
    # one subplot
    with fig.subplot(nrows=1, ncols=1, figsize=("15c", "16c")):
        fig.basemap(
            region=map_region, projection=f"M?", frame=["af", "wsNE"], panel=[0, 0]
        )

        grid = pygmt.datasets.load_earth_relief(resolution=topo_res, region=map_region)
        if hillshade:
            dgrid = pygmt.grdgradient(grid=grid, radiance=[0, 80], normalize=True)
            # pygmt.makecpt(cmap=topo_cmap, series=[-1.5, 0.3, 0.01])
            fig.grdimage(grid=grid, shading=dgrid, cmap=topo_cmap, panel=[0, 0])
        else:
            fig.grdimage(grid=grid, cmap=topo_cmap, panel=[0, 0])
        fig.coast(shorelines="1/0.5p")

        # Plot earthquakes
        if colours == "depth":
            depth_range = [depths.min(), depths.max()]
            if depth_range[0] == depth_range[1]:
                depth_range[0] -= 5
                depth_range[1] += 5
            pygmt.makecpt(cmap="plasma", series=depth_range)
            # plot used earthquakes
            fig.plot(
                x=lons,
                y=lats,
                size=0.02 * 2**mags,
                fill=depths,
                cmap=True,
                style="cc",
                pen="black",
                panel=[0, 0],
            )
            # plot outliers
            if len(lons_o):
                fig.plot(
                    x=lons_o,
                    y=lats_o,
                    size=0.02 * 2**mags_o,
                    fill=depths_o,
                    cmap=True,
                    style="cc",
                    pen="white",
                    transparency=50,
                    panel=[0, 0],
                )
            fig.colorbar(
                position="jBR+o0.88c/0.8c+h+w4c/0.25c+ml",
                frame=["xa10f10+lDepth", "y+lkm"],
            )
        if colours == "magnitude":
            pygmt.makecpt(cmap="jet", continuous=False, series=[0, 9, 1])
            # for used events
            lats_new, lons_new, mags_new = [], [], []
            for i, mag in enumerate(mags_round_no):
                if mag >= magcut:
                    lats_new.append(lats[i])
                    lons_new.append(lons[i])
                    mags_new.append(mag)
            mags_new = np.array(
                mags_new
            )  # would be good to sort all lists, so big mags plot on bottom
            fig.plot(
                x=lons_new,
                y=lats_new,
                size=0.05 * 2**mags_new,
                fill=mags_new,
                cmap=True,
                style="cc",
                transparency="40",
                pen="black",
                panel=[0, 0],
            )

            # for outliers
            lats_new, lons_new, mags_new_o = [], [], []
            for i, mag in enumerate(mags_round_o):
                if mag >= magcut:
                    lats_new.append(lats_o[i])
                    lons_new.append(lons_o[i])
                    mags_new_o.append(mag)
            mags_new_o = np.array(
                mags_new_o
            )  # would be good to sort all lists, so big mags plot on bottom
            fig.plot(
                x=lons_new,
                y=lats_new,
                size=0.05 * 2**mags_new_o,
                fill=mags_new_o,
                cmap=True,
                style="cc",
                transparency="70",
                pen="white",
                panel=[0, 0],
            )
            # fig.colorbar(frame='af+l"Magnitude"')
            fig.colorbar(
                position="jBR+o0.4c/0.8c+h+w4c/0.25c+ml",
                frame=["xa1f1+lMagnitude"],
            )
            fig.text(
                text="Showing "
                + str(len(mags_new) + len(mags_new_o))
                + " earthquakes M"
                + str(magcut)
                + "+",
                position="TR",
                offset="j0.5c/0.5c",
                font="14p,Helvetica,black",
            )

        # plot relocated mainshock
        mainshock_origin = RT_mainshock.preferred_origin() or RT_mainshock.origins[-1]
        fig.plot(
            x=mainshock_origin.longitude,
            y=mainshock_origin.latitude,
            style=f"a1c",
            fill="orange",
            pen="black",
            label=f"Relocated mainshock+S0.3c",
        )
        # plot mainshock
        mainshock_origin = mainshock.preferred_origin() or mainshock.origins[-1]
        fig.plot(
            x=mainshock_origin.longitude,
            y=mainshock_origin.latitude,
            style=f"a0.6c",
            fill="gold",
            pen="black",
            label=f"GeoNet mainshock+S0.3c",
        )
        # Plot stations
        if len(station_lons) and len(station_lats):
            fig.plot(
                x=station_lons,
                y=station_lats,
                style="i0.5c",
                fill="royalblue",
                pen="black",
                label=f"Seismograph+S0.3c",
            )
        # plot legend
        fig.legend(region=map_region, position="JTL+jTL+o1c")

        # plot corners of ellipse
        fig.plot(
            x=[
                corners[0][0],
                corners[1][0],
                corners[2][0],
                corners[3][0],
                corners[0][0],
            ],
            y=[
                corners[0][1],
                corners[1][1],
                corners[2][1],
                corners[3][1],
                corners[0][1],
            ],
            pen="3p,red",
            panel=[0, 0],
        )

        # plotting basemap again to get scale bar on top of plot
        fig.basemap(
            region=map_region,
            projection=f"M?",
            frame=["af", "wsNE"],
            map_scale="jTR+w10k+o0.5c/0.5c+f",
        )

        # plot inset map
        with fig.inset(
            position=f"jBL+w{inset_width}c+o0c/-0.1c+jBL",
            projection=inset_proj,
            box="+p1,black",
        ):
            fig.basemap(
                region=inset_region,
                projection=inset_proj,
                frame=["af", "wsne"],
            )
            fig.coast(
                region=inset_region,
                projection=inset_proj,
                land="gray",
                water="white",
            )
            if len(station_lons) and len(station_lats):
                fig.plot(
                    x=station_lons,
                    y=station_lats,
                    style="i0.1c",
                    fill="royalblue",
                    pen="black",
                )
            rectangle = [[map_region[0], map_region[2], map_region[1], map_region[3]]]
            fig.plot(
                data=rectangle, style="r+s", pen="1.5p,black", projection=inset_proj
            )

    return fig


###############################


def output_aftershock_map(
    catalog: Catalog,
    reference_catalog: Catalog,
    outlier_catalog: Catalog,
    mainshock: Event,
    RT_mainshock: Event,
    corners: list,
    cat_counts: list,
    colours: str,
    magcut: float = 4.0,
    inventory: Inventory = None,
    pad: float = 1,
    width: float = 15.0,
    inset_multiplier: float = 8,
    topo_res: str = None,
    topo_cmap: str = "geo",
    hillshade: bool = False,
    search_radius_deg: float = 2,
) -> pygmt.Figure:
    """
    Make a summary aftershock map for magnitudes

    Arguments
    ---------
    catalog:
        Events detected by RTEQcorrscan to plots
    reference_catalog:
        Reference catalog e.g. GeoNet
    outlier_catalog:
        catalogue of statistically identified outliers
    mainshock:
        Mainshock to plot as a gold star
    corners:
        list of lat lon positions for fault corners
    colours:
        whether to colour earthquakes by magnitude or depths
    magcut:
        minimum magnitude to plot when colours=magnitude
    inventory:
        station inventory, default none
    pad:
        Longitude and latitude pad as percentage of range of latitude and longitude
    width:
        Figure width in cm
    mainshock_size:
        Mainshock glyph size in cm
    inset_multiplier:
        Multiplier for inset map bounds
    topo_res:
        Topography grid resolution - leave set to None to not plot topography, set to True
        to work out resolution, or provide a reolution from here:
        https://www.generic-mapping-tools.org/remote-datasets/earth-relief.html#id1
    topo_cmap:
        Colormap to use for topography - see https://docs.generic-mapping-tools.org/6.2/cookbook/cpts.html for names
    hillshade:
        Whether to plot hillshade or not.
    corners:
        list of tuples defining ellipse corners
    """
    lats = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).latitude for ev in catalog]
    )
    lons = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).longitude for ev in catalog]
    )
    depths = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).depth / 1000.0 for ev in catalog]
    )

    lats_o = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).latitude for ev in outlier_catalog]
    )
    lons_o = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).longitude for ev in outlier_catalog]
    )
    depths_o = np.array(
        [
            (ev.preferred_origin() or ev.origins[-1]).depth / 1000.0
            for ev in outlier_catalog
        ]
    )

    mags = np.array([get_magnitude_attr(ev, "mag") or 2.0 for ev in catalog])
    ref_mags = np.array([get_magnitude_attr(ev, "mag") or 2.0 for ev in reference_catalog])

    times = np.array(
        [(ev.preferred_origin() or ev.origins[-1]).time.datetime for ev in catalog]
    )

    mags_o = np.array([get_magnitude_attr(ev, "mag") or 2.0 for ev in outlier_catalog])

    times_o = np.array(
        [
            (ev.preferred_origin() or ev.origins[-1]).time.datetime
            for ev in outlier_catalog
        ]
    )

    if inventory:
        station_lats = np.array([sta.latitude for net in inventory for sta in net])
        station_lons = np.array([sta.longitude for net in inventory for sta in net])
    else:
        station_lats, station_lons = np.array([]), np.array([])

    fig = _eq_map_summary(
        catalog=catalog,
        reference_catalog=reference_catalog,
        outlier_catalog=outlier_catalog,
        lats=lats,
        lons=lons,
        depths=depths,
        mags=mags,
        ref_mags=ref_mags,
        times=times,
        lats_o=lats_o,
        lons_o=lons_o,
        depths_o=depths_o,
        mags_o=mags_o,
        times_o=times_o,
        station_lons=station_lons,
        station_lats=station_lats,
        width=width,
        pad=pad,
        inset_multiplier=inset_multiplier,
        corners=corners,
        cat_counts=cat_counts,
        topo_res=topo_res,
        topo_cmap=topo_cmap,
        hillshade=hillshade,
        colours=colours,
        magcut=magcut,
        mainshock=mainshock,
        RT_mainshock=RT_mainshock,
        search_radius_deg=search_radius_deg,
    )

    return fig


def extract_xy(catalog, mainshock):
    x = []
    y = []
    for ev in catalog:
        fwd_azimuth, back_azimuth, distance = GEODESIC.inv(
            ev.origins[-1].longitude,
            ev.origins[-1].latitude,
            mainshock.preferred_origin().longitude,
            mainshock.preferred_origin().latitude,
        )
        if back_azimuth < 0:  # put into 0-360 range
            # TODO: Why is this unused?
            back_azimuth = 360 + back_azimuth
        if fwd_azimuth < 0:
            fwd_azimuth = 360 + fwd_azimuth

        if mainshock.preferred_origin().longitude - ev.origins[-1].longitude >= 0:
            eqx = 0 - (distance * math.sin(math.radians(fwd_azimuth)))
        else:
            eqx = distance * math.sin(math.radians(360 - fwd_azimuth))
        x.append(eqx / 1000)
        if mainshock.preferred_origin().latitude - ev.origins[-1].latitude >= 0:
            eqy = distance * math.cos(math.radians(fwd_azimuth))
        else:
            eqy = distance * math.cos(math.radians(360 - fwd_azimuth))
        y.append(-eqy / 1000)
    return np.array(x), np.array(y)


def extract_xy_csv(lats, lons, mainshock):
    x = []
    y = []
    for i, l in enumerate(lats):
        fwd_azimuth, back_azimuth, distance = GEODESIC.inv(
            lons[i],
            l,
            mainshock.preferred_origin().longitude,
            mainshock.preferred_origin().latitude,
        )
        if back_azimuth < 0:  # put into 0-360 range
            # TODO: Why is this unused?
            back_azimuth = 360 + back_azimuth
        if fwd_azimuth < 0:
            fwd_azimuth = 360 + fwd_azimuth
        if mainshock.preferred_origin().longitude - lons[i] >= 0:
            eqx = 0 - (distance * math.sin(math.radians(fwd_azimuth)))
        else:
            eqx = distance * math.sin(math.radians(360 - fwd_azimuth))
        x.append(eqx / 1000)
        if mainshock.preferred_origin().latitude - l >= 0:
            eqy = distance * math.cos(math.radians(fwd_azimuth))
        else:
            eqy = distance * math.cos(math.radians(360 - fwd_azimuth))
        y.append(-eqy / 1000)
    return np.array(x), np.array(y)


def get_cov_ellipse(cov, centre, nstd, ax, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.
    """
    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    ellipse = Ellipse(
        xy=centre, width=width, height=height, angle=np.degrees(theta), **kwargs
    )
    return ax.add_patch(ellipse)


def get_len_theta(x, y, sd):
    """
    Extract the length and azimuth of the confidence ellipsoid given the
    x and y scatter and number of standard deviations
    """
    cov = np.cov(x, y)
    scale_y = np.sqrt(cov[1, 1]) * sd
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    length, width = 2 * sd * np.sqrt(eigvals)
    theta = np.arctan2(vy, vx)
    azimuth = 90 - np.rad2deg(theta)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate directivity scalar and azimuth
    # epicentre and centroid offset:
    ao = np.sqrt((x_mean * x_mean) + (y_mean * y_mean))
    ang = azimuth - (np.rad2deg(np.arctan(x_mean / y_mean)))
    Ao = ao * np.cos(np.deg2rad(ang))
    Ds = 1 / (Ao / (0.5 * length))
    if x_mean > 0 and y_mean > 0:
        db = "NE"
    elif x_mean > 0 and y_mean < 0:
        db = "SE"
    elif x_mean < 0 and y_mean < 0:
        db = "SW"
    elif x_mean < 0 and y_mean > 0:
        db = "NW"

    # check azimuth is 0-180
    if azimuth > 180:
        azimuth = azimuth - 180

    return length, azimuth, cov, width, Ds, db, x_mean, y_mean


def get_len_LOWESS(x, y, frac):
    """
    Calculate the cumulative distance of the lowess fit to coordinates

    x, y:
        Coordinates (cartesian, non-rotated)
    frac:
        Fractal smoothing for lowess

    Returns
        Length of lowess (units same as coordinates input)
    """
    from scipy.spatial import distance
    
    x_95_lims = np.percentile(x, [2.5, 97.5])
    y_95_lims = np.percentile(y, [2.5, 97.5])
    x_95, y_95 = [], []
    for i, j in enumerate(x):
        if (
            j > x_95_lims[0]
            and j < x_95_lims[1]
            and y[i] > y_95_lims[0]
            and y[i] < y_95_lims[1]
        ):
            x_95.append(j)
            y_95.append(y[i])
    smoothed = sm.nonparametric.lowess(exog=y_95, endog=x_95, frac=frac)

    su = []
    for b in range(0, len(smoothed) - 1):
        # Points
        point1 = (smoothed[b][0], smoothed[b][1])
        point2 = (smoothed[b + 1][0], smoothed[b + 1][1])
        # Distance calculation
        D = distance.euclidean(point1, point2)
        su.append(D)

    return sum(su)


def plot_confidence_ellipsoid(
    catalog, x, y, xo, yo, xrm, yrm, sd, mainshock, radius_km, t, LOWESS, frac
):
    fig, ax_nstd = plt.subplots(figsize=(6, 6.7))
    dependency_nstd = [[0.8, 0.75], [-0.2, 0.35]]
    mu = 0, 0
    scale = 8, 5
    ax_nstd.axis("equal")
    plt.xlim([-radius_km * 1.1, radius_km * 1.1])
    plt.ylim([-radius_km * 1.1, radius_km * 1.1])
    plt.xlabel("km")
    plt.ylabel("km")
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    length, azimuth, cov, width, Ds, db, x_mean, y_mean = get_len_theta(x, y, sd=sd)
    # ax_nstd.axvline(c='grey', lw=1)
    # ax_nstd.axhline(c='grey', lw=1)
    import time

    # calculate LOWESS
    x_95_lims = np.percentile(x, [2.5, 97.5])
    y_95_lims = np.percentile(y, [2.5, 97.5])
    x_95, y_95 = [], []
    for i, j in enumerate(x):
        if (
            j > x_95_lims[0]
            and j < x_95_lims[1]
            and y[i] > y_95_lims[0]
            and y[i] < y_95_lims[1]
        ):
            x_95.append(j)
            y_95.append(y[i])

    smoothed = sm.nonparametric.lowess(exog=x_95, endog=y_95, frac=frac)

    # plot the earthquakes as scatter points
    ax_nstd.scatter(x, y, s=1, marker="o", color="#5f8dd3ff")
    # green ax_nstd.scatter(x, y, s=1, marker='o', color='#5aa02cff')
    ## orange ax_nstd.scatter(x, y, s=1, marker='o', color='#ff9955ff')
    circle = plt.Circle(
        (0, 0),
        radius_km,
        linestyle="dotted",
        edgecolor="gray",
        fc="None",
        label=r"Search area",
    )
    ax_nstd.add_patch(circle)
    ax_nstd.scatter(x_mean, y_mean, s=20.0, color="k", label="Aftershock Centroid")
    ax_nstd.scatter(0, 0, s=100.0, color="gold", marker="*", label="GeoNet Mainshock")
    ax_nstd.scatter(
        xrm, yrm, s=100.0, color="orange", marker="*", label="Relocated Mainshock"
    )
    if LOWESS == True and len(x) > 20:
        ax_nstd.plot(smoothed[:, 0], smoothed[:, 1], c="k", label="Lowess")
    get_cov_ellipse(
        cov,
        (x_mean, y_mean),
        nstd=sd,
        ax=ax_nstd,
        label=r"$2\sigma$ ellipse",
        edgecolor="tomato",
        linestyle="-",
        fc="none",
    )
    if len(xo) > 0:
        ax_nstd.scatter(xo, yo, s=1, marker="o", color="#cbdaf0ff", label="Outliers")
        # green ax_nstd.scatter(xo, yo, s=1, marker='o', color='#c6e9afff', label='Outliers')
        # orange ax_nstd.scatter(xo, yo, s=1, marker='o', color='#ffccaaff', label='Outliers')

    # plot on cross-section line
    # set perp_azi to always be 0-180
    perp_azi = azimuth + 90
    if perp_azi > 180:
        perp_azi = perp_azi - 180
    if perp_azi > 360:
        perp_azi = perp_azi - 360
    if perp_azi < 0:
        perp_azi = perp_azi + 180

    # calculate start and end points (up to search radius)
    x_start = 0 - (np.sin(np.deg2rad(180 - perp_azi)) * (width * 2))
    x_end = 0 + (np.sin(np.deg2rad(180 - perp_azi)) * (width * 2))
    y_start = 0 + (np.cos(np.deg2rad(180 - perp_azi)) * (width * 2))
    y_end = 0 - (np.cos(np.deg2rad(180 - perp_azi)) * (width * 2))

    # plot line
    ax_nstd.plot(
        [x_start, x_end],
        [y_start, y_end],
        linestyle="-",
        color="gray",
        label="cross-section",
    )

    ax_nstd.set_title(
        "Time since trigger = "
        + str(t).split(".")[0]
        + " (days, HH:MM:SS)"
        + "\n"
        + "$2\sigma$ length = "
        + str(round(length, 1))
        + " km,    azimuth = "
        + str(int(azimuth)).zfill(3)
        + "$\degree$/ "
        + str(int(azimuth + 180)).zfill(3)
        + "$\degree$"
    )
    ax_nstd.legend(loc="upper left")

    return fig


def to_xr_yr_mainshock(catalog, mainshock, rotation):
    """
    Convert catalog origins into new coordinate system rotated around mainshock

    :param rotation: Degrees clockwise from north to rotate system.
    """
    EARTHRADIUS = 6371  # Global definition of earth radius in km.
    x1, y1 = [], []
    mean_lat = math.radians(
        (mainshock.preferred_origin() or mainshock.origins[-1]).latitude
    )
    for ev in catalog:
        # Degrees east
        xr = math.radians(
            (ev.preferred_origin() or ev.origins[-1]).longitude
            - (mainshock.preferred_origin() or mainshock.origins[-1]).longitude
        )
        xr *= math.cos(mean_lat) * EARTHRADIUS
        yr = math.radians(
            (ev.preferred_origin() or ev.origins[-1]).latitude
            - (mainshock.preferred_origin() or mainshock.origins[-1]).latitude
        )
        yr *= EARTHRADIUS

        s = math.radians(rotation)
        # Rotate through strike (clockwise from North)
        x1.append((xr * math.cos(-s)) + (yr * math.sin(-s)))
        y1.append((-xr * math.sin(-s)) + (yr * math.cos(-s)))
    return x1, y1


def to_xz_yz_z_centroid(catalog, mainshock, azimuth):
    """
    Convert catalog origins into new coordinate system rotated around mainshock
    :param catalog: catalog to be rotated/projected
    :param azimuth: azimuth of ellipse, perpendicular will be used for rotation
    """
    EARTHRADIUS = 6371  # Global definition of earth radius in km.

    #
    # set perp_azi to always be 0-180
    perp_azi = azimuth + 90
    if perp_azi > 180:
        perp_azi = perp_azi - 180
    if perp_azi > 360:
        perp_azi = perp_azi - 360
    if perp_azi < 0:
        perp_azi = perp_azi + 180

    x_z, y_z, z_z = [], [], []
    mean_lat = math.radians(
        (mainshock.preferred_origin() or mainshock.origins[-1]).latitude
    )
    for ev in catalog:
        # Degrees east
        xr = math.radians(
            (ev.preferred_origin() or ev.origins[-1]).longitude
            - (mainshock.preferred_origin() or mainshock.origins[-1]).longitude
        )
        xr *= math.cos(mean_lat) * EARTHRADIUS
        yr = math.radians(
            (ev.preferred_origin() or ev.origins[-1]).latitude
            - (mainshock.preferred_origin() or mainshock.origins[-1]).latitude
        )
        yr *= EARTHRADIUS

        s = math.radians(perp_azi)
        # Rotate through strike (clockwise from North)
        x_z.append((xr * math.cos(-s)) + (yr * math.sin(-s)))
        y_z.append((-xr * math.sin(-s)) + (yr * math.cos(-s)))
        z_z.append((ev.preferred_origin() or ev.origins[-1]).depth / 1000 * -1)

    return x_z, y_z, z_z


def outliers_simple(values, k):
    from scipy.stats import iqr

    IQR = iqr(values)
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    upper = Q3 + k * IQR
    lower = Q1 - k * IQR
    return upper, lower


def find_outliers(x, y, x1, y1, k, catalog_origins):
    """
    Find outliers based on rotated coordinates.

    x, y:
        Non-rotated coordinates
    x1, y1:
        Rotated coordinates
    k:
        Multiplier of inter-quartile range to identify outliers
    catalog_origins:
        Catalog of events (in the same order as coordinates)

    Returns:
    x_o, y_o
        Outlier coordinate (non-rotated)
    x_no, y_no:
        Non-outlier coordinates (non-rotated)
    catalog:
        Catalog on non-outlier events
    catalog_outliers:
        Catalog of outlier events
    """
    from scipy.stats import iqr

    catalog_outliers = []
    catalog = []

    # calculate outliers
    IQR = iqr(x1)
    Q1 = np.percentile(x1, 25)
    Q3 = np.percentile(x1, 75)

    x1_upper = Q3 + k * IQR
    x1_lower = Q1 - k * IQR

    IQR = iqr(y1)
    Q1 = np.percentile(y1, 25)
    Q3 = np.percentile(y1, 75)

    y1_upper = Q3 + k * IQR
    y1_lower = Q1 - k * IQR

    x_no, y_no, x_o, y_o = [], [], [], []
    for i, r in enumerate(x1):
        if r > x1_upper or r < x1_lower or y1[i] < y1_lower or y1[i] > y1_upper:
            # define as outlier:
            x_o.append(x[i])
            y_o.append(y[i])
            catalog_outliers.append(catalog_origins[i])
        else:
            x_no.append(x[i])
            y_no.append(y[i])
            catalog.append(catalog_origins[i])

    return x_o, y_o, x_no, y_no, catalog, catalog_outliers


def plot_confidence_ellipsoid_vertical(
    catalog, x, z, xo, yo, xrm, zrm, z_m, t, sd, mainshock, w, LOWESS, frac
):
    fig, ax_nstd = plt.subplots(figsize=(6, 8))
    # dependency_nstd = [[0.8, 0.75],[-0.2, 0.35]]
    # mu = 0, 0
    # scale = 8, 5
    ax_nstd.axis("equal")
    plt.xlim([-w * 2, w * 2])
    # plt.xlim([-25, 25])
    # plt.ylim([-40,-15])
    plt.xlabel("Horizontal distance along cross-section from mainshock, km")
    plt.ylabel("Depth, km")
    x_mean = np.mean(x)
    y_mean = np.mean(z)
    length, azimuth, cov, width, Ds, db, x_mean, y_mean = get_len_theta(x, z, sd=sd)
    # ax_nstd.axvline(c='grey', lw=1)
    # ax_nstd.axhline(c='grey', lw=1)
    import time

    # calculate LOWESS
    x_95_lims = np.percentile(x, [2.5, 97.5])
    z_95_lims = np.percentile(z, [2.5, 97.5])
    x_95, z_95 = [], []
    for i, j in enumerate(x):
        if (
            j > x_95_lims[0]
            and j < x_95_lims[1]
            and z[i] > z_95_lims[0]
            and z[i] < z_95_lims[1]
        ):
            x_95.append(j)
            z_95.append(z[i])

    smoothed = sm.nonparametric.lowess(exog=z_95, endog=x_95, frac=frac)

    # plot the earthquakes as scatter points
    ax_nstd.scatter(x, z, s=1, marker="o", color="#5f8dd3ff")
    # green ax_nstd.scatter(x, z, s=1, marker='o', color='#5aa02cff')
    # orange ax_nstd.scatter(x, z, s=1, marker='o', color='#ff9955ff')
    ax_nstd.scatter(
        xrm, zrm, s=100.0, color="orange", marker="*", label="Relocated_Mainshock"
    )
    ax_nstd.scatter(x_mean, y_mean, s=20.0, color="k", label="Aftershock centroid")
    ax_nstd.scatter(0, -z_m, s=100.0, color="gold", marker="*", label="Mainshock")
    get_cov_ellipse(
        cov,
        (x_mean, y_mean),
        nstd=sd,
        ax=ax_nstd,
        label=r"$2\sigma$ ellipse",
        edgecolor="tomato",
        linestyle="-",
        fc="none",
    )
    # get_cov_ellipse(cov, (x_mean, y_mean), nstd=1, ax=ax_nstd, label=r'$1\sigma$', edgecolor='firebrick', linestyle='--', fc='none')
    # get_cov_ellipse(cov, (x_mean, y_mean), nstd=3, ax=ax_nstd, label=r'$3\sigma$', edgecolor='lightpink', linestyle='--', fc='none')
    if LOWESS == True and len(x) > 20:
        ax_nstd.plot(smoothed[:, 1], smoothed[:, 0], c="k", label="Lowess")
    # if len(xo) > 0:
    #    ax_nstd.scatter(xo, yo, s=1, marker='x', color='lightblue', label='Outliers')

    ax_nstd.set_title(
        "Time since trigger = "
        + str(t).split(".")[0]
        + " (days, HH:MM:SS)"
        + "\n"
        + "$2\sigma$ length (width) = "
        + str(round(length, 1))
        + " km,   Dip = "
        + str(int(90 - azimuth)).zfill(3)
        + "$\degree$"
    )
    # ax_nstd.legend()

    return fig


def ellipse_plots(
    catalog_origins,
    mainshock,
    relocated_mainshock,
    fabric_angle,
    IQR_k,
    ellipse_std,
    lowess,
    lowess_f,
    radius_km,
):

    # Calculate metrics
    x, y = extract_xy(catalog=catalog_origins, mainshock=mainshock)
    x_rm, y_rm = extract_xy(catalog=[relocated_mainshock], mainshock=mainshock)
    z_rm = relocated_mainshock.origins[-1].depth / 1000

    # Identifying outliers
    # rotate into new fabric aligned coordinate system
    x1, y1 = to_xr_yr_mainshock(
        catalog=catalog_origins, mainshock=mainshock, rotation=fabric_angle
    )
    # calculate outliers
    x_o, y_o, x_no, y_no, catalog, catalog_outliers = find_outliers(
        x=x, y=y, x1=x1, y1=y1, k=IQR_k, catalog_origins=catalog_origins
    )

    length, azimuth, cov, width, Ds, db, x_mean, y_mean = get_len_theta(
        x=x_no, y=y_no, sd=ellipse_std
    )
    ellipse_map_plot = plot_confidence_ellipsoid(
        catalog=catalog,
        x=x_no,
        y=y_no,
        xo=x_o,
        yo=y_o,
        xrm=x_rm,
        yrm=y_rm,
        LOWESS=lowess,
        frac=lowess_f,
        mainshock=mainshock,
        sd=ellipse_std,
        radius_km=radius_km,
        t=catalog[-1].origins[-1].time.datetime - mainshock.origins[-1].time.datetime,
    )
    len_lowess = get_len_LOWESS(x=x_no, y=y_no, frac=lowess_f)
    # write figure to file
    # need to colour by depth

    ###### VERTICAL X-SEC #####
    depths_good = []
    catalog_good_depths = []
    for ev in catalog:
        depths_good.append((ev.preferred_origin() or ev.origins[-1]).depth)

    upper, lower = outliers_simple(values=depths_good, k=IQR_k)

    for i, d in enumerate(depths_good):
        if d < upper and d > lower:
            catalog_good_depths.append(catalog[i])

    # rotate into new dip-parallel aligned coordinate system
    x_z, y_z, z_z = to_xz_yz_z_centroid(
        catalog=catalog_good_depths, mainshock=mainshock, azimuth=azimuth
    )
    x_zrm, y_zrm, z_zrm = to_xz_yz_z_centroid(
        catalog=[relocated_mainshock], mainshock=mainshock, azimuth=azimuth
    )

    length_z, azimuth_z, cov_z, width_z, Ds_z, db_z, x_mean_z, y_mean_z = get_len_theta(
        x=y_z, y=z_z, sd=ellipse_std
    )
    dip = 90 - azimuth_z
    ellipse_xsection = plot_confidence_ellipsoid_vertical(
        catalog=catalog_good_depths,
        x=y_z,
        z=z_z,
        xo=[],
        yo=[],
        xrm=y_zrm[0],
        zrm=z_rm * -1,
        z_m=mainshock.preferred_origin().depth / 1000.,
        mainshock=mainshock,
        sd=ellipse_std,
        LOWESS=True,  # TODO: Emily has this hard-coded to True rather than the var?
        frac=lowess_f,
        t=catalog[-1].origins[-1].time.datetime - mainshock.origins[-1].time.datetime,
        w=width,
    )

    ellipse_output = {
        "length": length,
        "length_lowess": len_lowess,
        "azimuth": azimuth,
        "width": width,
        "length_z": length_z,
        "dip": dip,
        "x_mean": x_mean,
        "y_mean": y_mean,
    }

    return ellipse_output, catalog_outliers, ellipse_map_plot, ellipse_xsection


def make_scaled_mag_list(length, width, rupture_area, scaled_mag_relation):

    mag_list, slip_list, ref_list = [], [], []

    if rupture_area == "rectangle":
        A = length * width
    else:
        A = 0.25 * length * width * math.pi  # ELLIPTICAL AREA

    ##### General Crustal
    # Wells coppersmith
    # mag_list.append(round(math.log10(length)*1.49+4.38,2))

    # Thingbaijam 2017
    mag_list.append((math.log10(A) + 3.292) / 0.949)
    slip_list.append("Crustal")
    ref_list.append("Thingbaijam et al. (2017)")

    # Stirling NSHM 2023
    mag_list.append(math.log10(A) + 4.1)
    slip_list.append("Crustal")
    ref_list.append("Stirling et al. (2023)")

    ###  CRUSTAL SS ####
    # Leonard interplate ss/Leonard 2014
    mag_list.append(((7 + (1.5 * (math.log10(A * 1000000)) + 6.087)) - 16.05) / 1.5)
    slip_list.append("SS")
    ref_list.append("Leonard et al. (2014), Interplate")

    # Leonard intraplate SS
    mag_list.append(((7 + (1.5 * (math.log10(A * 1000000)) + 6.37)) - 16.05) / 1.5)
    slip_list.append("SS")
    ref_list.append("Leonard et al. (2014), Intraplate")

    # cheng and cheng et al. (2020))
    mag_list.append(3.77 + (0.98 * (math.log10(A))))
    slip_list.append("SS")
    ref_list.append("Cheng et al. (2020)")

    # Yen and Ma SS
    mag_list.append((((((math.log10(A)) + 14.77) / 0.92) + 7) - 16.05) / 1.5)
    slip_list.append("SS")
    ref_list.append("Yen & Ma (2011)")

    # Thingbaijam and Thingbaijam et al. (2017) SS
    mag_list.append(round((math.log10(A) + 3.486) / 0.942))
    slip_list.append("SS")
    ref_list.append("Thingbaijam et al. (2017)")

    # Shaw and Shaw (2013) ss
    N = math.sqrt(A / (width * width))
    D = (1 + ((A) / (7.4 * (width * width)))) / 2
    mag_list.append((2 / 3) * (((math.log10(N / D)))) + (math.log10(A)) + 3.98)
    slip_list.append("SS")
    ref_list.append("Shaw (2013)")

    # Stafford SS
    mag_list.append(math.log10(A) - (-4.0449))
    slip_list.append("SS")
    ref_list.append("Stafford et al. (2014)")

    # Stirling NSHM 2023 generic mean (SS, RV and NN)
    mag_list.append(math.log10(A) + 4.2)
    slip_list.append("SS")
    ref_list.append("Stirling et al. (2023), Generic")

    #### CRUSTAL REVERSE SLIP ###
    # stafford 2014
    mag_list.append(math.log10(A) + 4.028)
    slip_list.append("RV")
    ref_list.append("Stafford et al. (2014)")

    # Thingbaijam 2027
    mag_list.append((math.log10(A) + 4.362) / 1.049)
    slip_list.append("RV")
    ref_list.append("Thingbaijam et al. (2017)")

    # Leonard interplate DS
    mag_list.append((((7 + (1.5 * math.log10(A * 1000000)) + 6.098)) - 16.05) / 1.5)
    slip_list.append("DS")
    ref_list.append("Leonard et al. (2014), Interplate")

    # Leonard Intraplate DS
    mag_list.append(((7 + (1.5 * (math.log10(A * 1000000)) + 6.38)) - 16.05) / 1.5)
    slip_list.append("DS")
    ref_list.append("Leonard et al. (2014), Intraplate")

    #### CRUSTAL NORMAL SLIP ###
    # Stafford 2014
    mag_list.append(math.log10(A) + 4.0165)
    slip_list.append("NN")
    ref_list.append("Stafford et al. (2014)")

    # Thingbaijam 2017
    mag_list.append((math.log10(A) + 2.551) / 0.808)
    slip_list.append("NN")
    ref_list.append("Thingbaijam et al. (2017)")

    ##### Subduction interface ####
    # Allen and Hayes 2017
    if A <= 74000:
        mag_list.append((math.log10(A) + 5.62) / 1.22)
        slip_list.append("SI")
    else:
        mag_list.append((math.log10(A) - 2.23) / 0.32)
        slip_list.append("SI")
    ref_list.append("Allen & Hayes (2017)")

    # Skarlatoudis SI
    SMo = ((math.log10(A) - math.log10(0.000000000177)) * 3) / 2
    mag_list.append((((SMo + 7) - 16.05) / 1.5))
    slip_list.append("SI")
    ref_list.append("Skarlatoudis et al. (2016)")

    # Thingbaijam SI
    mag_list.append((((math.log10(A)) - (-3.292)) / 0.949))
    slip_list.append("SI")
    ref_list.append("Thingbaijam et al. (2017)")

    # Murotani SI
    MMo = ((math.log10(A) - math.log10(0.000000000134)) * 3) / 2
    mag_list.append((((MMo + 7) - 16.05) / 1.5))
    slip_list.append("SI")
    ref_list.append("Murotani et al. (2013)")

    # Strasser SI
    mag_list.append((0.846 * (math.log10(A))) + 4.41)
    slip_list.append("SI")
    ref_list.append("Strasser et al. (2010)")

    # Stirling generic SI 2023
    mag_list.append(((math.log10(A)) * 1) + 4)
    slip_list.append("SI")
    ref_list.append("Stirling et al. (2023)")

    scaled_mag = round(mag_list[scaled_mag_relation], 2)

    return mag_list, slip_list, ref_list, scaled_mag


def plot_scaled_magnitudes(mag_list, scaled_mag, slip_list, ref_list, Mw, mainshock):
    fig = plt.figure()
    fig.set_size_inches(12, 8)
    # define axes limits + labels
    mag_listfull = mag_list
    for mag in mainshock.magnitudes:
        mag_listfull.append(mag.mag)
    max_mag = max(mag_listfull) + 0.5
    min_mag = min(mag_listfull) - 0.5
    ax = fig.add_subplot()
    ax.set_ylim([min_mag, max_mag])
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Slip Style")
    # map magnitudes to references
    mapping = {
        "Thingbaijam et al. (2017)": "o",
        "Stafford et al. (2014)": "x",
        "Allen & Hayes (2017)": ".",
        "Cheng et al. (2020)": "X",
        "Leonard et al. (2014), Interplate": ">",
        "Leonard et al. (2014), Intraplate": "<",
        "Murotani et al. (2013)": "s",
        "Shaw (2013)": "p",
        "Skarlatoudis et al. (2016)": "*",
        "Stirling et al. (2023)": "+",
        "Stirling et al. (2023), Generic": "P",
        "Strasser et al. (2010)": "D",
        "Yen & Ma (2011)": "d",
    }
    colors = {
        "NN": "royalblue",
        "Crustal": "brown",
        "SS": "mediumseagreen",
        "RV": "tomato",
        "SI": "blueviolet",
        "DS": "gray",
    }
    # plot magnitudes
    try:
        ax.fill_between(
            slip_list,
            mainshock.preferred_magnitude().mag
            - mainshock.preferred_magnitude().mag_errors.uncertainty,
            mainshock.preferred_magnitude().mag
            + mainshock.preferred_magnitude().mag_errors.uncertainty,
            alpha=0.4,
            color="lightpink",
            label="Preferred M unc",
        )
    except:  # what is this except catching?!
        x = 1
    for i in range(len(slip_list)):
        ax.scatter(
            slip_list[i],
            mag_list[i],
            marker=mapping[ref_list[i]],
            color=colors[slip_list[i]],
            label=ref_list[i],
        )
    if Mw is not None:
        ax.axhline(Mw, color="r", linestyle="solid", label="Mw")
    mag_cols = ["blue", "skyblue", "darkblue", "cornflower", "lightblue", "navy"]
    for i, m in enumerate(mainshock.magnitudes):
        if m.magnitude_type in ["MLv", "ML"]:
            ax.axhline(
                m.mag,
                color=mag_cols[i],
                linestyle="solid",
                label="GeoNet " + m.magnitude_type,
            )
    ax.axhline(
        mainshock.preferred_magnitude().mag,
        color="black",
        linestyle="solid",
        label="GeoNet Preferred M",
    )
    ### add labels to top left corner
    try:
        ax.text(
            0,
            max_mag - 0.1,
            "Preferred GeoNet Mag ("
            + mainshock.preferred_magnitude().magnitude_type
            + ") = "
            + str(round(mainshock.preferred_magnitude().mag, 2))
            + " (+/- "
            + str(round(mainshock.preferred_magnitude().mag_errors.uncertainty, 2))
            + ")",
            color="black",
            fontsize=11.0,
            horizontalalignment="left",
        )
    except:  # What is this except catching!?
        ax.text(
            0,
            max_mag - 0.1,
            "Preferred GeoNet Mag = N/A",
            color="black",
            fontsize=11.0,
            horizontalalignment="left",
        )
    ax.text(
        0,
        max_mag - 0.2,
        "Preferred Scaled Mag (Mas) = " + str(scaled_mag),
        color="black",
        fontsize=11.0,
        horizontalalignment="left",
    )
    if Mw:
        ax.text(
            0,
            max_mag - 0.3,
            "Moment Magnitude (Mw) = " + str(Mw),
            color="black",
            fontsize=11.0,
            horizontalalignment="left",
        )
    else:
        ax.text(
            0,
            max_mag - 0.3,
            "Moment Magnitude (Mw) = N/A",
            color="black",
            fontsize=11.0,
            horizontalalignment="left",
        )
    # Make space for the legend
    fig.subplots_adjust(right=0.7)
    fig.legend(bbox_to_anchor=(0.73, 0.87), loc="upper left")
    return fig


def focal_sphere_plots(
    azimuth: float,
    dip: float,
    MT_NP1: Tuple[float, float, float],
    MT_NP2: Tuple[float, float, float],
):
    """
    Plot focal sphere solutions.

    Parameters
    ----------
    azimuth
        Azimuth of plane to compare to moment tensor
    dip
        Dip of plane to compare to moment tensor
    MT_NP1
        Strike, dip, rake of preferred moment tensor nodal plane 1
    MT_NP2
        Strike, dip, rake of preferred moment tensor nodal plane 2

    Returns
    -------
    Figure.
    """
    import matplotlib.pyplot as plt
    import mplstereonet
    from obspy.imaging.beachball import beach

    # plot stereonet of fault plane
    fig, axs = plt.subplot_mosaic(
        [["s", "s", "b"], ["s", "s", "."]],
        per_subplot_kw={"s": {"projection": "stereonet"}},
        figsize=(8, 8),
    )
    ax = axs["s"]
    ax.plane(
        azimuth + 180,
        dip,
        c="b",
        label="Best fit aftershock plane %03d/%02d" % (azimuth + 180, dip),
    )
    if MT_NP1:
        ax.plane(
            MT_NP1[0],
            MT_NP1[1],
            c="r",
            label="Moment Tensor  %03d/%02d" % (MT_NP1[0], MT_NP1[1]),
        )
        ax.plane(MT_NP2[0], MT_NP2[1], c="r")
    ax.grid()
    ax.legend()

    # Focal mechanism parameters (strike, dip, rake)
    if MT_NP1:
        focal_mechanism_1 = (MT_NP1[0], MT_NP1[1], MT_NP1[2])  # Example values
        bb_ax = axs["b"]
        # Plot the first beachball
        bb1 = beach(
            focal_mechanism_1,
            xy=(0.3, 0.5),
            width=0.2,
            facecolor="grey",
            edgecolor="k",
            axes=None,
        )
        bb_ax.add_collection(bb1)

        # Set plot limits and remove axis ticks
        bb_ax.set_xlim(0, 1)
        bb_ax.set_ylim(0, 1)
        bb_ax.set_xticks([])
        bb_ax.set_yticks([])
        bb_ax.axis("equal")

        # Add title and show the plot
        bb_ax.set_title("GeoNet Moment Tensor")
    axs["b"].axis("off")
    return fig


################################################


def summary_files(
    eventid,
    current_time,
    elapsed_secs,
    catalog_RT,
    cat_counts,
    mainshock_cluster,
    catalog_geonet,
    catalog_outliers,
    length,
    length_lowess,
    azimuth,
    dip,
    length_z,
    scaled_mag,
    lowess_scaled_mag,
    geonet_mainshock_mag,
    geonet_mainshock_mag_uncertainty,
    mean_depth,
    RT_mainshock_depth,
    RT_mainshock_depth_uncertainty,
    geonet_mainshock_depth,
    geonet_mainshock_depth_uncertainty,
    output_dir,
):

    # list of column names
    field_names = [
        "Trigger_ID",
        "Timestamp",
        "Elapsed_secs",
        "N_evs",
        "No_origin",
        "No_mag",
        "N_mainshock_cluster",
        "N_geonet_evs",
        "Geonet_no_origin",
        "Geonet_no_mag",
        "Length",
        "Length_lowess",
        "Azimuth",
        "Dip",
        "Width",
        "Scaled_mag",
        "Lowess_Scaled_mag",
        "GeoNet_mag",
        "GeoNet_mag_unc",
        "Mean_depth",
        "Relocated_depth",
        "Relocated_depth_unc",
        "GeoNet_ms_depth",
        "GeoNet_ms_depth_unc",
    ]

    # Dictionary that we want to add as a new row
    output_dictionary = {
        "Trigger_ID": eventid,
        "Timestamp": str(current_time).split(".")[0],
        "Elapsed_secs": str(elapsed_secs),
        "N_evs": str(len(catalog_RT)),
        "No_origin": str(cat_counts[0]),
        "No_mag": str(cat_counts[1]),
        "N_mainshock_cluster": str(len(mainshock_cluster)),
        "N_geonet_evs": str(len(catalog_geonet)),
        "Geonet_no_origin": str(cat_counts[2]),
        "Geonet_no_mag": str(cat_counts[3]),
        "Length": str(round(length, 1)),
        "Length_lowess": str(round(length_lowess, 1)),
        "Azimuth": str(int(azimuth)).zfill(3),
        "Dip": str(int(dip)).zfill(2),
        "Width": str(round(length_z, 1)),
        "Scaled_mag": str(scaled_mag),
        "Lowess_Scaled_mag": str(lowess_scaled_mag),
        "GeoNet_mag": str(geonet_mainshock_mag),
        "GeoNet_mag_unc": str(geonet_mainshock_mag_uncertainty),
        "Mean_depth": str(mean_depth),
        "Relocated_depth": str(RT_mainshock_depth),
        "Relocated_depth_unc": str(RT_mainshock_depth_uncertainty),
        "GeoNet_ms_depth": str(geonet_mainshock_depth),
        "GeoNet_ms_depth_unc": str(geonet_mainshock_depth_uncertainty),
    }

    f_output = (
        output_dir + "/output_metrics_summary_file.csv"
    )  ### NOTE: THIS HAS TO STAY IN LATEST FOLDER
    if os.path.isfile(f_output):
        with open(f_output, "a") as f:
            dictwriter_object = csv.DictWriter(f, fieldnames=field_names)
            dictwriter_object.writerow(output_dictionary)
            f.close()
    else:
        with open(f_output, "w") as f:
            writer = csv.writer(f)
            writer.writerow(field_names)
            dictwriter_object = csv.DictWriter(f, fieldnames=field_names)
            dictwriter_object.writerow(output_dictionary)
            f.close()

    # write individual csv file
    f_output_t = (
        output_dir + "/output_metrics_file_" + str(current_time).split(".")[0] + ".csv"
    )
    with open(f_output_t, "w") as f:
        writer = csv.writer(f)
        writer.writerow(field_names)
        dictwriter_object = csv.DictWriter(f, fieldnames=field_names)
        dictwriter_object.writerow(output_dictionary)
        f.close()

    # Use a timedelta for nicer string formatting
    elapsed_time = datetime.timedelta(seconds=elapsed_secs)

    # write out file with text inputs for plotting next to aftershock map
    font_size = "12"
    with open(".plotting_text.txt", "w") as f:
        f.write("0.2 11 0 " + font_size + "p,Helvetica-Bold,black ML Trigger_ID: \n")
        f.write("5.5 11 0 " + font_size + "p,Helvetica,black ML " + eventid + "\n")
        f.write("0.2 10 0 " + font_size + "p,Helvetica-Bold,black ML Timestamp: \n")
        f.write(
            "5.5 10 0 "
            + font_size
            + "p,Helvetica,black ML "
            + str(current_time).split(".")[0]
            + " (UTC)"
            + "\n"
        )
        f.write(
            "0.2 9 0 " + font_size + "p,Helvetica-Bold,black ML Elapsed time: \n"
        )
        f.write(
            "5.5 9 0 " + font_size + "p,Helvetica,black ML " + str(elapsed_time) + "\n"
        )
        f.write("0.2 8 0 " + font_size + "p,Helvetica-Bold,black ML Total events: \n")
        f.write(
            "5.5 8 0 "
            + font_size
            + "p,Helvetica,black ML "
            + str(len(catalog_RT))
            + " ("
            + str(len(catalog_outliers))
            + " outliers)\n"
        )
        f.write("0.2 7 0 " + font_size + "p,Helvetica-Bold,black ML Fault Length: \n")
        f.write(
            "5.5 7 0 "
            + font_size
            + "p,Helvetica,black ML "
            + str(round(length, 1))
            + " km"
            + "\n"
        )
        f.write(
            "0.2 6 0 " + font_size + "p,Helvetica-Bold,black ML Fault Strike/Dip: \n"
        )
        f.write(
            "5.5 6 0 "
            + font_size
            + "p,Helvetica,black ML "
            + str(int(azimuth)).zfill(3)
            + "/"
            + str(int(dip)).zfill(2)
            + " degrees"
            + "\n"
        )
        f.write("0.2 5 0 " + font_size + "p,Helvetica-Bold,black ML Fault Width: \n")
        f.write(
            "5.5 5 0 "
            + font_size
            + "p,Helvetica,black ML "
            + str(round(length_z, 1))
            + " km"
            + "\n"
        )
        f.write(
            "0.2 4 0 " + font_size + "p,Helvetica-Bold,black ML Scaled magnitude: \n"
        )
        f.write(
            "5.5 4 0 " + font_size + "p,Helvetica,black ML " + str(scaled_mag) + "\n"
        )
        f.write(
            "0.2 3 0 " + font_size + "p,Helvetica-Bold,black ML GeoNet magnitude: \n"
        )
        f.write(
            "5.5 3 0 "
            + font_size
            + "p,Helvetica,black ML "
            + str(geonet_mainshock_mag)
            + " (+/- "
            + str(geonet_mainshock_mag_uncertainty)
            + ")\n"
        )
        f.write(
            "0.2 2 0 "
            + font_size
            + "p,Helvetica-Bold,black ML Mean aftershock depth: \n"
        )
        f.write(
            "5.5 2 0 "
            + font_size
            + "p,Helvetica,black ML "
            + f"{mean_depth:.2f}"
            + " km \n"
        )
        f.write(
            "0.2 1 0 "
            + font_size
            + "p,Helvetica-Bold,black ML Relocated mainshock depth: \n"
        )
        f.write(
            "5.5 1 0 "
            + font_size
            + "p,Helvetica,black ML "
            + str(RT_mainshock_depth)
            + " (+/- "
            + str(RT_mainshock_depth_uncertainty)
            + ") km \n"
        )
        f.write(
            "0.2 0 0 "
            + font_size
            + "p,Helvetica-Bold,black ML GeoNet mainshock depth: \n"
        )
        f.write(
            "5.5 0 0 "
            + font_size
            + "p,Helvetica,black ML "
            + str(geonet_mainshock_depth)
            + " (+/- "
            + str(geonet_mainshock_depth_uncertainty)
            + ") km"
        )

    return output_dictionary, f_output


def plot_geometry_with_time(
    times,
    events,
    mainshock_cluster,
    geonet_events,
    mean_depths,
    Relocated_depths,
    Relocated_depth_uncerts,
    lengths,
    lowess_lengths,
    azimuths,
    dips,
    mags,
    lowess_mags,
    GeoNet_mags,
    GeoNet_mags_uncerts,
    GeoNet_depths,
    GeoNet_depth_uncerts,
    log,
    mainshock,
    **kwargs,
):

    pclength = lengths[-1] * 0.1
    pcazimuth = 10
    fig = plt.figure()
    fig.set_size_inches(16, 14)
    fig.patch.set_facecolor("white")

    # plot summary statistics text
    ax0 = fig.add_subplot(4, 2, 1)
    ax0.axis("off")

    # Plot x-axis upper limit
    max_time = max(86400, times[-1])

    # Convert trigger time to datetime for string formatting
    elapsed_time = datetime.timedelta(seconds=times[-1])

    summary_text = [
        ["Trigger ID:", mainshock.resource_id.id.split("/")[-1]],
        ["Time since trigger:", f"{elapsed_time}"],
        ["Total events:", f"{events[-1]}"],
        ["Events in mainshock cluster:", f"{mainshock_cluster[-1]}"],
        ["Fault length:", f"{lengths[-1]:.2f} km"],
        ["Fault azimuth:", f"{azimuths[-1]:.2f} degrees"],
        ["Fault dip:", f"{dips[-1]:.2f} degrees"],
        ["Scaled magnitude:", f"{mags[-1]:.1f}"],
        ["Lowess scaled magnitude:", f"{lowess_mags[-1]:.1f}"],
        ["GeoNet magnitude:", f"{GeoNet_mags[-1]:.1f}"],
        ["Mean aftershock depth:", f"{mean_depths[-1]:.2f} km"],
        ["RT/GN mainshock depth:", f"{Relocated_depths[-1]:.1f}/{GeoNet_depths[-1]:.1f} km"]
    ]

    table = ax0.table(
        cellText=summary_text,
        cellLoc="left",
        edges="open",
        loc="center",
    )

    # Make left column bold
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))

    ax0.text(
        0.5,
        1,
        "RT-EQcorrscan Aftershock Analysis Outputs",
        weight="bold",
        color="black",
        fontsize=14.0,
        horizontalalignment="center",
    )

    def _additional_plot_elements(ax, xlim_upper=86400):
        ax.axvline(x=60, color="r", linestyle="-")
        ax.axvline(x=360, color="r", linestyle="-.")
        ax.axvline(x=3600, color="r", linestyle="dashed")
        ax.axvline(x=86400, color="r", linestyle="dotted")
        ax.set_xlim(10, xlim_upper)

    # Plot events
    ax1 = fig.add_subplot(4, 2, 3)
    if log == True:
        ax1.set_xscale("log")
    ax1.plot(
        times,
        events,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="N events (RT-EQcorrscan)",
        color="firebrick",
    )
    ax1.plot(
        times,
        mainshock_cluster,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="N events (mainshock cluster)",
        color="lightcoral",
    )
    ax1.plot(
        times,
        geonet_events,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=2,
        label="N events (GeoNet)",
        color="darkgray",
    )
    ax1.title.set_text("Events")
    ax1.text(47, min(events + geonet_events), "1 minute", rotation=90,
            color="red")
    ax1.text(270, min(events + geonet_events), "10 minutes", rotation=90,
            color="red")
    ax1.text(2700, min(events + geonet_events), "1 hour", rotation=90,
            color="red")
    ax1.text(66400, min(events + geonet_events), "1 day", rotation=90,
            color="red")
    _additional_plot_elements(ax=ax1, xlim_upper=max_time)
    ax1.legend(loc="upper left")
    ax1.set_ylabel("Number of detected events")

    # Plot depths
    ax2 = fig.add_subplot(4, 2, 5)
    if log == True:
        ax2.set_xscale("log")

    ax2.title.set_text("Depths")
    _additional_plot_elements(ax=ax2, xlim_upper=max_time)
    # Plot depths around the other way
    ax2.invert_yaxis()
    ax2.plot(
        times,
        mean_depths,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="Mean aftershock depths",
        color="blueviolet",
    )
    ax2.plot(
        times,
        Relocated_depths,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="Relocated mainshock depth",
        color="plum",
    )
    RT_max_depths, RT_min_depths = [], []
    for i, m in enumerate(Relocated_depths):
        RT_max_depths.append(m + Relocated_depth_uncerts[i])
        RT_min_depths.append(m - Relocated_depth_uncerts[i])
    ax2.fill_between(times, RT_min_depths, RT_max_depths, alpha=0.2, color="plum")
    ax2.plot(
        times,
        GeoNet_depths,
        linestyle="-",
        marker=".",
        markersize="0.01",
        label="GeoNet preferred mainshock depth",
        color="darkgray",
    )
    geonet_max_depths, geonet_min_depths = [], []
    for i, m in enumerate(GeoNet_depths):
        geonet_max_depths.append(m + GeoNet_depth_uncerts[i])
        geonet_min_depths.append(m - GeoNet_depth_uncerts[i])
    ax2.fill_between(
        times, geonet_min_depths, geonet_max_depths, alpha=0.2, color="grey"
    )
    ax2.legend()
    ax2.set_ylabel("Depth (km)")

    # plot lengths
    ax3 = fig.add_subplot(4, 2, 6)
    if log == True:
        ax3.set_xscale("log")
    ax3.title.set_text("Length")
    ax3.plot(
        times,
        lengths,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="2$\sigma$ Length",
        color="lightblue",
    )
    _additional_plot_elements(ax=ax3, xlim_upper=max_time)
    ax3.set_ylabel("Length (km)")

    # Lowess length
    ax3.plot(
        times,
        lowess_lengths,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="Lowess Length",
        color="cadetblue",
    )
    ax3.legend()

    # plot azimuths
    ax4 = fig.add_subplot(4, 2, 7)
    if log == True:
        ax4.set_xscale("log")
    ax4.title.set_text("Azimuths")
    ax4.plot(
        times,
        azimuths,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="Azimuths",
        color="orange",
    )
    _additional_plot_elements(ax=ax4, xlim_upper=max_time)
    ax4.legend()
    if log == True:
        ax4.set_xlabel("log(seconds since mainshock)")
    else:
        ax4.set_xlabel("seconds since mainshock")
    ax4.set_ylabel("Degrees ($\degree$)")

    # plot dips
    ax5 = fig.add_subplot(4, 2, 8)
    if log == True:
        ax5.set_xscale("log")
    ax5.title.set_text("Dips")
    ax5.plot(
        times,
        dips,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="Dips",
        color="green",
    )
    _additional_plot_elements(ax=ax5, xlim_upper=max_time)
    ax5.legend()
    if log == True:
        ax5.set_xlabel("log(seconds since mainshock)")
    else:
        ax5.set_xlabel("seconds since mainshock")
    ax5.set_ylabel("Degrees ($\degree$)")

    # plot magnitudes
    ax6 = fig.add_subplot(4, 2, 4)
    if log == True:
        ax6.set_xscale("log")
    ax6.title.set_text("Magnitudes")
    _additional_plot_elements(ax=ax6, xlim_upper=max_time)
    ax6.plot(
        times,
        mags,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="Scaled magnitude",
        color="blue",
    )
    ax6.plot(
        times,
        lowess_mags,
        linestyle="-",
        marker=".",
        markersize="0.01",
        linewidth=3,
        label="Lowess scaled magnitude",
        color="midnightblue",
    )
    ax6.plot(
        times,
        GeoNet_mags,
        linestyle="-",
        marker=".",
        markersize="0.01",
        label="GeoNet preferred magnitude",
        color="darkgray",
    )
    # plot geonet magnitudes and uncertainties
    geonet_max_mags, geonet_min_mags = [], []
    for i, m in enumerate(GeoNet_mags):
        geonet_max_mags.append(m + GeoNet_mags_uncerts[i])
        geonet_min_mags.append(m - GeoNet_mags_uncerts[i])
    ax6.fill_between(times, geonet_min_mags, geonet_max_mags, alpha=0.2, color="gray")
    ax6.set_ylabel("Magnitude")
    ax6.legend()
    fig.tight_layout()
    return fig

#################################


def rotate(x, y, azimuth, clockwise: bool = True):
    """Rotation."""
    from math import sin, cos, radians

    azimuth = radians(azimuth)

    if not clockwise:
        x_out = x * cos(azimuth) - y * sin(azimuth)
        y_out = x * sin(azimuth) + y * cos(azimuth)
    else:
        x_out = x * cos(azimuth) + y * sin(azimuth)
        y_out = y * cos(azimuth) - x * sin(azimuth)
    return (x_out, y_out)


def ellipse_to_rectangle(
    latitude: float,
    longitude: float,
    offset_x: float,
    offset_y: float,
    length: float,
    width: float,
    azimuth: float,  # Azimuth from north of long axis
):
    import matplotlib.pyplot as plt

    # Convert to m
    length *= 1000.0
    width *= 1000.0
    offset_x *= 1000.0
    offset_y *= 1000.0

    # fig, ax = plt.subplots()
    # Work out corner co-ordinates in x', y' space
    top_left = (-1 * (width / 2), length / 2)
    top_right = (width / 2, length / 2)
    bottom_right = (-1 * top_left[0], -1 * top_left[1])
    bottom_left = (-1 * top_right[0], -1 * top_right[1])

    corners = [top_left, top_right, bottom_right, bottom_left, top_left]
    # ax.plot([corner[0] for corner in corners],
    #        [corner[1] for corner in corners],
    #        label="original")

    # Rotate co-ordinate system by azimuth
    top_left = rotate(*top_left, azimuth)
    top_right = rotate(*top_right, azimuth)
    bottom_right = rotate(*bottom_right, azimuth)
    bottom_left = rotate(*bottom_left, azimuth)

    corners = [top_left, top_right, bottom_right, bottom_left, top_left]
    # ax.plot([corner[0] for corner in corners],
    #        [corner[1] for corner in corners],
    #        label=f"Rotated by {azimuth:.2f}")
    # ax.set_aspect("equal")
    # fig.legend()
    # fig.show()
    # Convert mainshock to NZTM
    nztm = CRS.from_epsg(2193)
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84, nztm, always_xy=True)
    origin_nztm = transformer.transform(longitude, latitude)

    top_left = (origin_nztm[0] + top_left[0], origin_nztm[1] + top_left[1])
    top_right = (origin_nztm[0] + top_right[0], origin_nztm[1] + top_right[1])
    bottom_right = (origin_nztm[0] + bottom_right[0], origin_nztm[1] + bottom_right[1])
    bottom_left = (origin_nztm[0] + bottom_left[0], origin_nztm[1] + bottom_left[1])
    # fig2, ax2 = plt.subplots()
    corners = [top_left, top_right, bottom_right, bottom_left, top_left]
    # ax2.plot([corner[0] for corner in corners],
    #        [corner[1] for corner in corners],
    #        label="NZTM")

    # Shift by offset
    top_left = (top_left[0] + offset_x, top_left[1] + offset_y)
    top_right = (top_right[0] + offset_x, top_right[1] + offset_y)
    bottom_right = (bottom_right[0] + offset_x, bottom_right[1] + offset_y)
    bottom_left = (bottom_left[0] + offset_x, bottom_left[1] + offset_y)
    corners = [top_left, top_right, bottom_right, bottom_left, top_left]
    # ax2.plot([corner[0] for corner in corners],
    #        [corner[1] for corner in corners],
    #        label="Shifted")

    # ax2.set_aspect("equal")
    # fig2.legend()
    # fig2.show()

    # Convert to Lat/Lon
    inverse_transformer = Transformer.from_crs(nztm, wgs84, always_xy=True)
    top_left = inverse_transformer.transform(*top_left)
    top_right = inverse_transformer.transform(*top_right)
    bottom_right = inverse_transformer.transform(*bottom_right)
    bottom_left = inverse_transformer.transform(*bottom_left)

    corners = [top_left, top_right, bottom_right, bottom_left, top_left]

    # Sanity plotting
    # map_fig = pygmt.Figure()
    # region=[min(c[0] for c in corners),
    #        max(c[0] for c in corners),
    #        min(c[1] for c in corners),
    #        max(c[1] for c in corners)]
    # map_fig.basemap(region=region, projection="M12c", frame=True)
    # map_fig.coast(shorelines="1/0.5p")
    # map_fig.plot(corners)
    # map_fig.plot(x=longitude, y=latitude, style="c0.3c", fill="red", pen="black")
    # map_fig.show()

    return corners[0:-1]


if __name__ == "__main__":
    print("Nothing to see here.")
