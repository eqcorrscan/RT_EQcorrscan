"""
Code for making pygmt maps for aftershock detection.

:author: Emily Warren-Smith

(lightly) Editted by Calum Chamberlain
"""

import pygmt
import numpy as np
import logging
import os
import datetime
import pyproj

from typing import Union

from obspy import UTCDateTime
from obspy.geodetics import kilometer2degrees
from obspy.core.event import Catalog, Event
from obspy.core.inventory import Inventory

from rt_eqcorrscan.helpers.sparse_event import get_origin_attr, \
    get_magnitude_attr

GEODESIC = pyproj.Geod(ellps="WGS84")

Logger = logging.getLogger(__name__)

def _eq_map(
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    mags: np.ndarray,
    middle_lon: float,
    middle_lat: float,
    search_radius_deg: float,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    pad: float,
    width: float,
    inset_multiplier: float,
    topo_res: Union[bool, str],
    topo_cmap: str,
    hillshade: bool,
    timestamp: Union[UTCDateTime, datetime.datetime],
    min_depth: float | None = None,
    max_depth: float | None = None
) -> pygmt.Figure:
    """ """
    if station_lons.max() - station_lons.min() > 180:
        station_lons %= 360

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

    try:
        grid = pygmt.datasets.load_earth_relief(resolution=topo_res, region=region)
    except Exception as e:
        # This seems to occasionally break - relies on a remote server that isn't perfectly stable?
        Logger.exception(f"Could not load topography due to {e}")
        grid = None

    if hillshade and grid is not None:
        dgrid = pygmt.grdgradient(grid=grid, radiance=[0, 80], normalize=True)
        # pygmt.makecpt(cmap=topo_cmap, series=[-1.5, 0.3, 0.01])
        fig.grdimage(grid=grid, shading=dgrid, cmap=topo_cmap)
    elif grid is not None:
        fig.grdimage(grid=grid, cmap=topo_cmap)
    fig.coast(shorelines="1/0.5p", water="white")

    if min_depth is None:
        min_depth = depths.min()
    if max_depth is None:
        max_depth = depths.max()
    depth_range = [min_depth, max_depth]
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

    return fig


def _eq_map_summary(
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    mags: np.ndarray,
    ref_mags: np.ndarray,
    lats_o: np.ndarray,
    lons_o: np.ndarray,
    depths_o: np.ndarray,
    mags_o: np.ndarray,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    magcut,
    width: float,
    inset_multiplier: float,
    topo_res: float,
    topo_cmap: str,
    hillshade: bool,
    corners: list,
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
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logos/EQcorrscan_logo.png"),
            position="g10/0+w3c+jCM",
            box=False,
        )
        fig.image(
            imagefile=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logos/VUW Logo.png"),
            position="g10/1.5+w3c+jCM",
            box=False,
        )
        fig.image(
            imagefile=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logos/Earth_sci_NZ.jpeg"),
            position="g10/3.8+w2c+jCM",
            box=False,
        )
        fig.image(
            imagefile=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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

        try:
            grid = pygmt.datasets.load_earth_relief(resolution=topo_res, region=map_region)
        except Exception as e:
            # Dodgy connection?
            Logger.exception(f"Could not load topography due to {e}")
            grid = None
        if hillshade and grid is not None:
            dgrid = pygmt.grdgradient(grid=grid, radiance=[0, 80], normalize=True)
            # pygmt.makecpt(cmap=topo_cmap, series=[-1.5, 0.3, 0.01])
            fig.grdimage(grid=grid, shading=dgrid, cmap=topo_cmap, panel=[0, 0])
        elif grid is not None:
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
                    projection=inset_proj,
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
    colours: str,
    magcut: float = 4.0,
    inventory: Inventory = None,
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

    mags_o = np.array([get_magnitude_attr(ev, "mag") or 2.0 for ev in outlier_catalog])

    if inventory:
        station_lats = np.array([sta.latitude for net in inventory for sta in net])
        station_lons = np.array([sta.longitude for net in inventory for sta in net])
    else:
        station_lats, station_lons = np.array([]), np.array([])

    fig = _eq_map_summary(
        lats=lats,
        lons=lons,
        depths=depths,
        mags=mags,
        ref_mags=ref_mags,
        lats_o=lats_o,
        lons_o=lons_o,
        depths_o=depths_o,
        mags_o=mags_o,
        station_lons=station_lons,
        station_lats=station_lats,
        width=width,
        inset_multiplier=inset_multiplier,
        corners=corners,
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


if __name__ == "__main__":
    pass
