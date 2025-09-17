"""
Console entry point for pygmt plots - need to be run in
seperate python kernel to properly free resources
"""

import logging
import pygmt
import json
import pandas as pd
import numpy as np

from obspy.geodetics import kilometer2degrees
from obspy import UTCDateTime, read_events

from rt_eqcorrscan.config.config import _setup_logging
from rt_eqcorrscan.helpers.sparse_event import get_origin_attr
from rt_eqcorrscan.plugins.plotter.rcet_plots.pygmt_plotting import (
    _eq_map_summary, _eq_map)


Logger = logging.getLogger("plotter-pygmt")


def aftershock_map(
    catalog_csv: str,
    inventory_csv: str,
    mainshock_qml: str | None,
    relocated_mainshock_qml: str | None,
    search_radius: float,
    topo_res: str = "03s",
    topo_cmap: str = "grayC",
    hillshade: bool = False,
    pad: float = 10.0,
    timestamp: UTCDateTime = UTCDateTime.now(),
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> pygmt.Figure:
    """

    Parameters
    ----------
    catalog_csv
    inventory_csv
    mainshock_qml
    relocated_mainshock_qml
    search_radius
    topo_res
    topo_cmap
    hillshade
    pad
    timestamp

    Returns
    -------

    """
    cat_df = pd.read_csv(catalog_csv)
    inv_df = pd.read_csv(inventory_csv)
    mainshock, relocated_mainshock = None, None
    if mainshock_qml:
        mainshock = read_events(mainshock_qml)[0]
    if relocated_mainshock_qml:
        relocated_mainshock = read_events(relocated_mainshock_qml)[0]

    mags = np.nan_to_num(cat_df.Magnitude.to_numpy(), nan=3.0)
    mainshock_origin = mainshock.preferred_origin() or mainshock.origins[-1]

    fig = _eq_map(
        lats=cat_df.Latitude.to_numpy(),
        lons=cat_df.Longitude.to_numpy(),
        depths=cat_df["Depth (km)"].to_numpy(),
        mags=mags,
        middle_lon=mainshock_origin.longitude,
        middle_lat=mainshock_origin.latitude,
        search_radius_deg=kilometer2degrees(search_radius),
        station_lons=inv_df.Longitude.to_numpy(),
        station_lats=inv_df.Latitude.to_numpy(),
        width=15.0,
        pad=pad,
        inset_multiplier=8,
        topo_res=topo_res,
        topo_cmap=topo_cmap,
        hillshade=hillshade,
        timestamp=timestamp,
        min_depth=min_depth,
        max_depth=max_depth
    )

    # Plot mainshock
    fig.plot(
        x=mainshock_origin.longitude,
        y=mainshock_origin.latitude,
        style=f"a0.7c",
        fill="gold",
        pen="black",
    )

    if relocated_mainshock is not None:
        fig.plot(
            x=get_origin_attr(relocated_mainshock, "longitude"),
            y=get_origin_attr(relocated_mainshock, "latitude"),
            style=f"a0.7c",
            fill="yellow",
            pen="black"
        )

    return fig


def summary_map(
    catalog_csv: str,
    template_csv: str,
    outlier_csv: str,
    inventory_csv: str,
    corners_json: str,
    mainshock_qml: str | None,
    relocated_mainshock_qml: str | None,
    search_radius: float,
    topo_res: str = "03s",
    topo_cmap: str = "grayC",
    hillshade: bool = False,
) -> pygmt.Figure:

    cat_df = pd.read_csv(catalog_csv)
    template_df = pd.read_csv(template_csv)
    outlier_df = pd.read_csv(outlier_csv)

    inv_df = pd.read_csv(inventory_csv)
    mainshock, relocated_mainshock = None, None
    if mainshock_qml:
        mainshock = read_events(mainshock_qml)[0]
    if relocated_mainshock_qml:
        relocated_mainshock = read_events(relocated_mainshock_qml)[0]

    mags = np.nan_to_num(cat_df.Magnitude.to_numpy(), nan=2.0)
    ref_mags = np.nan_to_num(template_df.Magnitude.to_numpy(), nan=2.0)
    outlier_mags = np.nan_to_num(outlier_df.Magnitude.to_numpy(), nan=2.0)

    with open(corners_json, "r") as f:
        corners = json.load(f)

    fig = _eq_map_summary(
        lats=cat_df.Latitude.to_numpy(),
        lons=cat_df.Longitude.to_numpy(),
        depths=cat_df['Depth (km)'].to_numpy(),
        mags=mags,
        ref_mags=ref_mags,
        lats_o=outlier_df.Latitude.to_numpy(),
        lons_o=outlier_df.Longitude.to_numpy(),
        depths_o=outlier_df["Depth (km)"].to_numpy(),
        mags_o=outlier_mags,
        station_lons=inv_df.Longitude.to_numpy(),
        station_lats=inv_df.Latitude.to_numpy(),
        width=20.0,
        inset_multiplier=8.,
        corners=corners,
        topo_res=topo_res,
        topo_cmap=topo_cmap,
        hillshade=hillshade,
        colours='depth',
        magcut=4.0,
        mainshock=mainshock,
        RT_mainshock=relocated_mainshock,
        search_radius_deg=kilometer2degrees(search_radius),
    )

    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PyGMT plotting")

    # Args for actually plotting
    parser.add_argument(
        "--eps-dpi", type=float, default=300.0,
        help="DPI for eps/pdf outputs")
    parser.add_argument(
        "--png-dpi", type=float, default=300.0,
        help="DPI for png outputs")
    parser.add_argument(
        "--out-dir", type=str, default=".",
        help="Directory to save plots to")
    parser.add_argument(
        "--template-csv", type=str, required=True,
        help="CSV file of template catalog")
    parser.add_argument(
        "--catalog-csv", type=str, required=True,
        help="CSV file of detected catalog")
    parser.add_argument(
        "--outlier-csv", type=str, required=True,
        help="CSV file of outlier catalog")
    parser.add_argument(
        "--inventory-csv", type=str, required=True,
        help="CSV file of station locations")
    parser.add_argument(
        "--mainshock-qml", type=str, default=None,
        help="QuakeML file of mainshock")
    parser.add_argument(
        "--relocated-mainshock-qml", type=str, default=None,
        help="QuakeML file of relocated mainshock")
    parser.add_argument(
        "--search-radius", type=float, required=True,
        help="Search radius in km")
    parser.add_argument(
        "--timestamp", type=UTCDateTime, default=UTCDateTime.now(),
        help="Timestamp for labelling plots")
    parser.add_argument(
        "--corners-json", type=str, required=True,
        help="Json file containining fault corners to plot")


    # Args for logging
    parser.add_argument(
        "--log-level", "-l", type=str, default="INFO")
    parser.add_argument(
        "--log-formatter", type=str,
        default="%(asctime)s\t[%(processName)s:%(threadName)s]: " \
                "%(name)s\t%(levelname)s\t%(message)s")
    parser.add_argument(
        "--log-file", type=str, default="pygmt-plotter.log")
    parser.add_argument(
        "--log-to-screen", "-s", action="store_true")

    args = parser.parse_args()

    _setup_logging(
        log_level=args.log_level, log_formatter=args.log_formatter,
        screen=args.log_to_screen, file=True, filename=args.log_file)

    # Run plots and save output

    Logger.info("Making template map")
    template_map = aftershock_map(
        catalog_csv=args.template_csv,
        inventory_csv=args.inventory_csv,
        mainshock_qml=args.mainshock_qml,
        relocated_mainshock_qml=None,
        search_radius=args.search_radius,
        timestamp=args.timestamp)
    template_map.savefig(
        f"{args.out_dir}/catalog_templates_latest.png", dpi=args.png_dpi)
    template_map.savefig(
        f"{args.out_dir}/catalog_templates_latest.pdf", dpi=args.eps_dpi)

    Logger.info("Making real-time map")
    detection_map = aftershock_map(
        catalog_csv=args.catalog_csv,
        inventory_csv=args.inventory_csv,
        mainshock_qml=args.mainshock_qml,
        relocated_mainshock_qml=args.relocated_mainshock_qml,
        search_radius=args.search_radius,
        timestamp=args.timestamp)
    detection_map.savefig(
        f"{args.out_dir}/catalog_RT_latest.png", dpi=args.png_dpi)
    detection_map.savefig(
        f"{args.out_dir}/catalog_RT_latest.pdf", dpi=args.eps_dpi)

    Logger.info("Making summary map")
    output_summary_map = summary_map(
        catalog_csv=args.catalog_csv,
        template_csv=args.template_csv,
        outlier_csv=args.outlier_csv,
        inventory_csv=args.inventory_csv,
        corners_json=args.corners_json,
        mainshock_qml=args.mainshock_qml,
        relocated_mainshock_qml=args.relocated_mainshock_qml,
        search_radius=args.search_radius,
        topo_cmap="terra")
    output_summary_map.savefig(
        f"{args.out_dir}/Aftershock_extent_depth_map_latest.png", dpi=args.png_dpi)
    output_summary_map.savefig(
        f"{args.out_dir}/Aftershock_extent_depth_map_latest.pdf", dpi=args.eps_dpi)

    return


if __name__ == "__main__":
    main()