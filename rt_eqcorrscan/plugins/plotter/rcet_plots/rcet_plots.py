"""
Code for making maps and other plots for aftershock detection.

:author: Emily Warren-Smith

(lightly) Editted by Calum Chamberlain
"""

import numpy as np
import logging
import statsmodels.api as sm

import os
import csv

from typing import Tuple

import datetime

from matplotlib.font_manager import FontProperties
import pyproj, math
import matplotlib.pyplot as plt

from rt_eqcorrscan.plugins.plotter.rcet_plots.calculations import (
    get_len_theta, get_len_LOWESS, get_cov_ellipse, extract_xy,
    to_xr_yr_mainshock, find_outliers, outliers_simple,
    to_xz_yz_z_centroid)

GEODESIC = pyproj.Geod(ellps="WGS84")

Logger = logging.getLogger(__name__)


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
        + str(datetime.timedelta(seconds=t)).split(".")[0]
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
        + str(datetime.timedelta(seconds=t)).split(".")[0]
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
    elapsed_secs,
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
        t=elapsed_secs,
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
    # Normalize dip - should be between 0 and 90
    if dip > 90:
        dip = 180 - dip

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
        t=elapsed_secs,
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
        ["Time since trigger:", str(elapsed_time).split(".")[0]],
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
    ax4.set_xlabel("Seconds since mainshock")
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
    ax5.set_xlabel("Seconds since mainshock")
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


if __name__ == "__main__":
    print("Nothing to see here.")
