"""
Code for calcualtions required for rcet plots.

:author: Emily Warren-Smith

Restructured and (lightly) editted by Calum Chamberlain
"""

import numpy as np
import logging
import statsmodels.api as sm

from pyproj import CRS, Transformer

from obspy.core.event import Catalog
import pyproj, math
from matplotlib.patches import Ellipse


GEODESIC = pyproj.Geod(ellps="WGS84")

Logger = logging.getLogger(__name__)

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
            (RT_mainshock.preferred_origin() or RT_mainshock.origins[-1]).depth_errors.uncertainty / 1000,
            1,
        )
    except (IndexError, TypeError):
        RT_mainshock_depth_uncertainty = 0.0

    return (
        geonet_mainshock_mag,
        geonet_mainshock_mag_uncertainty,
        geonet_mainshock_depth,
        geonet_mainshock_depth_uncertainty,
        RT_mainshock_depth,
        RT_mainshock_depth_uncertainty,
    )


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


def to_xr_yr_mainshock(catalog, mainshock, rotation):
    """
    Convert catalog origins into new coordinate system rotated around mainshock

    :param rotation: Degrees clockwise from north to rotate system.
    """
    EARTHRADIUS = 6371
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
    EARTHRADIUS = 6371
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
    pass