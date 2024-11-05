"""
Tools for extracting 1D models from 3D models
"""

import numpy as np
import pandas as pd

from functools import partial

from math import radians, cos, sin

from typing import Union, List

from pyproj import Transformer, CRS

MODEL_FILE = "vlnzw2p2dnxyzltln.tbl.txt"  # NZ3D 2.2 from https://zenodo.org/record/3779523#.YCRFaOrRVhF


def _round_down(num: Union[float, int], nearest: int = 10):
    return (num // nearest) * nearest

def _round_up(num: Union[float, int], nearest: int = 10):
    return (1 + (num // nearest)) * nearest


class RotatedTransformer:
    def __init__(self, transformer: Transformer, rotation: float):
        """
        Transform then rotate by rotation of axes counter-clockwise.

        See: https://en.wikipedia.org/wiki/Rotation_of_axes_in_two_dimensions

        """
        for axis in transformer.target_crs.axis_info:
            if not axis.unit_name == "metre":
                raise NotImplementedError(
                    "Output transform must output metres")
        self.transformer = transformer
        self.rotation = rotation

    @property
    def radrod(self):
        return radians(self.rotation)

    def transform(self, c1: float, c2: float):
        x1, y1 = self.transformer.transform(c1, c2)
        x2 = x1 * cos(self.radrod) + y1 * sin(self.radrod)
        y2 = -1 * x1 * sin(self.radrod) + y1 * cos(self.radrod)
        return x2, y2

def _project_latlon(
        latitude: float,
        longitude: float,
        projection: str,
        ellipsoid: str,
        origin_lat: float,
        origin_lon: float,
        origin_rotation: float,
) -> [float, float]:
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_proj4(
            f"+proj={projection} +lat_0={origin_lat} +lon_0={origin_lon} "
            f"+ellps={ellipsoid}"))
    if origin_rotation != 0:
        transformer = RotatedTransformer(
            transformer=transformer, rotation=origin_rotation)
    return transformer.transform(latitude, longitude)



def extract_one_1_lat_lon_simulps(
        model_file: str,
        lats: Union[List[float], np.ndarray],
        lons: Union[List[float], np.ndarray],
        projection: str,
        ellipsoid: str,
        origin_lat: float,
        origin_lon: float,
        origin_rotation: float,
) -> pd.DataFrame:
    """
    Extract a one-d spatial average velocity model from a simulPS
    formatted grid

    Parameters
    ----------
    model_file
    min_lat
    max_lat
    min_lon
    max_lon

    Returns
    -------

    """
    proj_kwargs = {
        'projection': projection,
        'ellipsoid': ellipsoid,
        'origin_lat': origin_lat,
        'origin_lon': origin_lon,
        'origin_rotation': origin_rotation}
    _project = partial(_project_latlon, **proj_kwargs)
    x, y = zip(*[_project(latitude=lat, longitude=lon)
                 for lat, lon in zip(lats, lons)])
    min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
    # Convert to km used by simulps
    min_x, max_x, min_y, max_y = (
        min_x / 1000., max_x / 1000., min_y / 1000., max_y / 1000.)
    # SimulPS uses a RH co-ord system, we need to flip the xs
    min_x, max_x = -1 * max_x, -1 * min_x
    return extract_one_d_simulps(
        model_file=model_file, min_x=min_x, min_y=min_y, max_x=max_x,
        max_y=max_y)


def extract_one_d_simulps(
        model_file: str,
        min_x: float = 72.0,
        max_x: float = 110.0,
        min_y: float = -100.0,
        max_y: float = 80.0,
) -> pd.DataFrame:
    """
    Extract a one-d spatial average velocity model from a simulps grid.

    Parameters
    ----------
    min_x:
        Minimum X value in NZ3D co-ordinate system
    max_x:
        Maximum X value in NZ3D co-ordinate system
    min_y:
        Minimim Y value in NZ3D co-ordinate system
    max_y:
        Maximum Y value in NZ3D co-ordinate system

    """
    v_model = pd.read_csv(model_file, header=1, delim_whitespace=True)

    x_mask = np.logical_and(
        v_model["x(km)"] <= max_x, v_model["x(km)"] >= min_x)
    y_mask = np.logical_and(
        v_model["y(km)"] <= max_y, v_model["y(km)"] >= min_y)

    mask = np.logical_and(x_mask, y_mask)

    region = v_model[mask]
    # Make a quick plot showing the region
    bl = region[np.logical_and(region["x(km)"] == region["x(km)"].min(),
                               region["y(km)"] == region["y(km)"].min())]
    br = region[np.logical_and(region["x(km)"] == region["x(km)"].min(),
                               region["y(km)"] == region["y(km)"].max())]
    tl = region[np.logical_and(region["x(km)"] == region["x(km)"].max(),
                               region["y(km)"] == region["y(km)"].min())]
    tr = region[np.logical_and(region["x(km)"] == region["x(km)"].max(),
                               region["y(km)"] == region["y(km)"].max())]
    #bl = (bl.Latitude.to_list()[0], bl.Longitude.to_list()[0])
    #br = (br.Latitude.to_list()[0], br.Longitude.to_list()[0])
    #tl = (tl.Latitude.to_list()[0], tl.Longitude.to_list()[0])
    #tr = (tr.Latitude.to_list()[0], tr.Longitude.to_list()[0])
    #plot_region(corners=[bl, tl, tr, br])

    depths = sorted(list(set(region["Depth(km_BSL)"])))

    # Get average vp and vs for each depth
    vp, vs = [], []
    for depth in depths:
        vp.append((region[region["Depth(km_BSL)"] == depth]).Vp.mean())
        vs.append((region[region["Depth(km_BSL)"] == depth]).Vs.mean())
    out = pd.DataFrame(data={"Depth": depths, "vp": vp, "vs": vs})
    return out


def plot_region(corners):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig = plt.figure()
    ax = fig.add_subplot(projection=ccrs.PlateCarree())

    lats, lons = zip(*corners)
    lats, lons = list(lats), list(lons)
    ax.set_extent((min(lons) - 1, max(lons) + 1, min(lats) - 1, max(lats) + 1),
                  crs=ccrs.PlateCarree())
    ax.coastlines()
    lons.append(lons[0])
    lats.append(lats[0])
    ax.plot(lons, lats, transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)

    plt.show()
