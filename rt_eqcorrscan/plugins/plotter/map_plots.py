"""
Map plots for RTEQC
"""

PYGMT_INSTALLED = True
try:
    import pygmt
except ModuleNotFoundError:
    PYGMT_INSTALLED = False

import numpy as np

from typing import Union, Iterable

from obspy.core.event import Catalog, Event

from rt_eqcorrscan.plugins.plotter.helpers import (
    get_origin_attr, get_magnitude_attr, SparseEvent)


def plot_map(catalog: Union[Catalog, Iterable[Event], Iterable[SparseEvent]]
             ) -> pygmt.Figure:
    """

    Parameters
    ----------
    catalog

    Returns
    -------

    """
    depths = np.array([get_origin_attr(ev, "depth") or np.nan
                       for ev in catalog]) / 1000.0
    latitudes = np.array([get_origin_attr(ev, "latitude") or np.nan
                          for ev in catalog])
    longitudes = np.array([get_origin_attr(ev, "longitude") or np.nan
                           for ev in catalog])
    magnitudes = np.array([get_magnitude_attr(ev, "mag") or np.nan
                           for ev in catalog])
    np.nan_to_num(magnitudes, copy=False, nan=0.0)

    # Mask out nans in lats and lons
    nan_mask = np.logical_or(np.isnan(latitudes), np.isnan(longitudes))
    nan_mask = ~np.logical_or(np.isnan(depths), nan_mask)
    depths = depths[nan_mask]
    latitudes = latitudes[nan_mask]
    longitudes = longitudes[nan_mask]
    magnitudes = magnitudes[nan_mask]

    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='ddd.xx')
    pygmt.makecpt(cmap='plasma', reverse=True, series=[
                  depths.min(), depths.max()])

    lat_range = latitudes.max() - latitudes.min()
    # Cope with possible wrap-around at dateline
    lon_range_grenwich = longitudes.max() - longitudes.min()
    longitudes_dateline = longitudes % 360
    lon_range_dateline = longitudes_dateline.max() - longitudes_dateline.min()
    if lon_range_dateline < lon_range_grenwich:
        lon_range = lon_range_dateline
        longitudes = longitudes_dateline
    else:
        lon_range = lon_range_grenwich

    lat_pad = 0.2 * lat_range
    lon_pad = 0.2 * lon_range
    region = [longitudes.min() - lon_pad,
              longitudes.max() + lon_pad,
              latitudes.min() - lat_pad,
              latitudes.max() + lat_pad]

    if 3 <= lat_range <= 10:
        lat_major_tick = 1
        lat_minor_tick = 0.25
    elif lat_range < 3:
        lat_major_tick = 0.5
        lat_minor_tick = 0.1
    else:
        lat_major_tick = 5
        lat_minor_tick = 2.5

    if 3 <= lon_range <= 10:
        lon_major_tick = 1
        lon_minor_tick = 0.25
    elif lon_range < 3:
        lon_major_tick = 0.5
        lon_minor_tick = 0.1
    else:
        lon_major_tick = 5
        lon_minor_tick = 2.5

    fig.coast(region=region,
              shorelines=True,
              land='grey',
              water='lightblue',
              projection='M10c',
              frame=[
                  'WSne',
                  f'xa{lon_major_tick}f{lon_minor_tick}',
                  f'ya{lat_major_tick}f{lat_minor_tick}'])
    fig.plot(x=longitudes,
             y=latitudes,
             size=0.02 * magnitudes,
             fill=depths,
             cmap=True,
             style='cc', pen='black')
    # We add a label to our colour scale with +l
    fig.colorbar(frame='af+l"Depth (km)""')
    return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
