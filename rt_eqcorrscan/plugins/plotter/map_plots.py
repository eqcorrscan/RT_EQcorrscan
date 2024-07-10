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
    depths = np.array([get_origin_attr(ev, "depth")
                       for ev in catalog]) / 1000.0
    latitudes = np.array([get_origin_attr(ev, "latitude")
                          for ev in catalog])
    longitudes = np.array([get_origin_attr(ev, "longitude")
                           for ev in catalog])
    magntiudes = np.array([get_magnitude_attr(ev, "mag") for ev in catalog])

    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='ddd.xx')
    pygmt.makecpt(cmap='lajolla', reverse=True, series=[
                  depths.min(), depths.max()])

    fig.coast(region=[173, 179, -42, -36],
              shorelines=True,
              land='grey',
              water='lightblue',
              projection='M10c',
              frame=['WSne', 'xa2f1', 'ya2f1'])
    fig.plot(x=longitudes,
             y=latitudes,
             size=0.02 * (2**magntiudes),
             fill=depths,
             cmap=True,
             style='cc', pen='black')
    # We add a label to our colour scale with +l
    fig.colorbar(frame='af+l"Depth (km)""')
    return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
