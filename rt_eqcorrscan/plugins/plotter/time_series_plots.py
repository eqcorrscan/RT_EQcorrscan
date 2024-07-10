"""
Detection time series plots for RT-EQcorrscan

"""

from typing import Union, Iterable

import matplotlib.pyplot as plt
import numpy as np

from obspy.core.event import Catalog, Event

from rt_eqcorrscan.plugins.plotter.helpers import get_origin_attr, SparseEvent


def inter_event_plot(
    catalog: Union[Catalog, Iterable[Event], Iterable[SparseEvent]],
    fig: plt.Figure = None
) -> plt.Figure:
    """

    Parameters
    ----------
    catalog
    fig

    Returns
    -------

    """
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    times = np.array(
        [np.datetime64(get_origin_attr(ev, "time")) for ev in catalog
         if get_origin_attr(ev, "time")])
    times.sort()
    inter_times = (times[1:] - times[0:-1]) / np.timedelta64(1, 's')

    ax.scatter(times[1:], inter_times, s=0.1, color="k")
    ax.set_yscale("log")
    ax.set_ylabel("Inter event time (s)")
    ax.set_xlabel("Origin time (UTC)")

    return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
