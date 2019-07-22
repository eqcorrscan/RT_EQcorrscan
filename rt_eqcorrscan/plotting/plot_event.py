"""
Event plotting for RT-EQcorrscan.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import numpy as np

from obspy.core.event import Event
from obspy import Stream, Trace

from matplotlib.figure import Figure


def plot_event(
    event: Event,
    st: Stream,
    length: float = 60.,
    size: tuple = (10.5, 10.5),
    show: bool = True,
) -> Figure:
    """
    Plot the waveforms for an event with pick and calculated arrival times.

    event
        Event to plot
    st
        Obspy Stream for this event
    length
        Length to plot, from origin time
    size
        Figure size parsed to matplotlib.
    show
        Whether to show the figure or not.

    Returns
        Figure.
    """
    import matplotlib.pyplot as plt

    try:
        origin_time = event.preferred_origin().time or event.origins[0].time
    except AttributeError:
        # If there isn't an origin time, use the start of the stream
        origin_time = st[0].stats.starttime
    st = st.slice(origin_time, origin_time + length)
    # Trim the event around the origin time
    fig, axes = plt.subplots(len(st), 1, sharex=True, figsize=size)
    if len(st) == 1:
        axes = [axes]
    lines, labels = ([], [])
    min_x = []
    max_x = []
    for ax, tr in zip(axes, st):
        picks, arrivals = ([], [])
        for pick in event.picks:
            if pick.waveform_id.station_code == tr.stats.station:
                picks.append(pick)
        try:
            origin = event.preferred_origin() or event.origins[0]
            for arrival in origin.arrivals:
                if arrival.pick_id.get_referred_object(
                        ).waveform_id.station_code == tr.stats.station:
                    arrivals.append(arrival)
        except IndexError:
            pass
        lines, labels, chan_min_x, chan_max_x = _plot_channel(
            ax=ax, tr=tr, picks=picks, arrivals=arrivals, lines=lines,
            labels=labels)
        min_x.append(chan_min_x)
        max_x.append(chan_max_x)
    axes[-1].set_xlim([np.min(min_x), np.max(max_x)])
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    fig.legend(lines, labels)
    if show:
        fig.show()
    return fig


def _plot_channel(
    ax,
    tr: Trace,
    picks: list = None,
    arrivals: list = None,
    lines: list = None,
    labels: list = None
):
    """
    Plot a single channel into an axis object.
    """
    picks = picks or []
    arrivals = arrivals or []
    lines = lines or []
    labels = labels or []
    x = np.arange(0, tr.stats.endtime - tr.stats.starttime + tr.stats.delta,
                  tr.stats.delta)
    y = tr.data
    if len(x) > len(y):
        x = x[0:len(y)]
    elif len(x) < len(y):
        last_x = x[-1]
        for i in range(len(y) - len(x)):
            x.append(last_x + (tr.stats.delta * i))
    x = np.array([(tr.stats.starttime + _x).datetime for _x in x])
    min_x, max_x = (x[0], x[-1])
    ax.plot(x, y, 'k', linewidth=1.2)
    for pick in picks:
        if not pick.phase_hint:
            pcolor = 'k'
            label = 'Unknown pick'
        elif 'P' in pick.phase_hint.upper():
            pcolor = 'red'
            label = 'P-pick'
        elif 'S' in pick.phase_hint.upper():
            pcolor = 'blue'
            label = 'S-pick'
        else:
            pcolor = 'k'
            label = 'Unknown pick'
        line = ax.axvline(x=pick.time.datetime, color=pcolor, linewidth=2,
                          linestyle='--', label=label)
        if label not in labels:
            lines.append(line)
            labels.append(label)
        if pick.time.datetime > max_x:
            max_x = pick.time.datetime
        elif pick.time.datetime < min_x:
            min_x = pick.time.datetime
    for arrival in arrivals:
        if not arrival.phase:
            pcolor = 'k'
            label = 'Unknown arrival'
        elif 'P' in arrival.phase.upper():
            pcolor = 'red'
            label = 'P-arrival'
        elif 'S' in arrival.phase.upper():
            pcolor = 'blue'
            label = 'S-arrival'
        else:
            pcolor = 'k'
            label = 'Unknown arrival'
        arrival_time = (
            arrival.pick_id.get_referred_object().time + arrival.time_residual)
        line = ax.axvline(x=arrival_time.datetime, color=pcolor, linewidth=2,
                          linestyle='-', label=label)
        if label not in labels:
            lines.append(line)
            labels.append(label)
        if arrival_time.datetime > max_x:
            max_x = arrival_time.datetime
        elif arrival_time.datetime < min_x:
            min_x = arrival_time.datetime
    ax.set_ylabel(tr.id, rotation=0, horizontalalignment="right")
    ax.yaxis.set_ticks([])
    return lines, labels, min_x, max_x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
