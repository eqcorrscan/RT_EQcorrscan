"""
Event plotting for RT-EQcorrscan.
"""
import numpy as np

from obspy.core.event import Event
from obspy import Stream, Trace, UTCDateTime
from obspy.core.util import AttribDict

from matplotlib.figure import Figure

PHASE_CMAP = {"P": ("red", "P-"), "S": ("blue", "S-")}


def _get_plot_starttime(event: Event, st: Stream) -> UTCDateTime:
    """Get starttime of a plot given an event and a stream."""
    try:
        attribute_with_time = event.preferred_origin() or event.origins[0]
    except (AttributeError, IndexError):
        try:
            attribute_with_time = AttribDict(
                {"time": min([p.time for p in event.picks]) - 5})
        except ValueError:
            attribute_with_time = AttribDict(
                {"time": min([tr.stats.starttime for tr in st])})
    return attribute_with_time.time


def plot_event(
    event: Event,
    st: Stream,
    length: float = 60.,
    passband: tuple = (2, 10),
    size: tuple = (10.5, 10.5),
    show: bool = True,
    fig: Figure = None
) -> Figure:
    """
    Plot the waveforms for an event with pick and calculated arrival times.

    event
        Event to plot
    st
        Obspy Stream for this event
    length
        Length to plot, from origin time
    passband
        Tuple of (lowcut, highcut) for filtering.
    size
        Figure size parsed to matplotlib.
    show
        Whether to show the figure or not.
    fig
        Figure to plot into.

    Returns
        Figure.
    """
    import matplotlib.pyplot as plt

    event.picks.sort(key=lambda p: p.time)
    origin_time = _get_plot_starttime(event, st)
    _st = st.slice(origin_time, origin_time + length).copy()
    _st = _st.split().detrend().filter(
        "bandpass", freqmin=passband[0], freqmax=passband[1]).merge()
    # Trim the event around the origin time
    if fig is None:
        fig, axes = plt.subplots(len(_st), 1, sharex=True, figsize=size)
        if len(_st) == 1:
            axes = [axes]
    else:
        axes = [fig.add_subplot(len(_st), 1, 1)]
        if len(_st) > 1:
            for i in range(len(_st) - 1):
                axes.append(fig.add_subplot(len(_st), 1, i + 2, sharex=axes[0]))
    lines, labels = ([], [])
    min_x = []
    max_x = []
    for ax, tr in zip(axes, _st):
        picks, arrivals = ([], [])
        for pick in event.picks:
            if pick.waveform_id.station_code == tr.stats.station:
                picks.append(pick)
        try:
            origin = event.preferred_origin() or event.origins[0]
            for arrival in origin.arrivals:
                referenced_pick = arrival.pick_id.get_referred_object()
                if referenced_pick.waveform_id.station_code == tr.stats.station:
                    arrivals.append(arrival)
        except IndexError: # pragma: no cover
            pass
        lines, labels, chan_min_x, chan_max_x = _plot_channel(
            ax=ax, tr=tr, picks=picks, arrivals=arrivals, lines=lines,
            labels=labels)
        min_x.append(chan_min_x)
        max_x.append(chan_max_x)
    # Make xlabels invisible for all axes apart from the last one
    for ax in axes[0:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlim([np.min(min_x), np.max(max_x)])
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    fig.legend(lines, labels)
    if show: # pragma: no cover
        fig.show()
    del _st
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
    y = tr.data
    x = np.arange(tr.stats.npts, dtype=np.float32)
    x /= tr.stats.sampling_rate
    x = np.array([(tr.stats.starttime + _x).datetime for _x in x])
    min_x, max_x = (x[0], x[-1])
    ax.plot(x, y, 'k', linewidth=1.2)
    for pick in picks:
        pick_hint = pick.phase_hint or "none"
        pcolor, label = PHASE_CMAP.get(pick_hint.upper(),
                                       ("k", "Unknown pick"))
        if label.endswith('-'):
            label += "pick"
        line = ax.axvline(x=pick.time.datetime, color=pcolor, linewidth=2,
                          linestyle='--', label=label)
        if label not in labels:
            lines.append(line)
            labels.append(label)
        max_x = max(pick.time.datetime, max_x)
        min_x = min(pick.time.datetime, min_x)
    for arrival in arrivals:
        arrival_hint = arrival.phase or "none"
        pcolor, label = PHASE_CMAP.get(arrival_hint.upper(),
                                       ("k", "Unknown arrival"))
        if label.endswith('-'):
            label += "arrival"
        arrival_time = (
            arrival.pick_id.get_referred_object().time + arrival.time_residual)
        line = ax.axvline(x=arrival_time.datetime, color=pcolor, linewidth=2,
                          linestyle='-', label=label)
        if label not in labels:
            lines.append(line)
            labels.append(label)
        max_x = max(arrival_time.datetime, max_x)
        min_x = min(arrival_time.datetime, min_x)
    ax.set_ylabel(tr.id, rotation=0, horizontalalignment="right")
    ax.yaxis.set_ticks([])
    return lines, labels, min_x, max_x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
