"""
Plotting for real-time seismic data.

:copyright:
    Calum Chamberlain

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from obspy import UTCDateTime


class PlottingError(Exception):
    pass


class PlotBuffer:
    """Control plotting loop for real-time plotting"""
    def __init__(self, buffer, plot_length=600, doblit=False,
                 plot_end=UTCDateTime.now(), ylimits=(-10, 10), size=(6, 6)):
        """
        :type buffer: obspy.core.stream.Stream
        :param buffer: Buffer of data to plot
        :type plot_length: float
        :param plot_length: Length of time to plot in seconds
        :type doblit: bool
        :param doblit: Whether to use "blitting" or not.
        :type ylimits: tuple
        :param ylimits:
            Multipliers to scale x-axis by based on first data chunk
        :type size: tuple
        :param size: Size in inches of plot (width, height)
        """
        self.data = buffer.copy()
        self.fig, axes = plt.subplots(len(self.data), 1, sharex=True)
        self.fig.set_size_inches(size)
        if len(self.data) == 1:
            axes = [axes]
        self.doblit = doblit
        self.plot_length = plot_length
        self.data.trim(plot_end - self.plot_length, plot_end)
        self.backgrounds = {}
        self.lines = {}
        self.axes = {tr.id: ax for (tr, ax) in zip(self.data, axes)}

        plt.show(False)
        plt.draw()

        # Set up the axes and cache things that need to be cached
        for tr_id, ax in self.axes.items():
            ax.set_xlim(-plot_length, 0)
            try:
                tr = self.data.select(id=tr_id)[0]
            except IndexError:
                raise PlottingError(
                    "Could not find {0} in buffer".format(tr_id))
            x = np.arange(-plot_length, 0, tr.stats.delta)
            offset_samples = int(
                (plot_end - tr.stats.endtime) * tr.stats.sampling_rate) + 1
            y = np.zeros(len(x))
            y[-(offset_samples + len(tr.data)):-offset_samples] = tr.data
            if self.doblit:
                self.backgrounds.update({
                    tr_id: self.fig.canvas.copy_from_bbox(ax.bbox)})
            self.lines.update({tr_id: ax.plot(x, y, 'k', linewidth=1.1)[0]})
            ax.text(0.0, 0.8, tr_id, transform=ax.transAxes,
                    bbox={'facecolor': 'white', 'alpha': 0.85})
            ax.set_ylim(ylimits[0] * (y.max() - y.min()),
                        ylimits[1] * (y.max() - y.min()))
        axes[-1].set_xlabel("Time (s)")
        self.fig.suptitle("Real-time plotting")
        plt.subplots_adjust(hspace=0)

        if doblit:
            # Required at least once to cache the renderer
            self.fig.canvas.draw()

    def update(self, new_data, plot_end=UTCDateTime.now()):
        # update the dataset - use obspy merge
        self.data += new_data.copy()
        # Merge and discard older overlapping data, assuming better data
        # has come in later.
        self.data.merge(method=1, interpolation_samples=0, fill_value=0)
        self.data.trim(plot_end - self.plot_length, plot_end)
        for tr in self.data:
            # x stays the same no mater what, just set the y
            offset_samples = int(
                (plot_end - tr.stats.endtime) * tr.stats.sampling_rate) + 1
            y = np.zeros(int(self.plot_length * tr.stats.sampling_rate))
            startind = offset_samples + len(tr.data)
            if startind > len(y):
                startind = len(y)
            y[-startind:-offset_samples] = \
                tr.data[-(startind - offset_samples):]
            if self.doblit:
                # restore background
                self.fig.canvas.restore_region(self.backgrounds[tr.id])
                # redraw just the line
                self.lines[tr.id].set_ydata(y)
                self.axes[tr.id].draw_artist(self.lines[tr.id])
                # fill in the axes rectangle
                self.fig.canvas.blit(self.axes[tr.id].bbox)
            else:
                # redraw everything
                self.lines[tr.id].set_ydata(y)
                self.fig.canvas.draw()

    def stop(self):
        plt.close(self.fig)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
