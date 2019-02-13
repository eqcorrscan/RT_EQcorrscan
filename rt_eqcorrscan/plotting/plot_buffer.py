"""
Plotting for real-time seismic data.

:copyright:
    Calum Chamberlain

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
import logging
import threading
import datetime as dt

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.layouts import gridplot, column
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

from functools import partial

Logger = logging.getLogger(__name__)


class EQcorrscanPlot:
    """
    Streaming bokeh plotting of waveforms.

    :type stream: :class: `obspy.core.stream.Stream`
    :param stream: List of strings of seed_ids for traces to be plotted.
    :type plot_length: float
    :param plot_length: Plot length in seconds
    :type template_catalog: :class:`obspy.core.event.Catalog`
    :param template_catalog:
        Catalog of template events - these will be plotted in map-view and
        will light up when they make a detection
    :type inventory: :class:`obspy.core.station.Inventory`
    :param inventory: Inventory of stations used - will be plotted on the map.
    :type update_interval: int
    :param update_interval: Update frequency of plot in ms
    """
    def __init__(self, rt_client, plot_length, template_catalog, inventory, 
                 update_interval=100, plot_height=800, plot_width=1200):
        channels = [tr.id for tr in rt_client.buffer]
        self.channels = sorted(channels)
        self.plot_length = plot_length
        self.template_catalog = template_catalog
        self.inventory = inventory

        # TODO: Hovertool is not outputting useful things
        self.hover = HoverTool(
            tooltips=[("index", "$index"), ("(x,y)", "(@x, $y)")])
        self.tools = "pan,box_zoom,reset"
        self.plot_options = {
            "plot_width": int(2 * (plot_width / 3)),
            "plot_height": int(plot_height / len(channels)),
            "tools": [self.hover, self.tools], "x_axis_type": "datetime"}
        self.map_options = {
            "plot_width": int(plot_width / 3), "plot_height": plot_height,
            "tools": [self.hover, self.tools]}
        self.updateValue = True
        Logger.info("Initializing plotter")
        make_doc = partial(
            define_plot, rt_client=rt_client, channels=channels,
            catalog=self.template_catalog, inventory=self.inventory,
            map_options=self.map_options, plot_options=self.plot_options,
            plot_length=self.plot_length, update_interval=update_interval)

        apps = {'/RT_EQcorrscan': Application(FunctionHandler(make_doc))}

        self.server = Server(apps)
        self.server.start()
        Logger.info("Plotting started")
        self.threads = []
    
    def background_run(self):
        plotting_thread = threading.Thread(
            target=self._bg_run, name="PlottingThread")
        plotting_thread.daemon = True
        plotting_thread.start()
        self.threads.append(plotting_thread)
        Logger.info("Started plotting")

    def _bg_run(self):
        print('Opening Bokeh application on http://localhost:5006/')
        self.server.io_loop.add_callback(self.server.show, "/")
        self.server.io_loop.start()

    def background_stop(self):
        self.server.io_loop.stop()
        for thread in self.threads:
            thread.join()


def define_plot(doc, rt_client, channels, catalog, inventory, map_options,
                plot_options, plot_length, update_interval):
    """ Set up the plot. """
    # Set up the data source
    stream = rt_client.get_stream()
    template_lats, template_lons, template_alphas = ([], [], [])
    for template in catalog:
        try:
            origin = template.preferred_origin() or template.origins[0]
        except IndexError:
            continue
        template_lats.append(origin.latitude)
        template_lons.append(origin.longitude)
        template_alphas.append(0)

    station_lats, station_lons = ([], [])
    for network in inventory:
        for station in network:
            station_lats.append(station.latitude)
            station_lons.append(station.longitude)

    map_source = ColumnDataSource({
        'template_lats': template_lats,
        'template_lons': template_lons,
        'template_alphas': template_alphas,
        'station_lats': station_lats,
        'station_lons': station_lons})

    trace_sources = {}
    # Allocate empty arrays
    for channel in channels:
        tr = stream.select(id=channel)[0]
        times = np.arange(
            # (tr.stats.endtime - plot_length).datetime,
            tr.stats.starttime.datetime,
            (tr.stats.endtime + tr.stats.delta).datetime,
            step=dt.timedelta(seconds=tr.stats.delta))
        # data = np.zeros(len(times))
        # data[-tr.stats.npts:] = tr.data
        data = tr.data
        trace_sources.update(
            {channel: ColumnDataSource({'time': times, 'data': data})})

    # Set up the map to go on the left side
    # TODO: Set up map plotting - use googlemaps API?
    map_plot = figure(title="Template map", **map_options)
    map_plot.circle(
        x="template_lons", y="template_lats", source=map_source,
        color="firebrick", fill_alpha="template_alphas")
    map_plot.triangle(
        x="station_lons", y="station_lats", source=map_source, color="blue",
        alpha=1.0)

    # Set up the trace plots
    trace_plots = []
    now = dt.datetime.now()
    p1 = figure(
        y_axis_location="right",
        x_range=[now - dt.timedelta(seconds=plot_length), now],
        **plot_options)
    p1.yaxis.axis_label = None
    p1.xaxis.axis_label = None
    p1.min_border_bottom = 0
    if len(channels) != 1:
        p1.xaxis.major_label_text_font_size = '0pt'
    p1_line = p1.line(
        x="time", y='data', source=trace_sources[channels[0]], color="black",
        line_width=2)
    legend = Legend(items=[(channels[0], [p1_line])])
    p1.add_layout(legend, 'left')
 
    datetick_formatter = DatetimeTickFormatter(
        days=["%m/%d"], months=["%m/%d"],
        hours=["%m/%d %H:%M:%S"], minutes=["%m/%d %H:%M:%S"], 
        seconds=["%m/%d %H:%M:%S"], hourmin=["%m/%d %H:%M:%S"],
        minsec=["%m/%d %H:%M:%S"])
    p1.xaxis.formatter = datetick_formatter
    trace_plots.append(p1)

    if len(channels) > 1:
        for i, channel in enumerate(channels[1:]):
            p = figure(
                x_range=p1.x_range,
                y_axis_location="right", **plot_options)
            p.yaxis.axis_label = None
            p.xaxis.axis_label = None
            p.min_border_bottom = 0
            p.min_border_top = 0
            p_line = p.line(
                x="time", y='data', source=trace_sources[channel],
                color="black", line_width=2)
            legend = Legend(items=[(channel, [p_line])])
            p.add_layout(legend, 'left')
            p.xaxis.formatter = datetick_formatter
            trace_plots.append(p)
            if i != len(channels) - 2:
                p.xaxis.major_label_text_font_size = '0pt'
    plots = gridplot([[map_plot, column(trace_plots)]])

    previous_timestamps = {
        channel: stream.select(id=channel)[0].stats.endtime
        for channel in channels}
    
    def update():
        Logger.debug("Plot updating")
        stream = rt_client.get_stream()
        for channel in channels:
            try:
                tr = stream.select(id=channel)[0]
            except IndexError:
                Logger.debug("No channel for {0}".format(channel))
                continue
            new_samples = int(tr.stats.sampling_rate * (
                    previous_timestamps[channel] - tr.stats.endtime))
            if new_samples == 0:
                Logger.debug("No new data for {0}".format(channel))
                continue
            new_times = np.arange(
                previous_timestamps[channel],
                (tr.stats.endtime + tr.stats.delta).datetime,
                step=dt.timedelta(seconds=tr.stats.delta))
            _new_data = tr.slice(
                starttime=previous_timestamps[channel]).data
            new_data = {'time': new_times[1:], 'data': _new_data[1:]}
            trace_sources[channel].stream(
                new_data=new_data,
                rollover=int(plot_length * tr.stats.sampling_rate))
            previous_timestamps.update({channel: tr.stats.endtime})
            Logger.debug("New data plotted for {0}".format(channel))
        now = dt.datetime.now()
        trace_plots[0].x_range.start = now - dt.timedelta(seconds=plot_length)
        trace_plots[0].x_range.end = now
        # TODO: Update the map - just alphas need to be changed. - streaming might not be the way.
        # TODO: This should look for a detection csv file - where this is should be passed to the function
        # self.map_source.stream(new_data=_new_detections)

    doc.add_periodic_callback(update, update_interval)
    doc.title = "EQcorrscan Real-time plotter"
    doc.add_root(plots)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
