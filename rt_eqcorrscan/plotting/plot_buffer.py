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

from pyproj import Proj, transform

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend, WMTSTileSource
from bokeh.models.glyphs import MultiLine
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
                 detection_catalog, update_interval=100, plot_height=800,
                 plot_width=1500):
        channels = [tr.id for tr in rt_client.buffer]
        self.channels = sorted(channels)
        self.plot_length = plot_length
        self.template_catalog = template_catalog
        self.inventory = inventory
        self.detection_catalog = detection_catalog

        self.hover = HoverTool(
            tooltips=[
                ("UTCDateTime", "@time{%m/%d %H:%M:%S}"),
                ("Amplitude", "@data")],
            formatters={'time': 'datetime'},
            mode='vline')
        self.tools = "pan,wheel_zoom,reset"
        self.plot_options = {
            "plot_width": int(2 * (plot_width / 3)),
            "plot_height": int((plot_height - 20) / len(channels)),
            "tools": [self.hover, self.tools], "x_axis_type": "datetime"}
        self.map_options = {
            "plot_width": int(plot_width / 3), "plot_height": plot_height,
            "tools": [self.hover, self.tools]}
        self.updateValue = True
        Logger.info("Initializing plotter")
        make_doc = partial(
            define_plot, rt_client=rt_client, channels=channels,
            catalog=self.template_catalog, inventory=self.inventory,
            detection_catalog=self.detection_catalog,
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


def define_plot(doc, rt_client, channels, catalog, inventory,
                detection_catalog, map_options, plot_options, plot_length,
                update_interval, data_color="grey"):
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

    # Get plot bounds in web mercator
    wgs_84 = Proj(init='epsg:4326')
    wm = Proj(init='epsg:3857')
    try:
        min_lat, min_lon, max_lat, max_lon = (
            min(template_lats + station_lats),
            min(template_lons + station_lons),
            max(template_lats + station_lats),
            max(template_lons + station_lons))
    except ValueError as e:
        Logger.error(e)
        Logger.info("Setting map bounds to NZ")
        min_lat, min_lon, max_lat, max_lon = (-47., 165., -34., 179.9)
    bottom_left = transform(wgs_84, wm, min_lon, min_lat)
    top_right = transform(wgs_84, wm, max_lon, max_lat)
    map_x_range = (bottom_left[0], top_right[0])
    map_y_range = (bottom_left[1], top_right[1])

    template_x, template_y = ([], [])
    for lon, lat in zip(template_lons, template_lats):
        _x, _y = transform(wgs_84, wm, lon, lat)
        template_x.append(_x)
        template_y.append(_y)

    station_x, station_y = ([], [])
    for lon, lat in zip(station_lons, station_lats):
        _x, _y = transform(wgs_84, wm, lon, lat)
        station_x.append(_x)
        station_y.append(_y)

    template_source = ColumnDataSource({
        'template_lats': template_y,
        'template_lons': template_x,
        'template_alphas': template_alphas})
    station_source = ColumnDataSource({
        'station_lats': station_y,
        'station_lons': station_x})

    trace_sources = {}
    # Allocate empty arrays
    for channel in channels:
        tr = stream.select(id=channel)[0]
        times = np.arange(
            tr.stats.starttime.datetime,
            (tr.stats.endtime + tr.stats.delta).datetime,
            step=dt.timedelta(seconds=tr.stats.delta))
        data = tr.data
        trace_sources.update(
            {channel: ColumnDataSource({'time': times, 'data': data})})

    # Set up the map to go on the left side
    map_plot = figure(
        title="Template map", x_range=map_x_range, y_range=map_y_range,
        x_axis_type="mercator", y_axis_type="mercator", **map_options)
    url = 'http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png'
    attribution = "Tiles by Carto, under CC BY 3.0. Data by OSM, under ODbL"
    map_plot.add_tile(WMTSTileSource(url=url, attribution=attribution))
    map_plot.circle(
        x="template_lons", y="template_lats", source=template_source,
        color="firebrick", fill_alpha="template_alphas")
    map_plot.triangle(
        x="station_lons", y="station_lats", source=station_source,
        color="blue", alpha=1.0)

    # Set up the trace plots
    trace_plots = []
    now = dt.datetime.utcnow()
    p1 = figure(
        y_axis_location="right", title="Streaming Data",
        x_range=[now - dt.timedelta(seconds=plot_length), now],
        plot_height=int(plot_options["plot_height"] * 1.2),
        **{key: value for key, value in plot_options.items()
           if key != "plot_height"})
    p1.yaxis.axis_label = None
    p1.xaxis.axis_label = None
    p1.min_border_bottom = 0
    p1.min_border_top = 0
    if len(channels) != 1:
        p1.xaxis.major_label_text_font_size = '0pt'
    p1_line = p1.line(
        x="time", y='data', source=trace_sources[channels[0]],
        color=data_color, line_width=1)
    legend = Legend(items=[(channels[0], [p1_line])])
    p1.add_layout(legend, 'right')

    datetick_formatter = DatetimeTickFormatter(
        days=["%m/%d"], months=["%m/%d"],
        hours=["%m/%d %H:%M:%S"], minutes=["%m/%d %H:%M:%S"],
        seconds=["%m/%d %H:%M:%S"], hourmin=["%m/%d %H:%M:%S"],
        minsec=["%m/%d %H:%M:%S"])
    p1.xaxis.formatter = datetick_formatter

    # Add detection lines
    detection_source = _get_pick_times(detection_catalog, channels[0])
    detection_source.update(
        {"pick_values": [[
            int(min(stream.select(id=channels[0])[0].data) * .9),
            int(max(stream.select(id=channels[0])[0].data) * .9)]
            for _ in detection_source['picks']]})
    detection_sources = {channels[0]: ColumnDataSource(detection_source)}
    detection_lines = MultiLine(
        xs="picks", ys="pick_values", line_color="red", line_dash="dashed",
        line_width=1)
    p1.add_glyph(detection_sources[channels[0]], detection_lines)

    trace_plots.append(p1)

    if len(channels) > 1:
        for i, channel in enumerate(channels[1:]):
            p = figure(
                x_range=p1.x_range,
                y_axis_location="right", **plot_options)
            p.yaxis.axis_label = None
            p.xaxis.axis_label = None
            p.min_border_bottom = 0
            # p.min_border_top = 0
            p_line = p.line(
                x="time", y="data", source=trace_sources[channel],
                color=data_color, line_width=1)
            legend = Legend(items=[(channel, [p_line])])
            p.add_layout(legend, 'right')
            p.xaxis.formatter = datetick_formatter

            # Add detection lines
            detection_source = _get_pick_times(detection_catalog, channel)
            detection_source.update(
                {"pick_values": [[
                    int(min(stream.select(id=channel)[0].data) * .9),
                    int(max(stream.select(id=channel)[0].data) * .9)]
                    for _ in detection_source['picks']]})
            detection_sources.update({
                channel: ColumnDataSource(detection_source)})
            detection_lines = MultiLine(
                xs="picks", ys="pick_values", line_color="red",
                line_dash="dashed", line_width=1)
            p.add_glyph(detection_sources[channel], detection_lines)

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
        for i, channel in enumerate(channels):
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
            new_picks = _get_pick_times(detection_catalog, channel)
            new_picks.update({
                'pick_values': [
                    [int(trace_plots[i].y_range.start * .9),
                     int(trace_plots[i].y_range.end * .9)]
                    for _ in new_picks['picks']]})
            detection_sources[channel].stream(
                new_data=new_picks,
                rollover=int(plot_length * tr.stats.sampling_rate))
            previous_timestamps.update({channel: tr.stats.endtime})
            Logger.debug("New data plotted for {0}".format(channel))
        now = dt.datetime.utcnow()
        trace_plots[0].x_range.start = now - dt.timedelta(seconds=plot_length)
        trace_plots[0].x_range.end = now
        # TODO: Update the map - just alphas need to be changed. - streaming might not be the way.
        # TODO: This should look take the detection catalog as it is updated
        # self.map_source.stream(new_data=_new_detections)

    doc.add_periodic_callback(update, update_interval)
    doc.title = "EQcorrscan Real-time plotter"
    doc.add_root(plots)


def _get_pick_times(catalog, seed_id):
    picks = []
    for event in catalog:
        try:
            pick = [p for p in event.picks
                    if p.waveform_id.get_seed_string() == seed_id][0]
        except IndexError:
            pick = None
            pass
        if pick:
            picks.append([pick.time.datetime, pick.time.datetime])
    return {"picks": picks}


if __name__ == "__main__":
    import doctest

    doctest.testmod()
