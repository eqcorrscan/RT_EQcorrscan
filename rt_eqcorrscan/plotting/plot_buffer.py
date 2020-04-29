"""
Plotting for real-time seismic data.
"""
import numpy as np
import logging
import threading
import datetime as dt
import asyncio

from pyproj import Proj, transform

from bokeh.document import Document
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend, WMTSTileSource
from bokeh.models.glyphs import MultiLine
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.layouts import gridplot, column
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

from functools import partial

from obspy import Inventory
from rt_eqcorrscan.rt_match_filter import RealTimeTribe
from rt_eqcorrscan.streaming.streaming import _StreamingClient

Logger = logging.getLogger(__name__)


class EQcorrscanPlot:
    """
    Streaming bokeh plotting of waveforms.

    Parameters
    ----------
    rt_client
        The real-time streaming client in use.
    plot_length
        Plot length in seconds
    tribe
        Tribe of templates used in real-time detection
    inventory
        Inventory of stations used - will be plotted on the map.
    detections
        List of `eqcorrscan.core.match_filter.Detection`
    update_interval
        Update frequency of plot in ms
    plot_height
        Plot height in screen units
    plot_width
        Plot width in screen units
    exclude_channels
        Iterable of channel codes to exclude from plotting
    offline
        Flag to set time-stamps to data time-stamps if True, else timestamps
        will be real-time
    """
    def __init__(
        self,
        rt_client: _StreamingClient,
        plot_length: float,
        tribe: RealTimeTribe,
        inventory: Inventory,
        detections: list,
        update_interval: float = 100.,
        plot_height: int = 800,
        plot_width: int = 1500,
        exclude_channels: iter = (),
        offline: bool = False,
        new_event_loop: bool = True,
        **plot_data_options,
    ) -> None:
        if new_event_loop:
            # Set up a new event loop for the plots
            asyncio.set_event_loop(asyncio.new_event_loop())
        channels = [tr.id for tr in rt_client.buffer
                    if tr.stats.channel not in exclude_channels]
        Logger.debug("Plot will contain the following channels: {0}".format(
            channels))
        self.channels = sorted(channels)
        self.tribe = tribe
        self.plot_length = plot_length
        self.inventory = inventory
        self.detections = detections

        self.hover = HoverTool(
            tooltips=[
                ("UTCDateTime", "@time{%m/%d %H:%M:%S}"),
                ("Amplitude", "@data")],
            formatters={'time': 'datetime'},
            mode='vline')
        self.map_hover = HoverTool(
            tooltips=[
                ("Latitude", "@lats"),
                ("Longitude", "@lons"),
                ("ID", "@id")])
        self.tools = "pan,wheel_zoom,reset"
        self.plot_options = {
            "plot_width": int(2 * (plot_width / 3)),
            "plot_height": int((plot_height - 20) / len(channels)),
            "tools": [self.hover], "x_axis_type": "datetime"}
        self.map_options = {
            "plot_width": int(plot_width / 3), "plot_height": plot_height,
            "tools": [self.map_hover, self.tools]}
        self.updateValue = True
        Logger.info("Initializing plotter")
        make_doc = partial(
            define_plot, rt_client=rt_client, channels=channels,
            tribe=self.tribe, inventory=self.inventory,
            detections=self.detections, map_options=self.map_options,
            plot_options=self.plot_options, plot_length=self.plot_length,
            update_interval=update_interval, offline=offline,
            **plot_data_options)

        self.apps = {'/RT_EQcorrscan': Application(FunctionHandler(make_doc))}

        self.server = Server(self.apps)
        self.server.start()
        Logger.info("Plotting started")
        self.threads = []
    
    def background_run(self):
        """ run the plotting in a daemon thread. """
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
        """ Stop the background plotting thread. """
        self.server.io_loop.stop()
        for thread in self.threads:
            thread.join()


def define_plot(
    doc: Document,
    rt_client: _StreamingClient,
    channels: list,
    tribe: RealTimeTribe,
    inventory: Inventory,
    detections: list,
    map_options: dict,
    plot_options: dict,
    plot_length: float,
    update_interval: int,
    data_color: str = "grey",
    lowcut: float = 1.0,
    highcut: float = 10.0,
    offline: bool = False,
):
    """
    Set up a bokeh plot for real-time plotting.

    Defines a moving data stream and a map.

    Parameters
    ----------
    doc
        Bokeh document to edit - usually called as a partial
    rt_client
        RealTimeClient streaming data
    channels
        Channels to plot
    tribe
        Tribe to plot
    inventory
        Inventory to plot
    detections
        Detections to plot - should be a list that is updated in place.
    map_options
        Dictionary of options for the map
    plot_options
        Dictionary of options for plotting in general
    plot_length
        Length of data plot
    update_interval
        Update frequency in seconds
    data_color
        Colour to data stream
    lowcut
        Lowcut for filtering data stream
    highcut
        Highcut for filtering data stream
    offline
        Flag to set time-stamps to data time-stamps if True, else timestamps
        will be real-time
    """
    # Set up the data source
    stream = rt_client.get_stream().copy().split().detrend()
    if lowcut and highcut:
        stream.filter("bandpass", freqmin=lowcut, freqmax=highcut)
        title = "Streaming data: {0}-{1} Hz bandpass".format(lowcut, highcut)
    elif lowcut:
        stream.filter("highpass", lowcut)
        title = "Streaming data: {0} Hz highpass".format(lowcut)
    elif highcut:
        stream.filter("lowpass", highcut)
        title = "Streaming data: {0} Hz lowpass".format(highcut)
    else:
        title = "Raw streaming data"
    stream.merge()

    template_lats, template_lons, template_alphas, template_ids = (
        [], [], [], [])
    for template in tribe:
        try:
            origin = (template.event.preferred_origin() or
                      template.event.origins[0])
        except IndexError:
            continue
        template_lats.append(origin.latitude)
        template_lons.append(origin.longitude)
        template_alphas.append(0)
        template_ids.append(template.event.resource_id.id.split("/")[-1])

    station_lats, station_lons, station_ids = ([], [], [])
    for network in inventory:
        for station in network:
            station_lats.append(station.latitude)
            station_lons.append(station.longitude)
            station_ids.append(station.code)

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
        'y': template_y, 'x': template_x,
        'lats': template_lats, 'lons': template_lons,
        'template_alphas': template_alphas, 'id': template_ids})
    station_source = ColumnDataSource({
        'y': station_y, 'x': station_x,
        'lats': station_lats, 'lons': station_lons, 'id': station_ids})

    trace_sources = {}
    trace_data_range = {}
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
        trace_data_range.update({channel: (data.min(), data.max())})

    # Set up the map to go on the left side
    map_plot = figure(
        title="Template map", x_range=map_x_range, y_range=map_y_range,
        x_axis_type="mercator", y_axis_type="mercator", **map_options)
    url = 'http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png'
    attribution = "Tiles by Carto, under CC BY 3.0. Data by OSM, under ODbL"
    map_plot.add_tile(WMTSTileSource(url=url, attribution=attribution))
    map_plot.circle(
        x="x", y="y", source=template_source, fill_color="firebrick",
        line_color="grey", line_alpha=.2,
        fill_alpha="template_alphas", size=10)
    map_plot.triangle(
        x="x", y="y", size=10, source=station_source, color="blue", alpha=1.0)

    # Set up the trace plots
    trace_plots = []
    if not offline:
        now = dt.datetime.utcnow()
    else:
        now = max([tr.stats.endtime for tr in stream]).datetime
    p1 = figure(
        y_axis_location="right", title=title,
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
    detection_source = _get_pick_times(detections, channels[0])
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
            detection_source = _get_pick_times(detections, channel)
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
        _stream = rt_client.get_stream().split().detrend()
        if lowcut and highcut:
            _stream.filter("bandpass", freqmin=lowcut, freqmax=highcut)
        elif lowcut:
            _stream.filter("highpass", lowcut)
        elif highcut:
            _stream.filter("lowpass", highcut)
        _stream.merge()

        for _i, _channel in enumerate(channels):
            try:
                _tr = _stream.select(id=_channel)[0]
            except IndexError:
                Logger.debug("No channel for {0}".format(_channel))
                continue
            new_samples = int(_tr.stats.sampling_rate * (
                    previous_timestamps[_channel] - _tr.stats.endtime))
            if new_samples == 0:
                Logger.debug("No new data for {0}".format(_channel))
                continue
            _new_data = _tr.slice(
                starttime=previous_timestamps[_channel])
            new_times = np.arange(
                _new_data.stats.starttime.datetime,
                (_tr.stats.endtime + _tr.stats.delta).datetime,
                step=dt.timedelta(seconds=_tr.stats.delta))
            new_data = {'time': new_times[1:], 'data': _new_data.data[1:]}
            Logger.debug("Channl: {0}\tNew times: {1}\t New data: {2}".format(
                _tr.id, new_data["time"].shape, new_data["data"].shape))
            trace_sources[_channel].stream(
                new_data=new_data,
                rollover=int(plot_length * _tr.stats.sampling_rate))
            new_picks = _get_pick_times(detections, _channel)
            new_picks.update({
                'pick_values': [
                    [int(np.nan_to_num(
                        trace_sources[_channel].data['data']).max() * .9),
                     int(np.nan_to_num(
                         trace_sources[_channel].data['data']).min() * .9)]
                    for _ in new_picks['picks']]})
            detection_sources[_channel].data = new_picks
            previous_timestamps.update({_channel: _tr.stats.endtime})
            Logger.debug("New data plotted for {0}".format(_channel))
        if not offline:
            now = dt.datetime.utcnow()
        else:
            try:
                now = max([tr.stats.endtime for tr in _stream]).datetime
            except ValueError:
                return
        trace_plots[0].x_range.start = now - dt.timedelta(seconds=plot_length)
        trace_plots[0].x_range.end = now
        _update_template_alphas(
            detections, tribe, decay=plot_length, now=now,
            datastream=template_source)

    doc.add_periodic_callback(update, update_interval)
    doc.title = "EQcorrscan Real-time plotter"
    doc.add_root(plots)


def _update_template_alphas(
    detections: list,
    tribe: RealTimeTribe,
    decay: float,
    now, datastream) -> None:
    """
    Update the template location datastream.

    Parameters
    ----------
    detections
        Detections to use to update the datastream
    tribe
        Templates used
    decay
        Colour decay length in seconds
    now
        Reference time-stamp
    datastream
        Data stream to update
    """
    wgs_84 = Proj(init='epsg:4326')
    wm = Proj(init='epsg:3857')
    template_lats, template_lons, template_alphas, template_ids = (
        [], [], [], [])
    template_x, template_y = ([], [])
    for template in tribe:
        try:
            origin = (template.event.preferred_origin() or
                      template.event.origins[0])
        except IndexError:
            continue
        template_lats.append(origin.latitude)
        template_lons.append(origin.longitude)

        template_ids.append(template.event.resource_id.id.split("/")[-1])
        _x, _y = transform(wgs_84, wm, origin.longitude, origin.latitude)
        template_x.append(_x)
        template_y.append(_y)
        template_detections = [
            d for d in detections if d.template_name == template.name]
        if len(template_detections) == 0:
            template_alphas.append(0)
        else:
            detect_time = min([d.detect_time for d in template_detections])
            offset = (now - detect_time.datetime).total_seconds()
            alpha = 1. - (offset / decay)
            Logger.debug('Updating alpha to {0:.4f}'.format(alpha))
            template_alphas.append(alpha)
    datastream.data = {'y': template_y, 'x': template_x, 'lats': template_lats,
                       'lons': template_lons,
                       'template_alphas': template_alphas, 'id': template_ids}
    return


def _get_pick_times(
    detections: list,
    seed_id: str,
    ignore_channel: bool = True
) -> dict:
    """
    Get new pick times from catalog for a given channel.

    Parameters
    ----------
    detections
        List of detections
    seed_id
        The full Seed-id (net.sta.loc.chan) for extract picks for
    ignore_channel
        Whether to return all picks for a given sensor (e.g. HH*)

    Returns
    -------
    Dictionary with one key ("picks") of the pick-times.
    """
    picks = []
    Logger.debug("Scanning {0} detections for new picks".format(
        len(detections)))
    net, sta, loc, chan = seed_id.split('.')
    for detection in detections:
        try:
            if ignore_channel:
                pick = [p for p in detection.event.picks
                        if p.waveform_id.network_code == net and
                        p.waveform_id.station_code == sta and
                        p.waveform_id.location_code == loc][0]
            else:
                pick = [p for p in detection.event.picks
                        if p.waveform_id.get_seed_string() == seed_id][0]
        except IndexError:
            pick = None
            pass
        if pick:
            Logger.debug("Plotting pick on {0} at {1}".format(
                seed_id, pick.time))
            picks.append([pick.time.datetime, pick.time.datetime])
    return {"picks": picks}


if __name__ == "__main__":
    import doctest

    doctest.testmod()
