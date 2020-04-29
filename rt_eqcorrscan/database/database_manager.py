"""
Tools for managing a database of template information.
"""

import logging
import os

from pathlib import Path
from collections import Counter

from typing import Optional, Union, Callable, Iterable
from concurrent.futures import Executor
from functools import partial

from obsplus import EventBank
from obsplus.constants import (
    EVENT_NAME_STRUCTURE, EVENT_PATH_STRUCTURE, get_events_parameters)
from obsplus.utils.docs import compose_docstring
from obsplus.utils.bank import _get_path, _get_time_values
from obsplus.utils.events import _summarize_event
from obsplus.utils.time import _get_event_origin_time

from obspy import Catalog, Stream
from obspy.core.event import Event

from eqcorrscan.core.match_filter import Tribe, read_template, Template

Logger = logging.getLogger(__name__)

TEMPLATE_EXT = ".tgz"


def _lazy_template_read(path) -> Template:
    """
    Handle exceptions in template reading.

    Parameters
    ----------
    path
        Path to template file

    Returns
    -------
    Template.
    """
    if not os.path.isfile(path):
        Logger.debug("{0} does not exist".format(path))
        return None
    try:
        return read_template(path)
    except Exception as e:
        Logger.error(e)
        return None


def _summarize_template(
    template: Template = None,
    event: Event = None,
    path: Optional[str] = None,
    name: Optional[str] = None,
    path_struct: Optional[str] = None,
    name_struct: Optional[str] = None,
) -> dict:
    """
    Function to extract info from templates for indexing.

    Parameters
    ----------
    template
        The template object
    event
        The template event, give either this or the template.
    path
        Other Parameters to the file
    name
        Name of the file
    path_struct
        directory structure to create
    name_struct
    """
    assert template or event
    if template:
        event = template.event
    res_id = str(event.resource_id)
    out = {
        "ext": TEMPLATE_EXT, "event_id": res_id, "event_id_short": res_id[-5:],
        "event_id_end": res_id.split('/')[-1]}
    t1 = _get_event_origin_time(event)
    out.update(_get_time_values(t1))
    path_struct = path_struct or EVENT_PATH_STRUCTURE
    name_struct = name_struct or EVENT_NAME_STRUCTURE

    out.update(_get_path(out, path, name, path_struct, name_struct))
    return out


class _Result(object):
    """
     Thin imitation of concurrent.futures.Future

     .. rubric:: example
     >>> result = _Result(12)
     >>> print(result)
     _Result(12)
     >>> print(result.result())
     12
     """
    def __init__(self, result):
        self._result = result

    def __repr__(self):
        return "_Result({0})".format(self._result)

    def result(self):
        return self._result


class _SerialExecutor(Executor):
    """
    Simple interface to mirror concurrent.futures.Executor in serial.

    .. rubric:: Example
    >>> with _SerialExecutor() as executor:
    ...     result = executor.submit(pow, 2, 12)
    >>> print(result.result())
    4096
    >>> def square(a):
    ...     return a * a
    >>> with _SerialExecutor() as executor:
    ...     futures = executor.map(square, range(5))
    >>> results = list(futures)
    >>> print(results)
    [0, 1, 4, 9, 16]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def map(
        self,
        fn: Callable,
        *iterables: Iterable,
        timeout=None,
        chunksize=1,
    ):
        """
        Map iterables to a function

        Parameters
        ----------
        fn
            callable
        iterables
            iterable of arguments for `fn`
        timeout
            Throw-away variable
        chunksize
            Throw-away variable

        Returns
        -------
        The result of mapping `fn` across `*iterables`
        """
        return map(fn, *iterables)

    def submit(self, fn: Callable, *args, **kwargs) -> _Result:
        """
        Run a single function.

        Parameters
        ----------
        fn
            Function to apply
        args
            Arguments for `fn`
        kwargs
            Key word arguments for `fn`

        Returns
        -------
        result with the result in the .result attribute
        """
        return _Result(fn(*args, **kwargs))


class TemplateBank(EventBank):
    # TODO: This should probably be it's own _Bank subclass with extra columns including if template exists and template channels - would allow for faster filtering
    """
    A database manager for templates. Based on obsplus EventBank.

    Parameters
    ----------
    base_path
        The path to the directory containing event files. If it does not
        exist an empty directory will be created.
    path_structure
        Define the directory structure used by the event bank. Characters are
        separated by /, regardless of operating system. The following
        words can be used in curly braces as data specific variables:
        year, month, day, julday, hour, minute, second, event_id,
        event_id_short. If no structure is provided it will be read from the
        index, if no index exists the default is {year}/{month}/{day}
    name_structure : str
        The same as path structure but for the event, template and waveform
        file names. Supports the same variables and a slash cannot be used
        in a file name on most operating systems. The default extension
        (.xml, .tgz, .ms) will be added for events, templates and waveforms
        respectively. The default is {time}_{event_id_short}.
    event_format
        The anticipated format of the event files. Any format supported by the
        obspy.read_events function is permitted.
    event_ext
        The extension on the event files. Can be used to avoid parsing
        non-event files.
    template_ext
        The extension on the template files. Can be used to avoid parsing
        non-template files.
    cache_size : int
        The number of queries to store. Avoids having to read the index of
        the database multiple times for queries involving the same start and
        end times.

    Notes
    -----
        Supports parallel execution of most methods using `concurrent.futures`
        executors. Set the `TemplateBank.executor` attribute to your required
        executor.
    """
    def __init__(
        self,
        base_path: Union[str, Path, "EventBank"] = ".",
        path_structure: Optional[str] = None,
        name_structure: Optional[str] = None,
        cache_size: int = 5,
        event_format="quakeml",
        event_ext=".xml",
        template_ext=".tgz",
        executor=None
    ):
        """Initialize the bank."""
        super().__init__(
            base_path=base_path, path_structure=path_structure,
            name_structure=name_structure, cache_size=cache_size,
            format=event_format, ext=event_ext)
        self.template_ext = template_ext
        self.executor = executor or _SerialExecutor()

    @compose_docstring(get_events_params=get_events_parameters)
    def get_templates(self, **kwargs) -> Tribe:
        """
        Get template waveforms from the database

        Supports passing an `concurrent.futures.Executor` using the `executor`
        keyword argument for parallel reading.

        {get_event_params}
        """
        paths = str(self.bank_path) + self.read_index(
            columns=["path", "latitude", "longitude"], **kwargs).path
        paths = [path.replace(self.ext, self.template_ext) for path in paths]
        future = self.executor.map(_lazy_template_read, paths)
        return Tribe([t for t in future if t is not None])

    def put_templates(
        self,
        templates: Union[list, Tribe],
        update_index: bool = True,
    ) -> None:
        """
        Save templates to the database.

        Parameters
        ----------
        templates
            Templates to put into the database
        update_index
            Flag to indicate whether or not to update the event index
            after writing the new events.
        """
        for t in templates:
            assert(isinstance(t, Template))
        catalog = Catalog([t.event for t in templates])
        self.put_events(catalog, update_index=update_index)
        inner_put_template = partial(
            _put_template, path_structure=self.path_structure,
            template_name_structure=self.name_structure,
            bank_path=self.bank_path)
        _ = [_ for _ in self.executor.map(inner_put_template, templates)]

    def make_templates(
        self,
        catalog: Catalog,
        stream: Stream = None,
        client=None,
        download_data_len: float = 90,
        save_raw: bool = True,
        update_index: bool = True,
        rebuild: bool = True,
        **kwargs,
    ) -> Tribe:
        """
        Make templates from data or client based on a given catalog.

        Templates will be put in the database. Requires either a stream or
        a suitable client with a get_waveforms method.

        Parameters
        ----------
        catalog
            Catalog of events to generate templates from.
        stream
            Optional: Stream encompassing the events in the catalog
        client
            Optional: Client with at-least a `get_waveforms` method, ideally
            the client should make the data for the events in catalog
            available.
        download_data_len
            If client is given this is the length of data to download. The
            raw continuous data will also be saved to disk to allow later
            processing if save_raw=True
        save_raw
            Whether to store raw data on disk as well - defaults to True.
        update_index
            Flag to indicate whether or not to update the event index after
            writing the new events.
        rebuild
            Flag to indicate whether templates already existing in the
            TemplateBank should be re-generated. `True` will overwrite existing
            templates.
        kwargs
            Keyword arguments supported by EQcorrscan's `Template.construct`
            method. Requires at least: lowcut, highcut, samp_rate, filt_order,
            prepick, length, swin

        Returns
        -------
        Tribe of templates
        """
        assert client or stream, "Needs either client or stream"
        if stream is not None:
            tribe = Tribe().construct(
                method="from_metafile", meta_file=catalog, stream=stream,
                **kwargs)
        else:
            Logger.debug("Making templates")
            inner_download_and_make_template = partial(
                _download_and_make_template, client=client,
                download_data_len=download_data_len,
                path_structure=self.path_structure,
                bank_path=self.bank_path,
                template_name_structure=self.name_structure,
                save_raw=save_raw, rebuild=rebuild, **kwargs)
            template_iterable = self.executor.map(
                inner_download_and_make_template, catalog)
            tribe = Tribe([t for t in template_iterable if t is not None])
        Logger.info(f"Putting {len(tribe)} templates into database")
        self.put_templates(tribe, update_index=update_index)
        Logger.info("Finished putting templates into database.")
        return tribe


def _put_template(
    template: Template,
    path_structure: str,
    template_name_structure: str,
    bank_path: str,
) -> str:
    """ Get path for template and write it. """
    path = _summarize_template(
        template=template, path_struct=path_structure,
        name_struct=template_name_structure)["path"]
    ppath = (Path(bank_path) / path).absolute()
    ppath.parent.mkdir(parents=True, exist_ok=True)
    output_path = str(ppath)
    template.write(output_path)
    # Issue with older EQcorrscan doubling up extension
    if os.path.isfile(output_path + TEMPLATE_EXT):
        os.rename(output_path + TEMPLATE_EXT, output_path)
    return output_path


def _download_and_make_template(
    event: Event,
    client,
    download_data_len: float,
    path_structure: str,
    bank_path: str,
    template_name_structure: str,
    save_raw: bool,
    rebuild: bool,
    **kwargs,
) -> Template:
    """ Make the template using downloaded data"""
    Logger.debug("Making template for event {0}".format(event.resource_id))
    if not rebuild:
        try:
            path = _summarize_template(
                event=event, path_struct=path_structure,
                name_struct=template_name_structure)["path"]
        except ValueError as e:
            Logger.error(f"Could not summarize event due to {e}")
            return None
        ppath = (Path(bank_path) / path).absolute()
        ppath.parent.mkdir(parents=True, exist_ok=True)
        output_path = str(ppath)
        if os.path.isfile(output_path):
            Logger.debug("Template exists and rebuild=False, skipping")
            return read_template(output_path)
    # Sanitize event - sometime Arrivals or not linked.
    pick_dict = {p.resource_id.id: p for p in event.picks}
    for origin in event.origins:
        origin.arrivals = [
            arr for arr in origin.arrivals 
            if arr.pick_id in pick_dict]
    _process_len = kwargs.pop("process_len", download_data_len)
    if _process_len > download_data_len:
        Logger.info(
            "Downloading {0}s of data as required by process len".format(
                _process_len))
        download_data_len = _process_len
    st = _get_data_for_event(
        event=event, client=client,
        download_data_len=download_data_len, path_structure=path_structure,
        bank_path=bank_path, template_name_structure=template_name_structure,
        save_raw=save_raw)
    if st is None:
        return None
    Logger.debug("Downloaded {0} traces for event {1}".format(
        len(st), event.resource_id))
    try:
        tribe = Tribe().construct(
            method="from_meta_file", meta_file=Catalog([event]), st=st,
            process_len=download_data_len, **kwargs)
    except Exception as e:
        Logger.error(e)
        return None
    try:
        template = tribe[0]
        Logger.info("Made template: {0}".format(template))
    except IndexError as e:
        Logger.error(e)
        return None
    template.name = event.resource_id.id.split('/')[-1]
    # Edit comment to reflect new template_name
    for comment in template.event.comments:
        if comment.text and comment.text.startswith("eqcorrscan_template_"):
            comment.text = "eqcorrscan_template_{0}".format(template.name)
    return template


def _get_data_for_event(
    event: Event,
    client,
    download_data_len: float,
    path_structure: str,
    bank_path: str,
    template_name_structure: str,
    save_raw: bool = True,
) -> Stream:
    """
    Get data for a given event.

    Parameters
    ----------
    event
        Event to download data for.
    client
        Optional: Client with at-least a `get_waveforms` method, ideally
        the client should make the data for the events in catalog
        available.
    download_data_len
        If client is given this is the length of data to download. The
        raw continuous data will also be saved to disk to allow later
        processing if save_raw=True
    path_structure
        Bank path structure for writing data
    bank_path
        Location of bank to write data to
    template_name_structure
        Naming structure for template files.
    save_raw
        Whether to store raw data on disk as well - defaults to True.

    Returns
    -------
    Stream as downloaded for the given event.
    """
    if len(event.picks) == 0:
        Logger.warning("Event has no picks, no template created")
        return Stream()
    bulk = [
        (p.waveform_id.network_code, p.waveform_id.station_code,
         p.waveform_id.location_code, p.waveform_id.channel_code,
         p.time - (.5 * download_data_len),
         p.time + (.6 * download_data_len)) for p in event.picks]
    Logger.debug(bulk)
    try:
        st = client.get_waveforms_bulk(bulk)
    except Exception as e:
        Logger.error(e)
        st = Stream()
        for channel in bulk:
            Logger.debug("Downloading individual channel {0}".format(channel))
            try:
                st += client.get_waveforms(
                    network=channel[0], station=channel[1],
                    location=channel[2], channel=channel[3],
                    starttime=channel[4], endtime=channel[5])
            except Exception as e:
                Logger.error(e)
    # Trim to expected length
    try:
        st.merge()
    except Exception as e:
        Logger.error(e)
        return None
    # Cope with multiple picks on the same channel at different times.
    trimmed_stream = Stream()
    for pick in event.picks:
        try:
            tr = st.select(id=pick.waveform_id.get_seed_string())[0]
        except IndexError:
            Logger.warning("No data downloaded for {0}".format(
                pick.waveform_id.get_seed_string()))
            continue
        trimmed_stream += tr.slice(
            starttime=pick.time - (.45 * download_data_len),
            endtime=pick.time + (.55 * download_data_len)).copy()
    if len(trimmed_stream) == 0:
        Logger.error("No data downloaded, no template.")
        return None
    if save_raw:
        path = _summarize_event(
            event=event, path_struct=path_structure,
            name_struct=template_name_structure)["path"]
        path, _ = os.path.splitext(path)
        path += ".ms"
        ppath = (Path(bank_path) / path).absolute()
        ppath.parent.mkdir(parents=True, exist_ok=True)
        trimmed_stream.split().write(str(ppath), format="MSEED")
        Logger.debug("Saved raw data to {0}".format(ppath))
    return trimmed_stream


def check_tribe_quality(
    tribe: Tribe,
    seed_ids: set = None,
    min_stations: int = None,
    lowcut: float = None,
    highcut: float = None,
    filt_order: int = None,
    samp_rate: float = None,
    process_len: float = None,
    *args, **kwargs
) -> Tribe:
    """
    Check that templates in the tribe have channels all the same length.

    Parameters
    ----------
    tribe
        A Tribe to check the quality of.
    seed_ids
        seed-ids of channels to be included in the templates - if None,
        then all channels will be included
    min_stations
        Minimum number of stations for a template to be included.
    lowcut
        Desired template processing lowcut in Hz, if None, will not check
    highcut
        Desired template processing highcut in Hz, if None, will not check
    filt_order
        Desired template filter order, if None, will not check
    samp_rate
        Desired template sampling rate in Hz, if None, will not check
    process_len
        Desired template processing length in s, if None, will not check

    Returns
    -------
    A filtered tribe.
    """
    processing_keys = dict(
        lowcut=lowcut, highcut=highcut, filt_order=filt_order,
        samp_rate=samp_rate, process_length=process_len)
    Logger.info("Checking processing parameters: {0}".format(processing_keys))
    min_stations = min_stations or 0
    _templates = []
    # Perform length check
    for template in tribe:
        counted_lengths = Counter([tr.stats.npts for tr in template.st])
        if len(counted_lengths) > 1:
            Logger.warning(
                "Multiple lengths found in template, using most common"
                " ({0})".format(counted_lengths.most_common(1)[0][0]))
            _template = template.copy()
            _template.st = Stream()
            for tr in template.st:
                if tr.stats.npts == counted_lengths.most_common(1)[0][0]:
                    _template.st += tr
            _templates.append(_template)
        else:
            _templates.append(template)
    templates = _templates

    # Check processing parameters
    _templates = []
    for template in templates:
        for processing_key, processing_value in processing_keys.items():
            if processing_value and template.__dict__[processing_key] != processing_value:
                Logger.warning("Template {0}: {1} does not match {2}".format(
                    processing_key, template.__dict__[processing_key],
                    processing_value))
                break
        else:
            _templates.append(template)
    templates = _templates

    # Perform station check
    if seed_ids is None:
        seed_ids = {tr.id for template in tribe for tr in template.st}
    for template in templates:
        _st = Stream()
        for tr in template.st:
            if tr.id in seed_ids:
                _st += tr
        template.st = _st
    return Tribe([t for t in templates
                  if len({tr.stats.station for tr in t.st}) > min_stations])


def remove_unreferenced(catalog: Union[Catalog, Event]) -> Catalog:
    """ Remove un-referenced arrivals, amplitudes and station_magnitudes. """
    if isinstance(catalog, Event):
        catalog = Catalog([catalog])
    catalog_out = Catalog()
    for _event in catalog:
        event = _event.copy()
        pick_ids = {p.resource_id for p in event.picks}
        # Remove unreferenced arrivals
        for origin in event.origins:
            origin.arrivals = [
                arr for arr in origin.arrivals if arr.pick_id in pick_ids]
        # Remove unreferenced amplitudes
        event.amplitudes = [
            amp for amp in event.amplitudes if amp.pick_id in pick_ids]
        amplitude_ids = {a.resource_id for a in event.amplitudes}
        # Remove now unreferenced station magnitudes
        event.station_magnitudes = [
            sta_mag for sta_mag in event.station_magnitudes
            if sta_mag.amplitude_id in amplitude_ids]
        station_magnitude_ids = {
            sta_mag.resource_id for sta_mag in event.station_magnitudes}
        # Remove unreferenced station_magnitude_contributions
        for magnitude in event.magnitudes:
            magnitude.station_magnitude_contributions = [
                sta_mag_contrib
                for sta_mag_contrib in magnitude.station_magnitude_contributions
                if sta_mag_contrib.station_magnitude_id in station_magnitude_ids]
        catalog_out.append(event)

    return catalog_out


if __name__ == "__main__":
    import doctest

    doctest.testmod()
