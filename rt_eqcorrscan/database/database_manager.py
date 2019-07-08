"""
Tools for managing a "database" (in the loosest sense) of template information.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import logging
import os

from pathlib import Path
from collections import Counter

from typing import Optional, Union

from obsplus import EventBank
from obsplus.constants import EVENT_NAME_STRUCTURE, get_events_parameters
from obsplus.utils import compose_docstring
from obsplus.bank.utils import _get_path

from obspy import Catalog, Stream
from obspy.core.event import Event

from eqcorrscan.core.match_filter import Tribe, read_template, Template

Logger = logging.getLogger(__name__)


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


class TemplateBank(EventBank):
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
            event_id_short
        If no structure is provided it will be read from the index, if no
        index exists the default is {year}/{month}/{day}
    event_name_structure : str
        The same as path structure but for the event file name. Supports the
        same variables and a slash cannot be used in a file name on most
        operating systems. The default extension (.xml) will be added.
        The default is {time}_{event_id_short}.
    template_name_structure
        The same as path structure but for the template file name. Supports the
        same variables and a slash cannot be used in a file name on most
        operating systems. The default extension (.tgz) will be added.
        The default is to use the same naming as event_name_structure.
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
    """
    def __init__(
        self,
        base_path: Union[str, Path, "EventBank"] = ".",
        path_structure: Optional[str] = None,
        event_name_structure: Optional[str] = None,
        template_name_structure: Optional[str] = None,
        cache_size: int = 5,
        event_format="quakeml",
        event_ext=".xml",
        template_ext=".tgz",
    ):
        """Initialize the bank."""
        super().__init__(
            base_path=base_path, path_structure=path_structure,
            name_structure=event_name_structure, cache_size=cache_size,
            format=event_format, ext=event_ext)
        self.template_ext = template_ext
        # get waveform structure based on structures of path and filename
        wns = (template_name_structure or self._name_structure or
               EVENT_NAME_STRUCTURE)
        self.template_name_structure = wns

    @compose_docstring(get_events_params=get_events_parameters)
    def get_templates(self, **kwargs) -> Tribe:
        """
        Get template waveforms from the database

        Supports passing an `concurrent.futures.Executor` using the `executor`
        keyword argument for parallel reading.

        {get_event_params}
        """
        executor = kwargs.pop("executor", None)
        paths = self.bank_path + self.read_index(
            columns=["path", "latitude", "longitude"], **kwargs).path
        paths = [path.replace(self.ext, self.template_ext) for path in paths]
        if executor:
            future = executor.map(_lazy_template_read, paths)
            return Tribe([t for t in future if t is not None])
        else:
            templates = [_lazy_template_read(path) for path in paths]
            return Tribe([t for t in templates if t is not None])

    def put_templates(
        self,
        templates: Union[list, Tribe],
        update_index=True
    ) -> None:
        """
        Save templates to the database.

        Parameters
        ----------
        templates
            Templates to put into the database
        update_index
            Flag to indicate whether or not to update the event index after
            writing the new events. Default is True.
        """
        for t in templates:
            assert(isinstance(t, Template))
        catalog = Catalog([t.event for t in templates])
        self.put_events(catalog, update_index=update_index)
        for template in templates:
            # Get path for template and write it
            res_id = str(template.event.resource_id)
            info = {"ext": self.template_ext, "event_id": res_id,
                    "event_id_short": res_id.split("/")[-1]}
            path = _get_path(
                info, path_struct=self.path_structure,
                name_struct=self.template_name_structure)["path"]
            ppath = (Path(self.bank_path) / path).absolute()
            ppath.parent.mkdir(parents=True, exist_ok=True)
            template.write(str(ppath))

    def make_templates(
        self,
        catalog: Catalog,
        stream: Stream = None,
        client=None,
        download_data_len: float = 600,
        save_raw: bool = True,
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
        kwargs
            Keyword arguments supported by EQcorrscan's `Tribe.construct`
            method.

        Returns
        -------
        Tribe of templates
        """
        assert client or stream, "Needs either client or stream"
        if stream is not None:
            tribe = Tribe().construct(
                method="from_metafile", meta_file=catalog, stream=stream,
                **kwargs)
            self.put_templates(tribe)
            return tribe
        tribe = Tribe()
        for event in catalog:
            # Get raw data and save to disk
            st = self._get_data_for_event(
                event, client, download_data_len, save_raw)
            # Make template add to tribe
            template = Template().construct(
                method="from_metafile", meta_file=event, stream=st,
                **kwargs)
            tribe += template
        self.put_templates(tribe)
        return tribe

    def _get_data_for_event(
        self,
        event: Event,
        client,
        download_data_len: float,
        save_raw: bool = True,
    ) -> Stream:
        """

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
        save_raw
            Whether to store raw data on disk as well - defaults to True.

        Returns
        -------
        Stream as downloaded for the given event.
        """
        if len(event.picks) == 0:
            Logger.warning("Event has no picks, no template created")
            return Stream()
        bulk = list({
            (p.waveform_id.network_code, p.waveform_id.station_code,
             p.waveform_id.location_code, p.waveforms_id.channel_code,
             p.time - (.45 * download_data_len),
             p.time + (.55 * download_data_len)) for p in event.picks})
        st = client.get_waveforms_bulk(bulk)
        if save_raw:
            res_id = str(event.resource_id)
            info = {"ext": "ms", "event_id": res_id,
                    "event_id_short": res_id.split("/")[-1]}
            path = _get_path(
                info, path_struct=self.path_structure,
                name_struct=self.template_name_structure)["path"]
            ppath = (Path(self.bank_path) / path).absolute()
            ppath.parent.mkdir(parents=True, exist_ok=True)
            st.write(str(ppath), format="MSEED")
            Logger.debug("Saved raw data to {0}".format(ppath))
        return st


def check_tribe_quality(
    tribe: Tribe,
    seed_ids: set = None,
    min_stations: int = None,
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

    Returns
    -------
    A filtered tribe.
    """
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

    # Perform station check
    if seed_ids is not None:
        _templates = []
        for template in templates:
            _template = template.copy()
            _st = Stream()
            for tr in _template.st:
                if tr.id in seed_ids:
                    _st += tr
            _template.st = _st
            if min_stations is not None:
                n_sta = len({tr.stats.station for tr in _template.st})
                if n_sta < min_stations:
                    continue
            _templates.append(_template)
        templates = _templates

    return Tribe(templates)


def _test_template_bank(base_path: str) -> TemplateBank:
    """
    Generate a test template bank.

    Parameters
    ----------
    base_path:
        The path to the test database.

    Returns
    -------
    A test template bank for testing porpoises only.
    """
    return TemplateBank(base_path=base_path)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
