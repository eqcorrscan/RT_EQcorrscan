"""
Tools for managing a "database" (in the loosest sense) of template information.


"""

import logging
import os

from pathlib import Path

from typing import Optional, Union

from obsplus import EventBank
from obsplus.constants import EVENT_NAME_STRUCTURE, get_events_parameters
from obsplus.utils import compose_docstring
from obsplus.bank.utils import _get_path

from obspy import Catalog, Stream

from eqcorrscan.core.match_filter import Tribe, read_template, Template

Logger = logging.getLogger(__name__)


def _lazy_template_read(path):
    if not os.path.isfile(path):
        Logger.debug("{0} does not exist".format(path))
        return None
    try:
        return read_template(path)
    except Exception as e:
        Logger.error(e)
        return None


class TemplateBank(EventBank):
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

    def put_templates(self, templates: Union[list, Tribe], update_index=True):
        """
        Save templates to the database.

        :param templates: Templates to put into the database
        :param update_index:
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
        **kwargs,
    ):
        """ Make templates from data or client """
        assert client or stream, "Needs either client or stream"
        if stream is not None:
            tribe = Tribe().construct(
                method="from_metafile", meta_file=catalog, stream=stream,
                **kwargs)
            self.put_templates(tribe)
            return
        tribe = Tribe()
        for event in catalog:
            # Get raw data and save to disk
            st = self._get_data_for_event(event, client, download_data_len)
            # Make template add to tribe
            template = Template().construct(
                method="from_metafile", meta_file=event, stream=st,
                **kwargs)
            tribe += template
        self.put_templates(tribe)
        return

    def _get_data_for_event(self, event, client, download_data_len):
        if len(event.picks) == 0:
            Logger.warning("Event has no picks, no template created")
            return Stream()
        bulk = list({
            (p.waveform_id.network_code, p.waveform_id.station_code,
             p.waveform_id.location_code, p.waveforms_id.channel_code,
             p.time - (.45 * download_data_len),
             p.time + (.55 * download_data_len)) for p in event.picks})
        st = client.get_waveforms_bulk(bulk)
        res_id = str(event.resource_id)
        info = {"ext": "ms", "event_id": res_id,
                "event_id_short": res_id.split("/")[-1]}
        path = _get_path(
            info, path_struct=self.path_structure,
            name_struct=self.template_name_structure)["path"]
        ppath = (Path(self.bank_path) / path).absolute()
        ppath.parent.mkdir(parents=True, exist_ok=True)
        st.write(str(ppath), format="MSEED")
        return st


if __name__ == "__main__":
    import doctest

    doctest.testmod()
