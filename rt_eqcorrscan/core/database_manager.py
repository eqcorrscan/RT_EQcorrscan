"""
Tools for managing a "database" (in the loosest sense) of template information.


"""

import logging

from pathlib import Path

from typing import Optional, Union

from obsplus import EventBank
from obsplus.constants import EVENT_NAME_STRUCTURE
from obsplus.utils import compose_docstring
from obsplus.bank.utils import _get_path

from obspy import Catalog

from eqcorrscan.core.match_filter import Tribe, read_template, Template

Logger = logging.getLogger(__name__)


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

    @compose_docstring
    def get_templates(self, **kwargs) -> Tribe:
        """
        Get template waveforms from the database

        {get_event_params}
        """
        paths = self.bank_path + self.read_index(columns="path", **kwargs).path
        paths = [path.replace(self.ext, self.template_ext) for path in paths]
        templates = [read_template(path) for path in paths]
        return Tribe(templates)

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
