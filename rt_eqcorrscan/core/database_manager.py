"""
Tools for managing a "database" (in the loosest sense) of template information.


"""

import logging

from pathlib import Path
from typing import Optional, Union

from obsplus import EventBank
from obsplus.constants import EVENT_NAME_STRUCTURE

from eqcorrscan import Tribe

from eqcorrscan.core.template_gen import template_gen

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
        template_format="mseed",
        template_ext=".ms",
    ):
        """Initialize the bank."""
        super().__init__(
            base_path=base_path, path_structure=path_structure,
            name_structure=event_name_structure, cache_size=cache_size,
            format=event_format, ext=event_ext)
        self.template_format = template_format
        self.template_ext = template_ext
        # get waveform structure based on structures of path and filename
        wns = (template_name_structure or self._name_structure or
               EVENT_NAME_STRUCTURE)
        self.template_name_structure = wns

    def get_templates(self, **kwargs) -> list:
        """
        Get template waveforms from the database

        :param kwargs:
        :return:
        """

    def put_templates(self, templates: Union[list, Tribe]):
        """
        Save templates to the database.

        :param templates:
        :return:
        """

    def update_templates(self):
        """
        Update the list of templates

        :return:
        """


if __name__ == "__main__":
    import doctest

    doctest.testmod()
