"""
Tools for managing a "database" (in the loosest sense) of template information.


"""

import logging
import os
import json

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from obspy import Catalog, read_events
from obspy.clients.fdsn import Client

from eqcorrscan.core.template_gen import template_gen

Logger = logging.getLogger(__name__)


def _event_time(event):
    """
    Get event time for an event: uses origin if available, else uses first pick

    :type event: `obspy.core.event.Event`
    :param event: event to get the time of
    :return: UTCDateTime
    """
    try:
        origin = event.preferred_origin() or event.origins[0]
    except IndexError:
        origin = None
    if origin and origin.time is not None:
        return origin.time
    if len(event.picks) == 0:
        return None
    return min([p.time for p in event.picks])


def event_to_template_info(event, database_path, st=None, template_file=None,
                           event_file=None, overwrite=True):
    """
    Convert an obspy event to TemplateInfo.

    :param event:
    :param database_path:
    :param st:
    :param template_file:
    :param event_file:
    :param overwrite:
    :return:
    """
    try:
        origin = event.preferred_origin() or event.origins[0]
    except IndexError:
        Logger.warning("Event has no origin attached, setting origin "
                       "parameters to None.")
        origin = None
    if origin is not None:
        lat, lon, depth, origin_time = (
            origin.latitude, origin.longitude, origin.depth, origin.time)
    else:
        lat, lon, depth, origin_time = (None, None, None, None)

    try:
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
    except IndexError:
        Logger.warning("Event has no magnitude attached, setting magnitude"
                       "parameters to None.")
        magnitude = None
    if magnitude is not None:
        mag, magnitude_type = (magnitude.mag, magnitude.type)
    else:
        mag, magnitude_type = (None, None)
    template_info = TemplateInfo(
        eventid=event.resource_id.id.split("/")[-1],
        latitude=lat, longitude=lon, depth=depth, origin_time=origin_time,
        magnitude=mag, magnitude_type=magnitude_type,
        database_path=database_path)
    if event_file is None:
        event_time = _event_time(event)
        template_info.event_file = os.path.join(
            event_time.strftime("%Y"), event_time.strftime("%m"),
            template_info.eventid, "{0}.xml".format(template_info.eventid))
        _db_file = os.path.join(
            template_info.database_path, template_info.event_file)
        if os.path.isfile(_db_file) and not overwrite:
            Logger.warning(
                "{0} exists and overwrite is False, not updating".format(
                    _db_file))
        else:
            event.write(_db_file)
    else:
        template_info.event_file = event_file
    if template_file is None and st is not None:
        event_time = _event_time(event)
        template_info.template_file = os.path.join(
            event_time.strftime("%Y"), event_time.strftime("%m"),
            template_info.eventid, "{0}.ms".format(template_info.eventid))
        _db_file = os.path.join(
            template_info.database_path, template_info.template_file)
        if os.path.isfile(_db_file) and not overwrite:
            Logger.warning(
                "{0} exists and overwrite is False, not updating".format(
                    _db_file))
        else:
            st.write(_db_file, format=format)
    elif template_file is None:
        Logger.info("No template made - recommend you run make_template")
    else:
        template_info.template_file = template_file
    return template_info


def _update_template(template, client_id, overwrite=True, format="MSEED",
                     **kwargs):
    """

    :param client_id:
    :param overwrite:
    :param format:
    :param kwargs:
    :return:
    """
    Logger.debug("Checking template {0}".format(template.eventid))
    if template.event_file is None:
        Logger.info("Downloading event {0}".format(template.eventid))
        _ = template.get_event(
            client_id=client_id, overwrite=overwrite)
    if template.template_file is None:
        Logger.info("Making template {0}".format(template.eventid))
        _ = template.make_template(
            client_id=client_id, overwrite=overwrite, format=format,
            **kwargs)


def read_template_db(db_file):
    """
    Read a json formatted database file into a TemplateDB object.

    :param db_file:
    :return:
    """
    with open(db_file, 'rb') as f:
        db = json.load(f)
    # TODO: Deserialize the dictionary
    return template_db


class TemplateInfo(object):
    def __init__(self, eventid, latitude, longitude, depth, origin_time,
                 magnitude, magnitude_type, database_path, template_file=None,
                 event_file=None):
        self.eventid = eventid
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.origin_time = origin_time
        self.magnitude = magnitude
        self.magnitude_type = magnitude_type
        self.database_path = database_path
        self.template_file = template_file
        self.event_file = event_file

    def __repr__(self):
        return "TemplateInfo({0})".format(", ".join(
            ["{0}={1}".format(key, value)
             for key, value in self.__dict__.items()]))

    def get_event(self, client_id, overwrite=True):
        """
        Download event from a client and write to disk.

        :type client_id:
        :param client_id:
        :type overwrite:
        :param overwrite:

        :return: event
        """
        client = Client(client_id)
        try:
            event = client.get_events(eventid=self.eventid)[0]
        except Exception as e:
            Logger.error(e)
            return None
        event_time = _event_time(event)
        self.event_file = os.path.join(
            event_time.strftime("%Y"), event_time.strftime("%m"),
            self.eventid, "{0}.xml".format(self.eventid))
        _db_file = os.path.join(self.database_path, self.event_file)
        if os.path.isfile(_db_file) and not overwrite:
            Logger.warning(
                "{0} exists and overwrite is False, not updating".format(
                    _db_file))
        else:
            event.write(_db_file)
        return event

    def make_template(self, client_id, overwrite=True, format="MSEED",
                      **kwargs):
        """

        :type client_id:
        :param client_id:
        :type overwrite:
        :param overwrite:
        :type format:
        :param format:
        :param kwargs:
        :return:
        """
        # Make processed template stream
        if self.event_file is None:
            event_file = None
        else:
            event_file = os.path.join(self.database_path, self.event_file)
        if os.path.isfile(event_file):
            cat = read_events(event_file)
        else:
            cat = Catalog(self.get_event(
                client_id=client_id, overwrite=overwrite))
        template = template_gen(
            method="from_client", catalog=cat, client_id=client_id, **kwargs)
        try:
            template = template[0]
        except IndexError:
            Logger.error("No template generated for event {0}".format(
                self.eventid))
            return None
        event_time = _event_time(cat[0])
        self.template_file = os.path.join(
            event_time.strftime("%Y"), event_time.strftime("%m"),
            self.eventid, "{0}.ms".format(self.eventid))
        _db_file = os.path.join(self.database_path, self.template_file)
        if os.path.isfile(_db_file) and not overwrite:
            Logger.warning(
                "{0} exists and overwrite is False, not updating".format(
                    _db_file))
        else:
            template.write(_db_file, format=format)
        # Write to disk
        return template


class TemplateDB(object):
    def __init__(self, db_file, template_path, templates=[]):
        self.db_file = db_file
        self.template_path = template_path
        self.templates = templates

    def __repr__(self):
        return "TemplateDB({0}, {1}, <{2} templates>)".format(
            self.db_file, self.template_path, len(self.templates))

    def __copy__(self):
        return deepcopy(self)

    def __add__(self, other):
        return self.__copy__().__iadd__(other)

    def __iadd__(self, other):
        if isinstance(other, TemplateInfo):
            self.templates.append(other)
        elif isinstance(other, TemplateDB):
            self.templates.extend(other.templates)
            if self.db_file != other.db_file:
                Logger.warning("Merging databases with different db_files, "
                               "recommend writing the output")
            if self.template_path != other.template_path:
                Logger.warning("Merging databases with different template"
                               " paths, recommend moving templates to the "
                               "same place")
        elif isinstance(other, list):
            if all(isinstance(o, TemplateInfo) for o in other):
                self.templates.extend(other)
            else:
                raise TypeError("Unrecognised elements in list")
        else:
            raise TypeError("Unrecognised format for add")
        return self

    def write(self):
        # TODO: serialize the db
        with open(self.db_file, 'wb') as f:
            json.dump(f, serialized_db)

    def update(self, client_id, max_workers=1, overwrite=True, format="MSEED",
               **kwargs):
        """
        Update the database - download any new event file and template waveforms

        :param client_id:
        :return:
        """
        func = partial(_update_template, client_id=client_id,
                       overwrite=overwrite, format=format, **kwargs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(func, self.templates)
        return self

    def select(self, min_latitude=None, max_latitude=None, min_longitude=None,
               max_longitude=None, min_depth=None, max_depth=None,
               latitude=None, longitude=None, radius=None, starttime=None,
               endtime=None, min_magnitude=None, max_magnitude=None):
        """
        Select events meeting criteria

        :param min_latitude:
        :param max_latitude:
        :param min_longitude:
        :param max_longitude:
        :param min_depth:
        :param max_depth:
        :param latitude:
        :param longitude:
        :param radius:
        :param starttime:
        :param endtime:
        :param min_magnitude:
        :param max_magnitude:
        :return:
        """
        selected_db = TemplateDB(
            db_file=self.db_file, template_path=self.template_path,
            templates=[])
        # TODO: Do selections

        return selected_db


if __name__ == "__main__":
    import doctest

    doctest.testmod()
