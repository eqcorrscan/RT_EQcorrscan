"""
Overarching tool for listening to and triggering from FDSN earthquakes.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import logging
import time
import gc
import os
import subprocess

from typing import Callable, Iterable
from multiprocessing import cpu_count

from obspy import UTCDateTime
from obspy.core.event import Event

from obsplus.events import get_events

from rt_eqcorrscan.database.database_manager import TemplateBank
from rt_eqcorrscan.event_trigger.catalog_listener import CatalogListener
from rt_eqcorrscan.streaming.streaming import _StreamingClient
from rt_eqcorrscan.config import Notifier, Config


Logger = logging.getLogger(__name__)


def _get_triggered_working_dir(
        triggering_event_id: str,
        exist_ok: bool = True
) -> str:
    working_dir = os.path.join(
        os.path.abspath(os.getcwd()), triggering_event_id)
    os.makedirs(working_dir, exist_ok=exist_ok)
    return working_dir


class Reactor(object):
    """
    A class to listen to a client and start-up a real-time instance.

    The real-time instance will be triggered by the listener when set
    conditions are met. Appropriate templates will be extracted from the
    template database on disk and used a Tribe for the real-time detection.

    Once triggered, the listener will keep listening, but will not trigger in
    the same region again while the real-time detector is running. The real-time
    detector has to be stopped manually.

    Parameters
    ----------
    client
        An obspy or obsplus client that supports event and station queries.
    listener
        Listener for checking current earthquake activity
    trigger_func:
        A function that returns a list of events that exceed some trigger
        parameters given a catalog. See note below.
    template_database
        A template database to be used to generate tribes for real-time
        matched-filter detection.
    config
        Configuration for RT-EQcorrscan.
    notifier
        Notifier that will send messages about triggers.

    Notes
    -----
    `trigger_func` must only take one argument, a catalog of events. To
    achieve this you should generate a partial function from your trigger
    function. For example, using the provided rate-and-magnitude triggering
    function:
    ```
        from rt_eqcorrscan.event_trigger import magnitude_rate_trigger_func
        from functools import partial

        trigger_func = partial(
            magnitude_rate_trigger_func, magnitude_threshold=4,
            rate_threshold=20, rate_bin=0.5)
    ```
    """
    _triggered_events = []
    _running_templates = dict()
    # Template events ids keyed by triggering-event-id
    _running_regions = dict()  # Regions keyed by triggering-event-id
    max_station_distance = 1000
    n_stations = 10
    sleep_step = 15

    # Maximum processors dedicated to one detection routine.
    _max_detect_cores = 12

    # The processes that are detecting away!
    detecting_processes = dict()
    # TODO: Build in an AWS spin-up functionality - would need some communication...

    def __init__(
        self,
        client,
        listener: CatalogListener,
        trigger_func: Callable,
        template_database: TemplateBank,
        config: Config,
        notifier: Notifier = None,
    ):
        self.client = client
        self.listener = listener
        self.trigger_func = trigger_func
        self.template_database = template_database
        self.config = config
        self._listener_kwargs = dict(
            min_stations=config.database_manager.min_stations,
            template_kwargs=config.template)
        self.notifier = notifier or Notifier()
        # Time-keepers
        self._run_start = None
        self.up_time = 0

    @property
    def available_cores(self) -> int:
        return max(1, cpu_count() - 1)

    @property
    def running_template_ids(self) -> set:
        _ids = set()
        for value in self._running_templates.values():
            _ids.update(value)
        return _ids

    def get_up_time(self):
        return self._up_time

    def set_up_time(self, now):
        if self._run_start is not None:
            self._up_time = now - self._run_start
        else:
            self._up_time = 0

    up_time = property(get_up_time, set_up_time)

    def run(self) -> None:
        """
        Run all the processes.
        """
        self.listener.background_run(**self._listener_kwargs)
        self._run_start = UTCDateTime.now()
        # Query the catalog in the listener every so often and check
        while True:
            if len(self.listener.old_events) > 0:
                working_ids = list(zip(*self.listener.old_events))[0]
                working_cat = self.template_database.get_events(
                    eventid=working_ids)
            else:
                working_cat = []
            Logger.debug("Currently analysing a catalog of {0} events".format(
                len(working_cat)))
            # Check if new events should be in one of the already running
            # tribes and add them.
            for triggering_event_id, tribe_region in self._running_regions.items():
                add_events = get_events(working_cat, **tribe_region)
                # TODO: Implement region growth based on new events added.
                added_ids = {e.resource_id.split('/') for e in add_events}
                if added_ids:
                    tribe = self.template_database.get_templates(
                        event_ids=added_ids)
                    tribe.write(os.path.join(
                        _get_triggered_working_dir(triggering_event_id),
                        "tribe.tgz"))
                    self._running_templates[triggering_event_id].update(
                        added_ids)
                    working_cat.events = [e for e in working_cat
                                          if e not in add_events]
            trigger_events = self.trigger_func(working_cat)
            for trigger_event in trigger_events:
                if trigger_event not in self._triggered_events:
                    Logger.warning(
                        "Listener triggered by event {0}".format(
                            trigger_event))
                    self.notifier.notify(
                        message="Listener triggered by event {0}".format(
                            trigger_event),
                        level=5)
                    if len(self._running_regions) >= self.available_cores:
                        Logger.error("No more available processors")
                        continue
                    self._triggered_events.append(trigger_event)
                    self.spin_up(trigger_event)
            self.set_up_time(UTCDateTime.now())
            time.sleep(self.sleep_step)

    def spin_up(
        self,
        triggering_event: Event,
    ):
        """
        Run the reactors response function as a subprocess.

        Parameters
        ----------
        triggering_event
            Event that triggered this run - needs to have at-least an origin.
        """
        triggering_event_id = triggering_event.resource_id.split('/')[-1]
        region = estimate_region(triggering_event)
        if region is None:
            return None, None
        region.update(
            {"starttime": self.config.database_manager.lookup_starttime})
        Logger.info("Getting templates within {0}".format(region))
        df = self.template_database.get_event_summary(**region)
        event_ids = {e.split('/')[-1] for e in df["event_id"]}
        event_ids = event_ids.difference(self.running_template_ids)
        # Write file of event id's
        if len(event_ids) == 0:
            return
        working_dir = _get_triggered_working_dir(
            triggering_event_id, exist_ok=False)
        tribe = self.template_database.get_templates(event_ids=event_ids)
        tribe.write(os.path.join(working_dir, "tribe.tgz"))
        self.config.write(
            os.path.join(working_dir, 'rt_eqcorrscan_config.yml'))
        triggering_event.write(
            os.path.join(working_dir, 'triggering_event.xml'), format="QUAKEML")
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "spin_up.py")
        proc = subprocess.Popen(
            ["python", script_path, "-w", working_dir,
             "-n", min(self.available_cores, self._max_detect_cores)])
        self.detecting_processes.update({triggering_event_id: proc})
        self._running_regions.update({triggering_event_id: region})
        self._running_templates.update(
            {triggering_event_id:
                 {t.event.resource_id.split('/')[-1] for t in tribe}})
        Logger.info("Started detector subprocess - continuing listening")

    def stop_tribe(self, triggering_event_id: str = None) -> None:
        """
        Stop a specific tribe.

        Parameters
        ----------
        triggering_event_id
            The id of the triggering event for which a tribe is running.
        """
        if triggering_event_id is None:
            return self.stop()
        self.detecting_processes[triggering_event_id].kill()
        self.detecting_processes.pop(triggering_event_id)
        self._running_templates.pop(triggering_event_id)
        return None

    def stop(self) -> None:
        """Stop all the processes."""
        for event_id in self.detecting_processes.keys():
            self.stop_tribe(event_id)
        self.listener.background_stop()


def estimate_region(event: Event, min_length: float = 50.) -> dict:
    """
    Estimate the region to find templates within given a triggering event.

    Parameters
    ----------
    event
        The event that triggered this function
    min_length
        Minimum length in km for diameter of event circle around the
        triggering event

    Returns
    -------
    Dictionary keyed by "latitude", "longitude" and "maxradius"

    Notes
    -----
    Uses a basic Wells and Coppersmith relation, scaled by 1.25 times.
    """
    from obspy.geodetics import kilometer2degrees
    try:
        origin = event.preferred_origin() or event.origins[0]
    except IndexError:
        Logger.error("Triggering event has no origin, not using.")
        return None

    try:
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
    except IndexError:
        Logger.warning("Triggering event has no magnitude, using minimum "
                       "length or {0}".format(min_length))
        magnitude = None
    if magnitude:
        length = 10 ** ((magnitude.mag - 5.08) / 1.16)  # Wells and Coppersmith
        # Scale up a bit - for Darfield this gave 0.6 deg, but the aftershock
        # region is more like 1.2 deg radius
        length *= 1.25
    else:
        length = min_length

    if length <= min_length:
        length = min_length
    length = kilometer2degrees(length)
    length /= 2.
    return {
        "latitude": origin.latitude, "longitude": origin.longitude,
        "maxradius": length}


if __name__ == "__main__":
    import doctest

    doctest.testmod()
