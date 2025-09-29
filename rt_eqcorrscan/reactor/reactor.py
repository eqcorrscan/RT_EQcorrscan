"""
Overarching tool for listening to and triggering from FDSN earthquakes.
"""
import logging
import time
import os
import signal
import subprocess
import platform
import pickle

from copy import deepcopy
from typing import Callable, Union, List, Tuple, Set
from multiprocessing import cpu_count

from obspy import UTCDateTime, Catalog, read_events
from obspy.core.event import Event, Magnitude

from obsplus.events import get_events
from obspy.geodetics import degrees2kilometers

from rt_eqcorrscan.database.database_manager import (
    TemplateBank, check_tribe_quality)
from rt_eqcorrscan.event_trigger.listener import _Listener
from rt_eqcorrscan.config import Config
from rt_eqcorrscan.helpers.sparse_event import get_origin_attr
from rt_eqcorrscan.reactor.scaling_relations import get_scaling_relation


Logger = logging.getLogger(__name__)


def _get_triggered_working_dir(
        triggering_event_id: str,
        exist_ok: bool = True
) -> str:
    working_dir = os.path.join(
        os.path.abspath(os.getcwd()), triggering_event_id)
    os.makedirs(working_dir, exist_ok=exist_ok)
    return working_dir


def _scan_for_events(directory: str) -> Tuple[Catalog, List[str]]:
    Logger.debug(f"Scanning {directory}")
    out_cat, files = Catalog(), []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_dir():
                c, f = _scan_for_events(entry.path)
                out_cat += c
                files.extend(f)
            elif entry.is_file():
                try:
                    event = read_events(entry.path)
                except Exception as e:
                    Logger.debug(f"Could not read {entry.path} due to {e}")
                    continue
                else:
                    out_cat += event
                    files.append(entry.path)
    Logger.debug(f"Found {len(out_cat)} events")
    return out_cat, files


class Reactor(object):
    """
    A class to listen to a client and start-up a real-time instance.

    The real-time instance will be triggered by the listener when set
    conditions are met. Appropriate templates will be extracted from the
    template database on disk and used as a Tribe for the real-time detection.

    Once triggered, the listener will keep listening, but will not trigger in
    the same region again while the real-time detector is running. The real-time
    detector has to be stopped manually, or when rate or maximum-duration
    conditions are met.

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

    Notes
    -----
    `trigger_func` must only take one argument, a catalog of events. To
    achieve this you should generate a partial function from your trigger
    function. For example, using the provided rate-and-magnitude triggering
    function:

    >>> from rt_eqcorrscan.event_trigger import magnitude_rate_trigger_func
    >>> from functools import partial
    >>> trigger_func = partial(
    ...     magnitude_rate_trigger_func, magnitude_threshold=4,
    ...     rate_threshold=20, rate_bin=0.5)
    """
    _triggered_events = []
    _running_templates = None # dict()
    # Template events ids keyed by triggering-event-id
    _running_regions = None # dict()  # Regions keyed by triggering-event-id
    max_station_distance = 1000
    n_stations = 10
    sleep_step = 15

    # Fudge factors for past sequence simulation
    _simulation = False  # Set to true to enable some extra simulation output
    _speed_up = 1
    _test_start_step = 0.0

    # Maximum processors dedicated to one detection routine.
    _max_detect_cores = 12

    # The processes that are detecting away!
    detecting_processes = None # Don't use mutable objects

    def __init__(
        self,
        client,
        listener: _Listener,
        trigger_func: Callable,
        template_database: TemplateBank,
        config: Config,
        listener_starttime: UTCDateTime = None,
    ):
        self.client = client
        self.listener = listener
        self.trigger_func = trigger_func
        self.template_database = template_database
        self.config = config
        self._listener_kwargs = dict(
            min_stations=config.database_manager.min_stations,
            template_kwargs=config.template,
            starttime=listener_starttime)
        self.notifier = config.notifier.notifier
        self._manual_trigger_dir = os.path.join(
            os.path.abspath(os.curdir), f"manual_triggers_{id(self)}")
        if not os.path.isdir(self._manual_trigger_dir):
            os.makedirs(self._manual_trigger_dir)
        Logger.info(f"To manually trigger this Reactor, write the trigger "
                    f"event quakeml to {self._manual_trigger_dir}")
        # Time-keepers
        self._run_start = None
        self._running = False
        self.up_time = 0
        signal.signal(signal.SIGINT, self._handle_interupt)
        signal.signal(signal.SIGTERM, self._handle_interupt)
        (self.detecting_processes, self._running_templates,
         self._running_regions) = dict(), dict(), dict()

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

    def run(self, max_run_length: float = None) -> None:
        """
        Run all the processes.
        
        Parameters
        ----------
        max_run_length
            Maximum run length in seconds, if None, will run indefinitely.
        """
        # Get the original keep value - we will over-write this temporarily
        listener_keep = deepcopy(self.listener.keep)
        if self._listener_kwargs.get("starttime"):
            # Set to keep at least until the starttime
            _keep_len = (UTCDateTime.now() -
                self._listener_kwargs["starttime"]) + 3600
            self.listener.keep = max(_keep_len, listener_keep)
            Logger.info(f"Setting listener keep to {self.listener.keep}")
        # Try a get and put to make sure that threads have the same
        # memory space...
        old_events = self.listener.old_events
        self.listener.old_events = old_events
        # Run the listener!
        self.listener.background_run(**self._listener_kwargs)
        self._run_start = UTCDateTime.now()
        # Query the catalog in the listener every so often and check
        self._running = True
        first_iteration = True
        previous_old_events, working_cat = [], Catalog()  # Initialise state
        while self._running:
            old_events = deepcopy(self.listener.old_events)
            Logger.info(f"Old events from the listener has {len(old_events)} "
                        f"events")
            # Clear out stale events from working_cat
            event_ids = [_[0] for _ in old_events]
            working_cat.events = [ev for ev in working_cat
                                  if ev.resource_id in event_ids]
            new_old_events = [ev for ev in old_events
                              if ev not in previous_old_events]
            # Get these locally to avoid accessing shared memory multiple times
            if len(new_old_events) > 0:
                working_ids = [_[0] for _ in new_old_events]
                Logger.info(f"Getting event info from database for "
                            f"{', '.join(working_ids)}")
                try:
                    new_working_cat = self.template_database.get_events(
                        eventid=working_ids, _allow_update=False)
                except Exception as e:
                    Logger.error(f"Could not get template events "
                                 f"from database due to {e}")
                    new_working_cat = []
                if len(working_ids) and not len(new_working_cat):
                    Logger.warning("Error getting events from database, "
                                   "getting individually")
                    for working_id in working_ids:
                        try:
                            working_cat += self.template_database.get_events(
                                eventid=working_id, _allow_update=False)
                        except Exception as e:
                            Logger.error(f"Could not read {working_id} "
                                         f"due to {e}")
                            continue
                else:
                    working_cat += new_working_cat
                Logger.info("Currently analysing a catalog of "
                            "{0} events".format(len(working_cat)))
                self.process_new_events(new_events=working_cat)
                Logger.debug("Finished processing new events")
            # Check for manual triggers
            manual_triggers = self.get_manual_triggers(clean=True)
            if len(manual_triggers) > 0:
                self.trigger(trigger_events=manual_triggers)
            # Check for updates to triggered events
            self.trigger_updates()
            # Process old events
            previous_old_events = old_events  # Overload
            self.set_up_time(UTCDateTime.now())
            Logger.debug(f"Up-time: {self.up_time}")
            if max_run_length is not None and self.up_time >= max_run_length:
                Logger.critical("Max run length reached. Stopping.")
                self.stop()
                break
            Logger.debug(f"Sleeping for {self.sleep_step} seconds")
            time.sleep(self.sleep_step)
            Logger.debug("Waking up")
            if first_iteration and len(working_cat):
                # Revert keep to original value
                Logger.info(f"Reverting keep to {listener_keep}")
                self.listener.keep = listener_keep
                first_iteration = False

    def trigger_updates(self):
        """
        Check for updates to trigger events and increase search regions if
        needed
        """
        for trigger_event in self._triggered_events:
            trigger_event_id = trigger_event.resource_id.id.split('/')[-1]
            Logger.info(f"Checking for changes in trigger {trigger_event_id}")
            new_trigger_event = self.listener.check_event(trigger_event_id)
            if (new_trigger_event.preferred_magnitude().mag <=
                    trigger_event.preferred_magnitude().mag):
                continue
            Logger.info(
                f"Trigger event {trigger_event_id} magnitude "
                f"has increased from "
                f"{trigger_event.preferred_magnitude().mag} to "
                f"{new_trigger_event.preferred_magnitude().mag}")
            new_region = self._trigger_region(new_trigger_event)
            # Get the new templates and put them in the right place
            new_tribe_files, new_event_ids = self._get_tribe_files(
                region=new_region)
            self._add_templates_to_running_trigger(
                added_ids=new_event_ids,
                triggering_event_id=trigger_event_id)
            # Alter the lookup region for this trigger so that new events
            # are put in as they come in.
            self._running_regions.update({trigger_event_id: new_region})
        return

    def check_running_tribes(self) -> None:
        """ 
        Check on the state of running tribes.

        Because this process starts subprocesses, they need to be stopped
        by this process if they "complete".  This is recorded by the
        real-time-tribe by writing a `.stopfile`
        """
        for trigger_event in self._triggered_events:
            trigger_event_id = trigger_event.resource_id.id.split('/')[-1]
            working_dir = _get_triggered_working_dir(
                trigger_event_id, exist_ok=True)
            if os.path.isfile(f"{working_dir}/.stopfile"):
                Logger.info(f"Found stopfile for {trigger_event_id}")
                self.stop_tribe(trigger_event_id)

    def get_manual_triggers(self, clean=True) -> Catalog:
        manual_triggers, event_files = _scan_for_events(
            self._manual_trigger_dir)
        # Remove events that have been read in
        if clean:
            for event_file in event_files:
                Logger.debug(f"Removing trigger file {event_file}")
                os.remove(event_file)
        return manual_triggers

    def process_new_events(
            self,
            new_events: Union[Catalog, List[Event]]
        ) -> None:
        """
        Process any new events in the system.

        Check if new events should be in one of the already running
        tribes and add them. Check all other events for possible triggers and
        spin-up a detector instance for triggers.

        Parameters
        ----------
        new_events
            Catalog of new-events to be assessed.
        """
        # Convert to catalog
        if isinstance(new_events, list):
            new_events = Catalog(new_events)
        for triggering_event_id, tribe_region in self._running_regions.items():
            try:
                add_events = get_events(new_events, **tribe_region)
            except Exception:
                # This only occurs when there are no events in the region
                # and is fixed by PR #177 on Obsplus.
                add_events = Catalog()
            # Don't trigger on events now running in another tribe.
            new_events.events = [e for e in new_events
                                 if e not in add_events]
            # TODO: Implement region growth based on new events added.
            added_ids = {e.resource_id.id for e in add_events}.difference(
                self.running_template_ids)
            if added_ids:
                self._add_templates_to_running_trigger(
                    added_ids=added_ids,
                    triggering_event_id=triggering_event_id)
        trigger_events = self.trigger_func(new_events)
        # Check for manually added trigger events
        trigger_events += self.get_manual_triggers()
        self.trigger(trigger_events=trigger_events)

    def _add_templates_to_running_trigger(
        self, added_ids: set, triggering_event_id: str) -> None:
        """
        Add templates to an already running RTTribe.

        Parameters
        ----------
        added_ids
            Event ids of events to be added
        triggering_event_id
            ID of triggering event for which an RTTribe is running.
        """
        tribe = self.template_database.get_templates(
            eventid=added_ids)
        tribe = check_tribe_quality(
            tribe,
            min_stations=self.config.rt_match_filter.min_stations,
            **self.config.template)
        if len(tribe) > 0:
            Logger.info(f"Adding {len(tribe)} events to {triggering_event_id}")
            template_dir = os.path.join(
                _get_triggered_working_dir(triggering_event_id),
                "new_templates")
            if not os.path.isdir(template_dir):
                os.makedirs(template_dir)
            for template in tribe:
                template.write(filename=os.path.join(
                    template_dir, template.name))
            Logger.info(f"Written new templates to {template_dir}")
            self._running_templates[triggering_event_id].update(
                added_ids)
        return

    def trigger(self, trigger_events: Catalog) -> None:
        # Sanitize trigger-events - make sure that multiple events that would otherwise
        # run together do not all trigger - sort by magnitude
        for trigger_event in trigger_events:
            # Make sure they all have a magnitude
            if len(trigger_event.magnitudes) == 0 or trigger_event.magnitudes[0] is None:
                trigger_event.magnitudes = [Magnitude(mag=-999)]
        trigger_events.events.sort(
            key=lambda e: (e.preferred_magnitude() or e.magnitudes[0]).mag,
            reverse=True)
        for trigger_event in trigger_events:
            if trigger_event in self._triggered_events:
                continue
            if trigger_event.resource_id.id in self.running_template_ids:
                Logger.info(
                    f"Not spinning up {trigger_event}: it is already running")
                continue
            Logger.warning(
                "Listener triggered by event {0}".format(trigger_event))
            # Send this as an email
            self.notifier.notify(
                content=f"Listener triggered on {platform.node()} "
                        f"by event {trigger_event}")
            if len(self._running_regions) >= self.available_cores:
                Logger.error("No more available processors")
                continue
            self._triggered_events.append(trigger_event)
            self.spin_up(trigger_event)

    def _trigger_region(self, triggering_event: Event) -> Union[dict, None]:
        """ Work out lookup starttime. Included for testing purposes. """
        if get_origin_attr(triggering_event, "depth") > self.config.reactor.scaling_depth_switch:
            scaling_relation = self.config.reactor.scaling_relation_deep
        else:
            scaling_relation = self.config.reactor.scaling_relation_shallow
        Logger.info(f"Using {scaling_relation} to determine search region")
        region = estimate_region(
            triggering_event,
            multiplier=self.config.reactor.scaling_multiplier or 1.0,
            min_radius=self.config.reactor.minimum_lookup_radius or 50.0,
            scaling_relation=scaling_relation)
        if region is None:
            return None

        # Allow lookups prior to trigger-time
        if isinstance(
                self.config.database_manager.lookup_starttime, (float, int)):
            lookup_starttime = (
                get_origin_attr(triggering_event, "time") -
                abs(self.config.database_manager.lookup_starttime))
        else:
            lookup_starttime = (
                self.config.database_manager.lookup_starttime or
                UTCDateTime(0))
        region.update({"starttime": lookup_starttime})
        return region

    def _get_tribe_files(self, region: dict = None) -> Tuple[List[str], Set[str]]:
        """
        Get the tribe files for a select region. Skips already running templates

        Parameters
        ----------
        region
            Region kwargs for .get_events

        Returns
        -------
        List of paths of templates within region.
        """
        Logger.info("Getting templates within {0}".format(region))
        df = self.template_database.get_event_summary(**region)
        event_ids = {e for e in df["event_id"]}
        event_ids = event_ids.difference(self.running_template_ids)
        Logger.debug(f"event-ids in region: {event_ids}")
        # Write file of event id's
        if len(event_ids) == 0:
            Logger.warning(f"Found no events in region: {region} - no detection to run.")
            return [], set()
        tribe_files = self.template_database.get_template_paths(
            eventid=event_ids)
        return tribe_files, event_ids

    def spin_up(self, triggering_event: Event) -> None:
        """
        Run the reactors response function as a subprocess.

        Parameters
        ----------
        triggering_event
            Event that triggered this run - needs to have at-least an origin.
        """
        triggering_event_id = triggering_event.resource_id.id.split('/')[-1]
        region = self._trigger_region(triggering_event)
        tribe_files, event_ids = self._get_tribe_files(region=region)
        if len(tribe_files) == 0:
            return
        working_dir = _get_triggered_working_dir(
            triggering_event_id, exist_ok=True)
        # tribe.write(os.path.join(working_dir, "tribe.tgz"))
        tribe_dir = os.path.join(working_dir, "tribe")
        os.makedirs(tribe_dir, exist_ok=True)
        for tf in tribe_files:
            os.symlink(tf, os.path.join(tribe_dir, os.path.basename(tf)))
        # Add search radius for things that need it
        if self.config.plugins["plotter"]:
            # Roughly convert to km
            self.config.plugins["plotter"].search_radius = degrees2kilometers(
                region["maxradius"])
        if self.config.plugins["output"]:
            # Roughly convert to km
            self.config.plugins["output"].cluster_threshold = degrees2kilometers(
                region["maxradius"])

        self.config.write(
            os.path.join(working_dir, 'rt_eqcorrscan_config.yml'))
        triggering_event.write(
            os.path.join(working_dir, 'triggering_event.xml'), format="QUAKEML")
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "spin_up.py")
        _call = ["python", script_path, "-w", working_dir,
                 "-n", str(min(self.available_cores, self._max_detect_cores)),
                 "-s", str(self._speed_up),
                 "-o", str(self._test_start_step)]
        if self._simulation:
            _call.append("--simulation")
        Logger.info("Running `{call}`".format(call=" ".join(_call)))
        proc = subprocess.Popen(_call)
        self.detecting_processes.update({triggering_event_id: proc})
        self._running_regions.update({triggering_event_id: region})
        self._running_templates.update(
            {triggering_event_id: event_ids})
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

    def _handle_interupt(self, signum, frame) -> None:
        Logger.critical(f"Received signal: {signum}")
        self.notifier.notify(
            content=f"CRITICAL: Stopping reactor on {platform.node()} "
                    f"after interrupt: {signum}")
        self.stop()

    def stop(self, raise_exit: bool=True) -> None:
        """
        Stop all the processes.

        Parameters
        ----------
        raise_exit:
            Whether to exit the Python system or not.
        """
        Logger.critical("Stopping the reactor")
        self._running = False
        Logger.critical("Stopping the listener")
        self.listener.background_stop()
        Logger.critical("Stopped the listener")
        triggering_event_ids = list(self.detecting_processes.keys())
        for event_id in triggering_event_ids:
            Logger.critical(f"Stopping tribe running for {event_id}")
            self.stop_tribe(event_id)
        Logger.critical("Stopped")
        if raise_exit:
            raise SystemExit


def estimate_region(
    event: Event,
    min_radius: float = 50.,
    scaling_relation: Union[str, Callable] = 'default',
    multiplier: float = 1.25,
) -> dict:
    """
    Estimate the region to find templates within given a triggering event.

    Parameters
    ----------
    event
        The event that triggered this function
    min_radius
        Minimum length in km for diameter of event circle around the
        triggering event
    scaling_relation
        Name of registered scaling-relationship or Callable that takes only
        the earthquake magnitude as an argument and returns length in km
    multiplier
        Fudge factor to scale the scaling relation up by a constant.

    Returns
    -------
    Dictionary keyed by "latitude", "longitude" and "maxradius"

    Notes
    -----
    The `scaling_relation` * `multiplier` defines the `maxradius` of the region
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
                       "length or {0}".format(min_radius))
        magnitude = None

    if magnitude:
        if not callable(scaling_relation):
            try:
                scaling_relation = get_scaling_relation(scaling_relation)
            except Exception as e:
                Logger.exception(
                    f"Could not get {scaling_relation} scaling due to {e}. "
                    f"Reverting to default")
                scaling_relation = get_scaling_relation("default")
        length = scaling_relation(magnitude.mag)
        length *= multiplier
    else:
        length = min_radius * 2

    # Convert from length to radius
    radius = length / 2.

    if radius <= min_radius:
        radius = min_radius
    radius = kilometer2degrees(radius)

    return {
        "latitude": origin.latitude, "longitude": origin.longitude,
        "maxradius": radius}


if __name__ == "__main__":
    import doctest

    doctest.testmod()
