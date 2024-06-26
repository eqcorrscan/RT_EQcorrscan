"""
Run EQcorrscan's lag-calc as a plugin

Steps:
1. Watch detection directory (sleep between checks)
2. Read in new detections (events)
    - Build detections for events
3. Get relevant waveforms for detections
4. Get relevant waveforms for templates
5. Run lag-calc
6. Output events to output directory
7. Repeat
"""
import time
import logging
import os
import pickle

from typing import Union

from obspy import read_events, Catalog, Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.event import Event
from obsplus import WaveBank

from eqcorrscan import Party, Family, Detection

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    Watcher, PLUGIN_CONFIG_MAPPER)


Logger = logging.getLogger(__name__)


class LagCalcConfig(_PluginConfig):
    """
    Configuration holder for the lag-calc plugin
    """
    defaults = {
        "shift_len": 0.2,
        "min_cc": 0.4,
        "min_cc_from_mean_cc_factor": None,
        "all_vert": False,
        "all_horiz": False,
        "horizontal_chans": ['E', 'N', '1', '2'],
        "vertical_chans": ['Z'],
        "cores": 1,
        "interpolate": False,
        "plot": False,
        "plotdir": None,
        "export_cc": False,
        "cc_dir": None,
        "sleep_interval": 600,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"lag_calc": LagCalcConfig})


def _make_detection_from_event(event: Event) -> Detection:
    """
    Make an EQcorrscan Detection object for an Event

    Parameters
    ----------
    event
        Event (detected by EQcorrscan) to make a Detection for

    Returns
    -------
    Detection for the event provided.
    """
    # Info in comments is template name, threshold, detect_val and channels
    t_name, threshold, detect_val, channels = None, None, None, None
    for comment in event.comments:
        if comment.text.lower().startswith("template: "):
            t_name = comment.text.split(": ")[-1]
        elif comment.text.lower().startswith("threshold"):
            threshold = float(comment.text.split("=")[-1])
        elif comment.text.lower().startswith("detect_val"):
            detect_val = float(comment.text.split("=")[-1])
        elif comment.text.lower().startswith("channels used:"):
            channel_string = comment.text.lstrip("channels used: ")
            channel_string = channel_string.lstrip("(")
            channel_string = channel_string.rstrip(")")
            channel_string = channel_string.split(") (")
            # Remove quotation marks
            channel_string = [cs.replace("'", "") for cs in channel_string]
            # Make into list of tuples
            channels = [tuple(cs.split(", ")) for cs in channel_string]

    for thing in [t_name, threshold, detect_val, channels]:
        if thing is None:
            raise NotImplementedError(
                f"Event is not an EQcorrscan detected event:\n{event}")

    # Get the detection ID from the event resource id
    rid = event.resource_id.id
    # Get the detect time from the rid
    d_time_str = rid.split(f"{t_name}_")[-1]
    if len(d_time_str) == 21:
        time_format = "%Y%m%d_%H%M%S%f"
    elif len(d_time_str) == 22:
        time_format = "%Y%m%dT%H%M%S.%f"
    else:
        raise NotImplementedError(f"Unknown time format for {d_time_str}")
    d_time = UTCDateTime.strptime(d_time_str, time_format)

    d = Detection(
        template_name=t_name,
        detect_time=d_time,
        no_chans=len(channels),
        detect_val=detect_val,
        threshold=threshold,
        typeofdet="corr",
        threshold_type="Unknown",
        threshold_input=None,
        chans=channels,
        event=event,
        id=rid,
        strict=False,
    )
    return d


def events_to_party(events: Catalog, template_dir: str) -> Party:
    """

    Parameters
    ----------
    events
        Catalog of events to convert to Detections
    template_dir
        Directory of pickled templates used for detections.

    Returns
    -------
    Party for the Catalog provided
    """
    template_names = {
        c.text.split(": ")[-1] for ev in events for c in ev.comments
        if c.text.startswith("Template:")}
    # Find the relevant template files
    templates = []
    for tname in template_names:
        tfile = f"{template_dir}/{tname}.pkl"
        if not os.path.isfile(tfile):
            Logger.warning(f"Could not file template file {tfile}")
            continue
        with open(tfile, "rb") as f:
            t = pickle.load(f)
        templates.append(t)

    # Make detections
    detections = [_make_detection_from_event(event) for event in events]

    # Make families
    party = Party()
    for template in templates:
        fam = Family(template=template)
        fam.detections = [d for d in detections
                          if d.template_name == template.name]
        party += fam
    return party


def get_stream(
    party: Party,
    wavebank: Union[WaveBank, Client],
    length: float,
    pre_pick: float,
) -> Stream:
    """

    Parameters
    ----------
    party
        Party of detections to get streams for
    wavebank
        Wavebank or Client to get waveforms from
    length
        Length in seconds to get data for each event
    pre_pick
        Time in seconds to get data for before the expected pick-time

    Returns
    -------
    Single gappy-stream for all events in Party.
    """
    stream = Stream()
    for f in party:
        for d in f:
            ev = d.event or d._calculate_event(template=f.template)
            bulk = []
            for p in ev.picks:
                bulk.append((
                    p.waveform_id.network_code or "*",
                    p.waveform_id.station_code,
                    p.waveform_id.location_code or "*",
                    p.waveform_id.channel_code,
                    p.time - pre_pick,
                    p.time + (length - pre_pick)))
            Logger.debug(
                f"Getting {len(bulk)} channels of data for detection: {d.id}")
            Logger.debug(bulk)
            stream += wavebank.get_waveforms_bulk(bulk)
    stream.merge()
    return stream


def main(
    config_file: str,
) -> None:
    """

    Parameters
    ----------
    config_file
        Path to configuration file for lag-calc runner.
    """
    config = LagCalcConfig.read(config_file=config_file)
    detection_dir = config.pop("in_dir")
    template_dir = config.pop("template_dir")
    wavebank_dir = config.pop("wavebank_dir")
    outdir = config.pop("out_dir")
    # Initialise watcher
    watcher = Watcher(
        top_directory=detection_dir, watch_pattern="*.xml", history=None)
    # Watch for a poison file.
    kill_watcher = Watcher(
        top_directory=outdir, watch_pattern="poison", history=None)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Loop!
    while True:
        tic = time.time()
        kill_watcher.check_for_updates()
        if len(kill_watcher):
            Logger.critical("Lag-calc plugin killed")
            Logger.critical(f"Found files: {kill_watcher}")
            break

        watcher.check_for_updates()
        if not len(watcher):
            Logger.debug(
                f"No new events found, sleeping for {config.sleep_interval}")
            time.sleep(config.sleep_interval)
            continue

        # We have some events to process!
        new_files, processed_files = watcher.new.copy(), []
        new_events, event_files = Catalog(), dict()
        for f in new_files:
            event = read_events(f)
            if len(event) > 1:
                Logger.warning(f"Found {len(event)} events in {f}, "
                               f"using zeroth")
            elif len(event) == 0:
                Logger.warning(f"Found no events in {f}, skipping")
                processed_files.append(f)
                continue
            new_events += event[0]
            event_files.update({event[0].resource_id.id: f})
            # Link event id to input filename so that we can reuse the filename
            # for output

        # Convert to party
        party = events_to_party(events=new_events, template_dir=template_dir)

        # Access the wavebank (read in the table) once at the start of the
        # loop over events - we need to re-access the bank every time loop to
        # get the new data. We also have to cope with other python processes
        # potentially writing to the table when we try to open it.
        _retrires = 0
        while True:
            try:
                # We start a new access to the wavebank every time
                # incase it has updated
                wavebank = WaveBank(wavebank_dir)
            except Exception as e:
                Logger.warning(
                    f"Could not access the wavebank due to {e}. "
                    f"Retrying in 1s")
                time.sleep(1)
                _retrires += 1
            else:
                # We got the wavebank!
                break

        # Loop over detections - party.lag-calc works okay for longer
        # process-lengths, but not so well for these short ones
        lag_calced = Catalog()
        for family in party:
            Logger.info(f"Setting up lag-calc for {family.template.name}")
            for detection in family:
                Logger.debug(f"Setting up lag-calc for {detection.id}")
                d_party = Party(
                    [Family(family.template, [detection])])
                stream = get_stream(
                    d_party, wavebank=wavebank,
                    length=family.template.process_length * 2.1,
                    pre_pick=min(config.shift_len * 2,
                                 family.template.process_length))
                # Get an excess of data to cope with missing "future" data
                Logger.debug(f"Have stream: \n{stream}")
                if len(stream):
                    min_len = min([
                        tr.stats.endtime - tr.stats.starttime
                        for tr in stream])
                else:
                    min_len = 0.0
                if min_len < family.template.process_length:
                    Logger.debug(
                        f"Insufficient data for {detection.id}, waiting.")
                    continue
                # Run lag-calc
                Logger.info(f"Running lag-calc for {detection.id}")
                event_back = None
                try:
                    event_back = d_party.lag_calc(
                        stream=stream, pre_processed=False, ignore_length=True,
                        **config.__dict__)
                except Exception as e:
                    Logger.error(
                        f"Could not run lag-calc for {detection} due to {e}")
                if event_back and len(event_back):
                    # Merge the event info
                    event = detection.event
                    assert len(event_back) == 1, f"Multiple events: {event_back}"
                    event.picks = event_back[0].picks
                    lag_calced += event
                    processed_files.append(
                        event_files[detection.event.resource_id.id])
                else:  # Keep all the events even if we don't re-pick them
                    Logger.info(
                        f"Unsuccessful lag-calc for "
                        f"{event_files[detection.event.resource_id.id]}, will "
                        f"retry")
                    # event = detection.event
                    # event.picks = []
                    # lag_calced += event
        # Write out
        for ev in lag_calced:
            fname = event_files[ev.resource_id.id].split(detection_dir)[-1]
            fname = fname.lstrip(os.path.sep)  # Strip pathsep if it is there
            outpath = os.path.join(outdir, fname)
            Logger.info(f"Writing out to {outpath}")
            if not os.path.isdir(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            ev.write(f"{outpath}", format="QUAKEML")

        # Mark files as processed
        watcher.processed(processed_files)

        # Check for poison again before sleeping
        kill_watcher.check_for_updates()
        if len(kill_watcher):
            Logger.error("Lag-calc plugin killed")
        # Sleep and repeat
        toc = time.time()
        elapsed = toc - tic
        Logger.info(f"Lag-calc loop took {elapsed:.2f} s")
        if elapsed < config.sleep_interval:
            time.sleep(config.sleep_interval - elapsed)
        continue
    return


if __name__ == "__main__":
    import doctest

    doctest.testmod()