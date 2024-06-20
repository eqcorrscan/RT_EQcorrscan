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

from rt_eqcorrscan.plugins.plugin import Watcher, _PluginConfig


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


def _make_detection_from_event(event: Event) -> Detection:
    """
    Make an EQcorrscan Detection object for an Event

    Parameters
    ----------
    event

    Returns
    -------

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
    assert len(d_time_str) == 21 # YYYYmmdd_HHMMSSssssss
    d_time = UTCDateTime.strptime(d_time_str, "%Y%m%d_%H%M%S%f")

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
    template_dir

    Returns
    -------

    """
    # TODO: This should build a Party from a set of Events detected by rteqc
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
    wavebank
    length
    pre_pick

    Returns
    -------

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
            Logger.info(
                f"Getting {len(bulk)} channels of data for detection: {d.id}")
            stream += wavebank.get_waveforms_bulk(bulk)
    stream.merge()
    return stream


def main(
    config_file: str,
    detection_dir: str,
    template_dir: str,
    wavebank_dir: str,
    outdir: str,
):
    """

    Parameters
    ----------
    config_file
    detection_dir
    template_dir
    wavebank_dir
    outdir

    Returns
    -------

    """
    config = LagCalcConfig.read(config_file=config_file)
    # Initialise watcher
    watcher = Watcher(watch_pattern=f"{detection_dir}/*.xml", history=None)
    # Watch for a poison file.
    kill_watcher = Watcher(watch_pattern=f"{detection_dir}/poison",
                           history=None)

    # Loop!
    while True:
        tic = time.time()
        kill_watcher.check_for_updates()
        if len(kill_watcher):
            Logger.error("Lag-calc plugin killed")
            Logger.error(f"Found files: {kill_watcher}")
            break

        watcher.check_for_updates()
        if not len(watcher):
            Logger.debug(
                f"No new events found, sleeping for {config.sleep_interval}")
            time.sleep(config.sleep_interval)
            continue

        # We have some events to process!
        new_files = watcher.new.copy()
        new_events, event_files = Catalog(), dict()
        for f in new_files:
            event = read_events(f)
            if len(event) != 0:
                Logger.warning(f"Found {len(event)} events in {f}, "
                               f"using zeroth")
            new_events += event[0]
            event_files.update({event[0].resource_id.id: f})
            # Link event id to input filename so that we can reuse the filename
            # for output

        # Convert to party
        party = events_to_party(events=new_events, template_dir=template_dir)
        max_len_party_template = max(f.template.process_length for f in party)
        # Get streams for party
        stream = get_stream(
            party=party, wavebank=WaveBank(wavebank_dir),
            length=max_len_party_template,
            pre_pick=config.shift_len * 2,
        )
        # Run lag-calc
        lag_calced = party.lag_calc(stream=stream, pre_processed=False,
                                    **config.__dict__)
        # Write out
        for ev in lag_calced:
            fname = os.path.split(event_files[ev.resource_id.id])[-1]
            ev.write(f"{outdir}/{fname}", format="QUAKEML")

        # Mark files as processed
        watcher.processed(new_files)

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