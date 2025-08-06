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
import logging
import os
import pickle

from typing import Iterable, List

from obspy import (
    read_events, Catalog, Stream, UTCDateTime, read_inventory, Inventory)
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.core.event import Event, Arrival

from eqcorrscan import Party, Family, Detection
from eqcorrscan.utils.mag_calc import amp_pick_event

from rt_eqcorrscan.config.config import _PluginConfig
from rt_eqcorrscan.plugins.plugin import (
    PLUGIN_CONFIG_MAPPER, _Plugin)
from rt_eqcorrscan.plugins.waveform_access import InMemoryWaveBank


Logger = logging.getLogger(__name__)


class PickerConfig(_PluginConfig):
    """
    Configuration holder for the picker plugin
    """
    defaults = _PluginConfig.defaults.copy()
    defaults.update({
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
        "winlen": 0.5,
        "ps_multiplier": 0.15,
        "station_file": "stations.xml",
    })
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PLUGIN_CONFIG_MAPPER.update({"lag_calc": PickerConfig})


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
    rid = event.resource_id.id.split('/')[-1]
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
    in_memory_wavebank: InMemoryWaveBank,
    length: float,
    pre_pick: float,
) -> Stream:
    """

    Parameters
    ----------
    party
        Party of detections to get streams for
    in_memory_wavebank
        On-disk in-memory wavebank to get waveforms from
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
            stream += in_memory_wavebank.get_event_waveforms(
                event=ev, pre_pick=pre_pick, length=length)
    stream.merge()
    return stream


def _insert_arrivals(
    event: Event,
    inventory: Inventory,
) -> Event:
    """
    Insert arrivals for picks into origin.

    Parameters
    ----------
    event
        Event to add arrivals to
    inventory
        Inventory with station locations picked

    Returns
    -------
    Event with arrivals added
    """
    ev = event.copy()  # Don't work on the users event
    try:
        ori = ev.preferred_origin() or ev.origins[-1]
    except IndexError:
        Logger.debug("No origin found, cannot add arrivals")
        return event
    for pick in ev.picks:
        sid = pick.waveform_id.get_seed_string()
        n, s, l, c = sid.split('.')
        station = inventory.select(network=n, station=s, location=l, channel=c)
        if len(station) == 0:
            Logger.debug(f"No inventory for {n}.{l}.{s}.{c}")
            continue
        lat, lon = station[0][0].latitude, station[0][0].longitude
        dist, _, _ = gps2dist_azimuth(lat, lon, ori.latitude, ori.longitude)
        dist /= 1000.0
        arr = Arrival(distance=kilometers2degrees(dist))
        arr.pick_id = pick.resource_id
        ori.arrivals.append(arr)
    return ev


class Picker(_Plugin):
    def __init__(self, config_file: str, name: str = "Picker"):
        super().__init__(config_file=config_file, name=name)
        self.in_memory_wavebank = InMemoryWaveBank(self.config.wavebank_dir)
        self.in_memory_wavebank.get_data_availability()

    def _read_config(self, config_file: str):
        return PickerConfig.read(config_file=config_file)

    def core(self, new_files: Iterable, cleanup: bool = True) -> List:
        internal_config = self.config.copy()
        detection_dir = internal_config.pop("in_dir")
        template_dir = internal_config.pop("template_dir")
        outdir = internal_config.pop("out_dir")
        inv = read_inventory(self.config.station_file)

        processed_files = []
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
            rid = event[0].resource_id.id.split('/')[-1]
            event_files.update({rid: f})
            Logger.info(f"Linked event {rid} from file {f}")
            # Link event id to input filename so that we can reuse the filename
            # for output

        # Convert to party - detection ids are named by the events resource ids
        party = events_to_party(events=new_events, template_dir=template_dir)

        # Loop over detections - party.lag-calc works okay for longer
        # process-lengths, but not so well for these short ones
        lag_calced = Catalog()
        for family in party:
            Logger.info(f"Setting up picker for Family: {family.template.name}")
            for detection in family:
                Logger.info(f"Setting up picker for Detection: {detection.id}")
                d_party = Party(
                    [Family(family.template, [detection])])
                Logger.info(f"Getting stream from {self.in_memory_wavebank}")
                # Ideally this would be an integer multiple of process-len
                desired_stream_length = family.template.process_length * 3
                stream = get_stream(
                    d_party, in_memory_wavebank=self.in_memory_wavebank,
                    length=desired_stream_length,
                    pre_pick=max(internal_config.shift_len * 2,
                                 family.template.process_length / 2))
                stream = stream.merge(method=1)
                # Get an excess of data to cope with missing "future" data
                Logger.info(f"Have stream: \n{stream}")
                # lag-calc will throw away the end of data if that is not
                # long enough - we need the data length to be a multiple of
                # process-len
                for tr in stream:
                    tr_length = tr.stats.endtime - tr.stats.starttime
                    process_len_multiples = (
                        tr_length // family.template.process_length)
                    Logger.debug(f"Trimming {tr.id} to "
                                 f"{process_len_multiples} process lengths")
                    tr.trim(
                        starttime=tr.stats.endtime - (
                                process_len_multiples *
                                family.template.process_length),
                        endtime=tr.stats.endtime)
                if len(stream):
                    min_len = min([
                        tr.stats.endtime - tr.stats.starttime
                        for tr in stream])
                else:
                    min_len = 0.0
                if min_len < family.template.process_length:
                    Logger.info(
                        f"Insufficient data ({min_len}s) for {detection.id} "
                        f"which requires {family.template.process_length}s, "
                        f"waiting.")
                    continue
                # Run lag-calc
                Logger.info(f"Running lag-calc for {detection.id}")
                event_back = None
                try:
                    event_back = d_party.lag_calc(
                        stream=stream, pre_processed=False, ignore_length=True,
                        **internal_config.__dict__)
                except Exception as e:
                    Logger.error(
                        f"Could not run lag-calc for {detection.id} due to {e}",
                        exc_info=True)
                if event_back and len(event_back):
                    # Merge the event info
                    event = detection.event
                    assert len(event_back) == 1, f"Multiple events: {event_back}"
                    event.picks = event_back[0].picks
                    # Now try and pick amplitudes for magnitudes
                    # We need arrivals for this
                    event = _insert_arrivals(event=event, inventory=inv)
                    try:
                        event = amp_pick_event(
                            event=event, st=stream, inventory=inv,
                            chans=["1", "2", "N", "E"], iaspei_standard=False,
                            var_wintype=True, winlen=internal_config.winlen,
                            ps_multiplier=internal_config.ps_multiplier,
                            win_from_p=True)
                        # Change here requires this PR:
                        # https://github.com/eqcorrscan/EQcorrscan/pull/572
                    except Exception as e:
                        Logger.error(f"Could not pick amplitudes for "
                                     f"{detection.id} due to {e}",
                                     exc_info=True)
                    event.resource_id = detection.id
                    lag_calced += event
                    processed_files.append(event_files[detection.id])
                else:  # Keep all the events even if we don't re-pick them
                    Logger.info(
                        f"Unsuccessful lag-calc for "
                        f"{event_files[detection.id]}, will retry")
                    # event = detection.event
                    # event.picks = []
                    # lag_calced += event
        # Write out
        for ev in lag_calced:
            fname = event_files[ev.resource_id.id.split('/')[-1]].split(
                detection_dir)[-1]
            fname = fname.lstrip(os.path.sep)  # Strip pathsep if it is there
            outpath = os.path.join(outdir, fname)
            Logger.info(f"Writing out to {outpath}")
            if not os.path.isdir(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            ev.write(f"{outpath}", format="QUAKEML")

        return processed_files


if __name__ == "__main__":
    import doctest

    doctest.testmod()