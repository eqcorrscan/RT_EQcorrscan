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

from obpsy import read_events, Catalog, Stream
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
    template_names = {c.split(": ")[-1] for ev in events for c in ev.comments
                      if c.startswith("Template:")}
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

    # Make families
    party = Party()
    for template in templates:
        fam = Family(template=template)
        fam.detections =
    return party


def get_stream(party: Party, wavebank_dir: str) -> Stream:
    """

    Parameters
    ----------
    party
    wavebank_dir

    Returns
    -------

    """
    # TODO: This should geta a gappy stream covering all the detections
    bank = WaveBank(wavebank_dir)
    raise NotImplementedError()


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
    kill_watcher = Watcher(watch_pattern=f"{detection_dir}/poison", history=None)

    # Loop!
    while True:
        tic = time.time()
        kill_watcher.check_for_updates()
        if len(kill_watcher):
            Logger.error("Lag-calc plugin killed")

        watcher.check_for_updates()
        if not len(watcher):
            time.sleep(config.sleep_interval)
            continue

        # We have some events to process!
        new_files = watcher.new.copy()
        new_events = Catalog([read_events(f) for f in new_files])
        # Link event id to input filename so that we can reuse the filename
        # for output
        event_files = {ev.resource_id.id: fname
                       for ev, fname in zip(new_events, new_files)}
        # Convert to party
        party = events_to_party(events=new_events, template_dir=template_dir)
        # Get streams for party
        stream = get_stream(party=party, wavebank_dir=wavebank_dir)
        # Run lag-calc
        lag_calced = party.lag_calc(st=stream, **config.__dict__())
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()