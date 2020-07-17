#!/usr/bin/env python3
"""
Script to benchmark RTEQcorrscan.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""
import psutil

import numpy as np
import json

from typing import Iterable
from memory_profiler import memory_usage

from obspy import Stream, Trace, UTCDateTime

from eqcorrscan.core.match_filter import Tribe, Template
from eqcorrscan.utils.timer import Timer

from rt_eqcorrscan import RealTimeTribe
from rt_eqcorrscan import read_config


def random_string(length: int) -> str:
    """
    Generate a random string.

    Parameters
    ----------
    length
        Length of string to generate

    Returns
    -------
    A random string.
    """
    import random
    import string

    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def make_synthetic_tribe(
    n_templates: int,
    n_channels: int,
    process_length: float,
    template_length: float,
    sampling_rate: float = 100.0,
) -> Tribe:
    """
    Generate a synthetic tribe of templates

    Parameters
    ----------
    n_templates
        Number of templates to generate
    n_channels
        Number of channels for each template
    process_length
        Process length in seconds
    template_length
        Template length in seconds
    sampling_rate
        Sampling rate in Hz

    Returns
    -------
    Tribe of synthetic templates
    """
    nslc = [(random_string(2).upper(), random_string(4).upper(),
             random_string(2).upper(), random_string(3).upper())
            for _ in range(n_channels)]
    tribe = Tribe()
    for i in range(n_templates):
        st = Stream()
        for j in range(n_channels):
            tr = Trace(
                data=np.random.randn(int(sampling_rate * template_length)),
                header=dict(
                    network=nslc[j][0], station=nslc[j][1],
                    location=nslc[j][2], channel=nslc[j][3],
                    sampling_rate=sampling_rate))
            st += tr
        name = f"synth_template_{i}"
        tribe.templates.append(
            Template(
                name=name, st=st, lowcut=2.0, highcut=10.0,
                samp_rate=sampling_rate, filt_order=4, prepick=0.1,
                process_length=process_length))
    return tribe


def make_synthetic_data(
    n_channels: int,
    length: float,
    sampling_rate: float,
    nslc: list,
) -> Stream:
    """
    Make a synthetic stream to be correlated with

    Parameters
    ----------
    n_channels
        Number of channels to generate
    length
        Lenth of data in seconds
    sampling_rate
        Sampling rate in Hz
    nslc
        seed-ids to generate

    Returns
    -------
    Stream of synthetic data.
    """
    st = Stream()
    for i in range(n_channels):
        tr = Trace(
            data=np.random.randn(int(sampling_rate * length)),
            header=dict(
                network=nslc[i][0], station=nslc[i][1],
                location=nslc[i][2], channel=nslc[i][3],
                sampling_rate=sampling_rate))
        st += tr
    return st


def plot_time_memory(
    timings: dict,
    memory: dict,
    process_length: float,
    show: bool = True
):
    """
    Plot the time elapsed and memory used by the profiler.

    Parameters
    ----------
    timings
        Dictionary of {number of templates: time in seconds}
    memory
        Dictionary of {number of templates: memory use in MB}
    process_length
        Process length used in seconds
    show
        Whether to show the plot or not.

    Returns
    -------
    Figure of the plot.
    """
    import matplotlib.pyplot as plt

    n_templates = sorted(list(timings.keys()))

    fig, ax = plt.subplots()

    times = [timings[key] for key in n_templates]
    ax.plot(n_templates, times, linewidth=1.0, color="blue")
    ax.scatter(
        n_templates, times, marker="+", color="blue", label="Detection time")
    ax.set_xlabel("Number of templates")
    ax.set_ylabel("Detection time (s)")
    ax.tick_params(axis="y", labelcolor="blue")

    mem_ax = ax.twinx()
    mems = [memory[key] for key in n_templates]
    mem_ax.plot(n_templates, mems, linewidth=1.0, color="red")
    mem_ax.scatter(
        n_templates, mems, marker="+", color="red", label="Memory Use")
    mem_ax.set_ylabel("Peak memory (MB)")
    mem_ax.tick_params(axis="y", labelcolor="red")

    ax.axhline(y=process_length, linestyle="--", label="Real-time limit")
    ax.grid()
    fig.legend()

    if show:
        plt.show()
    return fig


def bench(
    n_templates: Iterable,
    n_channels: int,
    process_length: float,
    template_length: float,
    sampling_rate: float,
    reruns: int = 3,
    outfile: str = None,
):
    """
    Benchmark the matched-filter detection process for a given configuration.

    Profiles time and memory use - note that this only profiles the EQcorrscan
    Tribe.detect method, not the full real-time process. Use this to give you
    an idea of how many templates you can run within real-time.

    Parameters
    ----------
    n_templates
        Numbers of templates to profile using
    n_channels
        Number of channels of data to profile using
    process_length
        Length of data to profile using in seconds
    template_length
        Length of templates in seconds
    sampling_rate
        Sampling-rate in Hz
    reruns
        Number of times to re-run profiling - the average of these runs will
        be reported
    """
    import matplotlib.pyplot as plt

    timings, memory = dict(), dict()
    for template_set in n_templates:
        tribe = make_synthetic_tribe(
            n_channels=n_channels, n_templates=template_set,
            process_length=process_length, template_length=template_length,
            sampling_rate=sampling_rate)
        st = make_synthetic_data(
            n_channels=n_channels, length=process_length,
            sampling_rate=sampling_rate,
            nslc=list({tuple(tr.id.split('.')) for tr in tribe[0].st}))
        print(f"Running for {template_set} templates")
        time = 0.0
        mem = 0.0
        for _ in range(reruns):
            with Timer() as t:
                mem_use = memory_usage(
                    proc=(tribe.detect, (), dict(
                        stream=Stream(st), threshold=8, threshold_type="MAD",
                        trig_int=2, parallel_process=False)),
                    interval=0.05, multiprocess=True, include_children=True)
            time += t.secs
            mem += max(mem_use)
        time /= reruns
        mem /= reruns
        print(f"It took {time:.3f} s and used {mem:.3f} MB to run "
              f"{template_set} templates")
        timings.update({template_set: time})
        memory.update({template_set: mem})
    fig = plot_time_memory(
        timings, memory, process_length=process_length, show=False)
    fig.suptitle(
        f"RTEQcorrscan benchmark: {n_channels} channels of {process_length} "
        f"s\n{psutil.cpu_count()} CPU cores, max clock: "
        f"{psutil.cpu_freq().max} Hz")
    plt.show()

    # Reshape for output
    times_mems = {key: {"time": timings[key], "memory": memory[key]}
                  for key in timings.keys()}
    now = UTCDateTime.now().strftime("%Y%m%dT%H%M%S")
    outfile = outfile or f"rteqcorrscan-bench_{now}"
    with open(outfile, "w") as f:
        json.dump(fp=f, obj=times_mems)
    print(f"Written results to {outfile}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark rteqcorrscan for a range of templates")
    parser.add_argument(
        "-t", "--n-templates", type=int, nargs="+", required=True,
        help="Sequence of the number of templates to run")
    parser.add_argument(
        "-n", "--n-channels", type=int, required=True,
        help="Number of channels to run with")
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to configuration file", required=False)
    parser.add_argument(
        "--outfile", "-o", type=str, default=None,
        help="File to write results to", required=False)
    parser.add_argument(
        "--verbose", '-v', action="store_true",
        help="Print output from logging to screen")

    args = parser.parse_args()

    config = read_config(args.config)
    if args.verbose:
        config.setup_logging()
    bench(
        n_templates=args.n_templates, n_channels=args.n_channels,
        process_length=config.template.process_len,
        template_length=config.template.length,
        sampling_rate=config.template.samp_rate,
        outfile=args.outfile)


if __name__ == "__main__":
    main()
