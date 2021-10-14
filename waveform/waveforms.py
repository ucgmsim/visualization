#!/usr/bin/env python3
"""
Plots 3 components for seismograms.
"""
from argparse import ArgumentParser
from functools import partial
from glob import glob
from multiprocessing import Pool
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qcore.timeseries import BBSeis, LFSeis, HFSeis, read_ascii
from visualization.util import intersection

# extension names of 3 components (text based) if waveform data is ascii-based
components = ["090", "000", "ver"]

BINARY_FORMATS = {"BB": BBSeis, "LF": LFSeis, "HF": HFSeis}
colours = ["black", "red", "blue", "magenta", "darkgreen", "orange"]


def load_args():
    """
    Process command line arguments.
    """
    # read
    parser = ArgumentParser(description="Plots components for seismograms. ")

    parser.add_argument(
        "--waveforms",
        help="directory to text data or binary file followed by label",
        nargs=2,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--out",
        help="output folder to place plots",
        type=os.path.abspath,
        default="waveforms",
    )
    parser.add_argument(
        "--n-stations",
        help="Limit number of stations to plot (selected randomly).",
        type=int,
    )
    parser.add_argument("--stations", help="Specific stations to plot.", nargs="+")
    parser.add_argument("-v", help="verbose messages", action="store_true")
    parser.add_argument(
        "-n", "--nproc", help="number of processes to use", type=int, default=1
    )
    parser.add_argument(
        "-t", "--tmax", type=float, help="maximum duration of waveform simulation"
    )

    args = parser.parse_args()

    # validate
    for source in args.waveforms:
        if not os.path.exists(source[0]):
            parser.error(f"Cannot find waveform source: {source[0]}")

    if args.tmax is not None and args.tmax <= 0:
        parser.error("Duration -t / --tmax must be greater than 0")

    os.makedirs(args.out, exist_ok=True)

    return args


def load_location(path, verbose=False):
    """
    Return opened binary file or text directory (automatically detected).
    """
    if os.path.isfile(path):
        try:
            binary = HFSeis(path)
            if verbose:
                print(f"HF: {path}")
        except ValueError:
            # file is not an HF seis file
            binary = BBSeis(path)
            if verbose:
                print(f"BB: {path}")
    else:
        try:
            binary = LFSeis(path)
            if verbose:
                print(f"LF: {path}")
        except ValueError:
            # cannot find e3d.par... if text data
            if verbose:
                print(f"TEXT: {path}")
            return path
    return binary


def load_stations(source):
    """
    Retrieve stations for waveforms.
    """
    if type(source).__name__ != "str":
        # opened binary object
        return list(source.stations.name)

    # path to directory containing text data
    files = glob(os.path.join(source, f"*.{components[0]}"))
    stations = list(map(lambda f: os.path.basename(f)[:-4], files))
    return stations


def plot_station(output, sources, labels, tmax, verbose, station):
    """Creates a waveform plot for a specific station."""

    if verbose:
        print("Plotting station: {}...".format(station))

    timeseries = []

    for source in sources:
        if type(source).__name__ != "str":
            # opened binary object
            timeline = (
                np.arange(source.nt, dtype=np.float32) * source.dt + source.start_sec
            )
            ts_pair_dict = {}
            for j, comp in enumerate(components):
                vals = source.vel(station, comp=j)
                ts_pair_dict[comp] = (vals, timeline)
            timeseries.append(ts_pair_dict)
        else:
            # text directory
            ts_pair_dict = {}
            for comp in components:
                vals, meta = read_ascii(
                    os.path.join(source, f"{station}.{comp}"), meta=True
                )
                timeline = (
                    np.arange(meta["nt"], dtype=np.float32) * meta["dt"] + meta["sec"]
                )
                ts_pair_dict[comp] = (
                    vals,
                    timeline,
                )  # each comp may have different timeline if it is observation data
            timeseries.append(ts_pair_dict)

    x_max = -1 * np.inf
    ppgvs = np.zeros(len(components))
    npgvs = np.zeros(len(components))
    pgvs = np.zeros(len(components))

    vals_from_same_comp = {}
    for i, ts_pair_dict in enumerate(timeseries):
        for j, comp in enumerate(components):
            vals, timeline = ts_pair_dict[comp]
            x_max = max(
                x_max, max(timeline)
            )  # work out the timeline for all components and all sources of timeseries
            if j not in vals_from_same_comp:
                vals_from_same_comp[j] = np.array([])
            vals_from_same_comp[j] = np.concatenate(
                [vals_from_same_comp[j], vals]
            )  # group vals from the same components together

    if tmax is not None:
        x_max = min(tmax, x_max)

    for j in range(len(components)):
        ppgvs[j] = np.max(vals_from_same_comp[j])
        npgvs[j] = np.min(vals_from_same_comp[j])
        pgvs[j] = np.max(np.abs(vals_from_same_comp[j]))
    y_min = np.min(npgvs)
    y_max = np.max(ppgvs)

    y_diff = y_max - y_min

    scale_length = max(int(round(x_max / 25.0)) * 5, 5)

    # start plot
    f, axis = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 4), dpi=96)
    f.subplots_adjust(
        left=0.08, bottom=0.12, right=0.96, top=None, wspace=0.08, hspace=0
    )
    plt.suptitle(
        station,
        fontsize=20,
        x=0.02,
        y=0.5,
        horizontalalignment="left",
        verticalalignment="center",
    )
    plt.xlim([0, x_max])

    # subplots
    for i, ts_pair_dict in enumerate(timeseries):
        for j, comp in enumerate(components):
            ax = axis[j]
            ax.set_axis_off()
            ax.set_ylim([y_min - y_diff * 0.15, y_max])
            vals, timeline = ts_pair_dict[comp]
            assert j < len(ppgvs) and len(npgvs), f"{i} {j}"
            (line,) = ax.plot(
                timeline,
                vals * min(y_max / ppgvs[j], y_min / npgvs[j]),
                color=colours[i % len(colours)],
                linewidth=1,
            )
            if j == len(components) - 1:
                line.set_label(labels[i])
                ax.legend()

            if i == 1 and j == 0:
                # Add scale
                ax.plot(
                    [0, scale_length],
                    [y_min - y_diff * 0.1] * 2,
                    color="black",
                    linewidth=1,
                )
                ax.text(
                    0,
                    y_min - y_diff * 0.15,
                    "0",
                    size=12,
                    verticalalignment="top",
                    horizontalalignment="center",
                )
                ax.text(
                    scale_length,
                    y_min - y_diff * 0.15,
                    str(scale_length),
                    size=12,
                    verticalalignment="top",
                    horizontalalignment="center",
                )
                ax.text(
                    scale_length / 2.0,
                    y_min - y_diff * 0.225,
                    "sec",
                    size=12,
                    verticalalignment="top",
                    horizontalalignment="center",
                )

            if i == 0:
                # Add component label
                ax.set_title(components[j][1:], fontsize=18)
                ax.text(x_max, y_max, "{:.1f}".format(pgvs[j]), fontsize=14)

    plt.savefig(os.path.join(output, f"{station}.png"))
    plt.close()


if __name__ == "__main__":
    args = load_args()

    # binary class object or text folder location
    sources = [load_location(source[0], args.v) for source in args.waveforms]

    # station list
    stations = intersection([load_stations(source) for source in sources])
    if args.n_stations is not None and args.n_stations < len(stations):
        # random station selection
        stations = np.random.choice(stations, args.n_stations, replace=False)
    elif args.stations is not None:
        # specific station selection
        stations = np.intersect1d(stations, args.stations)
    assert len(stations) > 0

    p = Pool(args.nproc)
    single_station = partial(
        plot_station,
        args.out,
        sources,
        [source[1] for source in args.waveforms],
        args.tmax,
        args.v,
    )
    p.map(single_station, stations)
