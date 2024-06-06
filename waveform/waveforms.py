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

# files that contain the 3 components (text based)
extensions = [".090", ".000", ".ver"]
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
        help="directory to ascii data or binary file followed by label",
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

    parser.add_argument(
        "--acc",
        help="Compare acceleration (g) - By default, velocity (cm/s)",
        action="store_true",
    )

    parser.add_argument(
        "--no-amp-normalize",
        help="Do not normalize component-wise amplitude",
        action="store_true",
    )

    parser.add_argument("--axis-label", help="Add axis labels", action="store_true")

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
    files = glob(os.path.join(source, f"*{extensions[0]}"))
    stations = list(map(lambda f: os.path.basename(f)[:-4], files))
    return stations


def plot_station(
    output,
    sources,
    labels,
    tmax,
    verbose,
    acc,
    no_normalize_amp,
    axis_label,
    station,
):
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
            if acc:
                timeseries.append(np.vstack((source.acc(station).T, timeline)))
            else:
                timeseries.append(np.vstack((source.vel(station).T, timeline)))
        else:
            # text directory
            meta = read_ascii(
                os.path.join(source, f"{station}{extensions[0]}"), meta=True
            )[1]
            vals = np.array(
                [
                    read_ascii(os.path.join(source, f"{station}{ext}"))
                    for ext in extensions
                ]
            )
            timeline = (
                np.arange(meta["nt"], dtype=np.float32) * meta["dt"] + meta["sec"]
            )
            timeseries.append(np.vstack((vals, timeline)))
    x_max = max([ts[-1, -1] for ts in timeseries])
    if tmax is not None:
        x_max = min(tmax, x_max)

    all_y = np.concatenate([ts[:-1] for ts in timeseries], axis=1)
    # get axis min/max
    y_min, y_max = np.min(all_y), np.max(all_y)
    y_diff = y_max - y_min
    pgvs = np.max(np.abs(all_y), axis=1)
    ppgvs = np.max(all_y, axis=1)
    npgvs = np.min(all_y, axis=1)

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
    for i, s in enumerate(timeseries):
        for j in range(len(extensions)):
            ax = axis[j]
            if not axis_label:
                ax.set_axis_off()
            ax.set_ylim([y_min - y_diff * 0.15, y_max])

            if no_normalize_amp:
                amp_normalize_factor = 1
            else:
                amp_normalize_factor = min(y_max / ppgvs[j], y_min / npgvs[j])
            (line,) = ax.plot(
                s[len(extensions)],
                s[j] * amp_normalize_factor,
                color=colours[i % len(colours)],
                linewidth=1,
            )
            if j == 2:
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
                if acc:
                    ax.set_ylabel("Acc (g)")
                else:
                    ax.set_ylabel("Vel (cm/s)")
            if i == 0:
                # Add component label
                ax.set_title(extensions[j][1:], fontsize=18)
                ax.text(x_max, y_max, "{:.1f}".format(pgvs[j]), fontsize=14)
                ax.set_xlabel("Time (s)")

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
        args.acc,
        args.no_amp_normalize,
        args.axis_label,
    )
    p.map(single_station, stations)
