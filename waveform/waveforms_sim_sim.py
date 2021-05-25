#!/usr/bin/env python3
"""
Plots 3 components for seismograms.
"""
from typing import Union

from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qcore.timeseries import BBSeis, LFSeis, HFSeis

# files that contain the 3 components (text based)
# must be in same order as binary results (x, y, z)
extensions = [".000", ".090", ".ver"]
BINARY_FORMATS = {"BB": BBSeis, "LF": LFSeis, "HF": HFSeis}


def load_args():
    """
    Process command line arguments.
    """
    # read
    parser = ArgumentParser(
        description="Plots components for seismograms. "
    )

    parser.add_argument(
        "waveforms", help="directory to text data or binary file", nargs="+", type=os.path.abspath
    )
    parser.add_argument(
        "--out", help="output folder to place plots", type=os.path.abspath, default="waveforms",
    )
    parser.add_argument(
        "--n-stations",
        default=-1,
        help="Number of stations, selected randomly, to plot. Default is all (-1)",
        type=int,
    )
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
        if not os.path.exists(source):
            parser.error(f"Cannot find waveform source: {source}")

    if args.tmax is not None and args.tmax <= 0:
        parser.error("Duration -t / --tmax must be greater than 0")

    os.makedirs(args.out, exist_ok=True)

    return args


def load_location(path, v=False):
    """
    Return opened binary file or text directory.
    """
    if os.path.isfile(path):
        try:
            binary = HFSeis(path)
            if v:
                print(f"HF: {path}")
        except ValueError:
            # file is not an HF seis file
            binary = BBSeis(path)
            if v:
                print(f"BB: {path}")
    else:
        try:
            binary = LFSeis(path)
            if v:
                print(f"LF: {path}")
        except ValueError:
            # cannot find e3d.par... if text data
            if v:
                print(f"TEXT: {path}")
            return path
    return binary


def load_stations(location):
    """
    Retrieve stations for waveforms.
    """
    if type(location).__name__ != "str":
        # opened binary object
        return list(location.stations.name)

    # path to directory containing text data
    files = glob(os.path.join(location, f"*{extensions[0]}"))
    stations = list(map(lambda f: os.path.basename(f)[:-4], files))
    return stations


def load_station_inter(
    benchmark_binary: Union[BBSeis, LFSeis, HFSeis],
    comparison_binary: Union[BBSeis, LFSeis, HFSeis],
    verbose=False,
):
    """
    Determine stations available for plotting.
    returns numpy array of intersecting station names
    """
    # stations available in sim
    benchmark_stations = benchmark_binary.stations.name
    comparison_stations = comparison_binary.stations.name

    # interested only if station available in both
    both = np.isin(benchmark_stations, comparison_stations)
    if verbose:
        print(
            "n_stations: {} benchmark, {} comparison, intersection: {}".format(
                benchmark_stations.size, comparison_stations.size, np.sum(both)
            )
        )
    return benchmark_stations[both]


def plot_station(
    station_name,
    output_directory,
    benchmark_binary: Union[BBSeis, LFSeis, HFSeis],
    comparison_binary: Union[BBSeis, LFSeis, HFSeis],
    tmax=None,
    verbose=False,
):
    """Creates a waveform plot for a specific station."""

    if verbose:
        print("Plotting station: {}...".format(station_name))

    data_to_plot = []
    x_max = 0

    # Load sim data
    bench_yx = (
        benchmark_binary.vel(station_name).transpose(),
        np.arange(benchmark_binary.nt) * benchmark_binary.dt
        + benchmark_binary.start_sec,
    )
    x_max = max(x_max, bench_yx[1][-1])
    data_to_plot.append(bench_yx)

    # Load sim data
    comparison_yx = (
        comparison_binary.vel(station_name).transpose(),
        np.arange(comparison_binary.nt) * comparison_binary.dt
        + comparison_binary.start_sec,
    )
    x_max = max(x_max, comparison_yx[1][-1])
    data_to_plot.append(comparison_yx)

    # If both, combine
    all_y = np.concatenate(
        (
            np.array(comparison_yx[0], dtype=np.float32),
            np.array(bench_yx[0], dtype=np.float32),
        ),
        axis=1,
    )

    # get axis min/max
    y_min, y_max = np.min(all_y), np.max(all_y)

    pgvs = np.max(np.abs(all_y), axis=1)
    ppgvs = np.max(all_y, axis=1)
    npgvs = np.min(all_y, axis=1)
    y_diff = y_max - y_min

    if tmax is not None:
        x_max = tmax
    scale_length = max(int(round(x_max / 25.0)) * 5, 5)

    # start plot
    colours = ["black", "red"]
    f, axis = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 4), dpi=96)
    f.subplots_adjust(
        left=0.08, bottom=0.12, right=0.96, top=None, wspace=0.08, hspace=0
    )
    plt.suptitle(
        station_name,
        fontsize=20,
        x=0.02,
        y=0.5,
        horizontalalignment="left",
        verticalalignment="center",
    )
    plt.xlim([0, x_max])

    # subplots
    for i, s in enumerate(data_to_plot):
        for j in range(3):
            ax = axis[j]
            ax.set_axis_off()
            ax.set_ylim([y_min - y_diff * 0.15, y_max])

            if s is not None:
                (line,) = ax.plot(
                    s[1],
                    s[0][j] * min(y_max / ppgvs[j], y_min / npgvs[j]),
                    color=colours[i],
                    linewidth=1,
                )
            if j == 2:
                line.set_label(["Benchmark", "Comparison"][i])
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
                ax.set_title(extensions[j][1:], fontsize=18)
                ax.text(x_max, y_max, "{:.1f}".format(pgvs[j]), fontsize=14)

    plt.savefig(os.path.join(output_directory, "{}.png".format(station_name)))
    plt.close()


if __name__ == "__main__":
    args = load_args()

    # binary class object or text folder location
    locations = [load_location(path, args.v) for path in args.waveforms]
    # station list
    stations = [load_stations(location) for location in locations]
    # common stations
    stations_all = stations[0]
    for stats in stations[1:]:
        stations_all = np.intersect1d(stations_all, stats)

    print(stations_all)
    # Select stations randomly
    #if 0 < args.n_stations < stations.shape[0]:
    #    stations = np.random.choice(stations, args.n_stations, replace=False)

    #p = Pool(args.nproc)
    #msgs = [
    #    (s, args.out, benchmark_binary, comparison_binary, args.tmax, args.v)
    #    for s in stations
    #]
    #p.starmap(plot_station, msgs)
