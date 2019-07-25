#!/usr/bin/env python2
"""
Plots 3 components for simulated and observed seismograms.
USAGE: run with -h parameter

sample command:
python waveforms_sim_obs.py /nesi/project/nesi00213/dev/impp_datasets/Darfield/sim/Vel /nesi/project/nesi00213/dev/impp_datasets/Darfield/obs/velBB/ ~/test_mpl/waveforms
"""

import matplotlib as mpl

mpl.use("Agg")

from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
import os

import numpy as np
import matplotlib.pyplot as plt

from qcore.timeseries import BBSeis, read_ascii

# files that contain the 3 components (text based)
# must be in same order as binary results (x, y, z)
extensions = [".000", ".090", ".ver"]


def load_args():
    """
    Process command line arguments.
    """
    # read
    parser = ArgumentParser(
        description="Plots 3 components for simulated and observed "
        "seismograms. One of --sim or --obs has "
        "to be set. If both are set, the intersection of "
        "the stations is used."
    )

    parser.add_argument("out", help="output folder " "to place plots")
    parser.add_argument(
        "--sim", help="path to binary file or " "text dir for simulated seismograms"
    )
    parser.add_argument("--obs", help="path to text dir " "for observed seismograms")
    parser.add_argument(
        "--sim-prefix",
        default="",
        help="sim text files " "are named <prefix>station.comp",
    )
    parser.add_argument(
        "--obs-prefix", default="", help="obs text files are named <prefix>station.comp"
    )
    parser.add_argument(
        "--n_stations",
        default=-1,
        help="Number of stations, selected randomly, to plot. " "Default is all (-1)",
        type=int,
    )
    parser.add_argument("-v", help="verbose messages", action="store_true")
    parser.add_argument(
        "-n", "--nproc", help="number of processes to use", type=int, default=1
    )
    parser.add_argument(
        "-t", "--tmax", type=float, help="maximun duration of " "waveform simulation"
    )
    args = parser.parse_args()

    # validate
    if args.sim is not None:
        if os.path.isfile(args.sim):
            args.binary_sim = True
        elif os.path.isdir(args.sim):
            args.binary_sim = False
        else:
            raise ValueError("sim location not found")
        if args.v:
            print("sim data is binary: %r" % (bool(args.binary_sim)))

    if args.obs is not None:
        if not os.path.isdir(args.obs):
            raise ValueError("obs location not found")

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    if args.tmax is not None and args.tmax <= 0:
        parser.error("Duration -t/--tmax must be greater than 0")
    return args


def load_station_inter(args, sim_bb=None):
    """
    Determine stations available for plotting.
    returns numpy array of intersecting station names
    """
    # stations available in sim
    sim_stations = load_sim_stations(args, sim_bb)

    # stations available in obs
    obs_stations = load_obs_stations(args)

    # interested only if station available in both
    both = np.isin(sim_stations, obs_stations)
    if args.v:
        print(
            "n_stations: %d sim, %d obs, intersection: %d"
            % (sim_stations.size, len(obs_stations), np.sum(both))
        )
    return sim_stations[both]


def station_from_filename(path, n_prefix, n_suffix):
    return os.path.basename(path)[n_prefix:-n_suffix]


def load_sim_stations(args, sim_bb=None):
    """Loads all sim available stations"""
    if args.binary_sim:
        sim_stations = sim_bb.stations.name
    else:
        sim_stations = glob(
            os.path.join(args.sim, "%s*%s" % (args.sim_prefix, extensions[0]))
        )
        sim_stations = np.array(
            [
                station_from_filename(s, len(args.sim_prefix), len(extensions[0]))
                for s in sim_stations
            ]
        )
    return sim_stations


def load_obs_stations(args):
    """Loads all available obs stations"""
    obs_stations = glob(
        os.path.join(args.obs, "%s*%s" % (args.obs_prefix, extensions[0]))
    )
    obs_stations = [
        station_from_filename(o, len(args.obs_prefix), len(extensions[0]))
        for o in obs_stations
    ]
    return np.asarray(obs_stations)


def plot_station(args, name, sim_bb=None):
    """Creates a waveform plot for a specific station.

    If only sim data is provided the obs line is hidden and vice versa.
    """

    def load_txt(folder, prefix):
        """Load ascii data"""
        sim = [
            read_ascii(
                os.path.join(folder, "%s%s%s" % (prefix, name, extensions[i])),
                meta=True,
            )
            for i in xrange(len(extensions))
        ]
        return (
            [s[0] for s in sim],
            np.arange(sim[0][1]["nt"]) * sim[0][1]["dt"] + sim[0][1]["sec"],
        )

    if args.v:
        print("Plotting station: %s..." % (name))

    data_to_plot = []
    x_max = 0

    # Load obs data
    if args.obs is not None:
        obs_yx = load_txt(args.obs, args.obs_prefix)
        all_y = np.asarray(obs_yx[0], dtype=np.float32)
        x_max = obs_yx[1][-1]
        data_to_plot.append(obs_yx)
    else:
        data_to_plot.append(None)

    # Load sim data
    if args.sim is not None:
        if args.binary_sim:
            sim_yx = (
                sim_bb.vel(name).T,
                np.arange(sim_bb.nt) * sim_bb.dt + sim_bb.start_sec,
            )
        else:
            sim_yx = load_txt(args.sim, args.sim_prefix)
        all_y = np.asarray(sim_yx[0], dtype=np.float32)
        x_max = x_max if sim_yx[1][-1] < x_max else sim_yx[1][-1]
        data_to_plot.append(sim_yx)
    else:
        data_to_plot.append(None)

    # If both, combine
    if args.sim is not None and args.obs is not None:
        all_y = np.concatenate(
            (
                np.array(sim_yx[0], dtype=np.float32),
                np.array(obs_yx[0], dtype=np.float32),
            ),
            axis=1,
        )

    # get axis min/max
    y_min, y_max = all_y.min(), all_y.max()

    pgvs = np.max(np.abs(all_y), axis=1)
    ppgvs = np.max(all_y, axis=1)
    npgvs = np.min(all_y, axis=1)
    y_diff = y_max - y_min

    if args.tmax is not None:
        x_max = args.tmax
    scale_length = max(int(round(x_max / 25.0)) * 5, 5)

    # start plot
    colours = ["black", "red"]
    f, axis = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(20, 4), dpi=96)
    f.subplots_adjust(
        left=0.08, bottom=0.12, right=0.96, top=None, wspace=0.08, hspace=0
    )
    plt.suptitle(
        name,
        fontsize=20,
        x=0.02,
        y=0.5,
        horizontalalignment="left",
        verticalalignment="center",
    )
    plt.xlim([0, x_max])

    # subplots
    for i, s in enumerate(data_to_plot):
        for j in xrange(3):
            ax = axis[i, j]
            ax.set_axis_off()
            ax.set_ylim([y_min - y_diff * 0.15, y_max])

            if s is not None:
                ax.plot(
                    s[1],
                    s[0][j] * min(y_max / ppgvs[j], y_min / npgvs[j]),
                    color=colours[i],
                    linewidth=1,
                )

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
                ax.text(x_max, y_max, "%.1f" % (pgvs[j]), fontsize=14)

    plt.savefig(os.path.join(args.out, "%s.png" % (name)))
    plt.close()


if __name__ == "__main__":
    args = load_args()

    sim_bb = None
    if args.sim is not None:
        # Load binary
        if args.binary_sim:
            sim_bb = BBSeis(args.sim)

        # Only sim stations
        if args.obs is None:
            stations = load_sim_stations(args, sim_bb)
        # Intersection of sim and obs stations
        else:
            stations = load_station_inter(args, sim_bb)
    # Only obs stations
    elif args.sim is None and args.obs is not None:
        stations = load_obs_stations(args)
    else:
        raise ValueError(
            "Invalid argument combination. "
            "At least one of --sim or --obs has to be set."
        )

    # Select stations randomly
    if 0 < args.n_stations < stations.shape[0]:
        stations = np.random.choice(stations, args.n_stations, replace=False)

    # multiprocessing or serial (debug friendly)
    if args.nproc > 1:

        def plot_station_star(params):
            return plot_station(*params)

        p = Pool(args.nproc)
        msgs = [(args, s, sim_bb) for s in stations]
        p.map(plot_station_star, msgs)
    else:
        [plot_station(args, s, sim_bb) for s in stations]
