#!/usr/bin/env python
"""
Ratios for each IM.
"""

import matplotlib as mpl

mpl.use("Agg")

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np

from qcore.formats import load_im_file
from qcore.nputil import argsearch

np_startswith = np.core.defchararray.startswith


def load_args():
    """
    Process command line arguments.
    """

    # read
    parser = ArgumentParser()
    parser.add_argument("obs", help="path to observerd IM file")
    parser.add_argument("sim", help="path to simulated IM file")
    parser.add_argument("stats", help="ll or rrup file for locations")
    parser.add_argument(
        "-d", "--out-dir", default=".", help="output folder to place xyz file"
    )
    parser.add_argument(
        "--run-name",
        help="run_name",
        default="event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm",
    )
    parser.add_argument("--comp", help="component", default="geom")
    args = parser.parse_args()

    # validate
    assert os.path.isfile(args.obs)
    assert os.path.isfile(args.sim)
    assert os.path.isfile(args.stats)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args


def get_print_name(im, comp):
    if im.startswith("pSA_"):
        im = f"pSA({im.split('_')[-1].split('.')[0]:d}p{im.split('.')[-1]}"
        im = f"{im.rstrip('p0')})"
    return f"{im}_comp_{comp}"


if __name__ == "__main__":
    args = load_args()

    # load rrups - station_name,lon,lat,rrup...
    name_rrup = np.loadtxt(
        args.stats, dtype="|U7,f", usecols=(0, 3), skiprows=1, delimiter=","
    )

    # load im files (slow) for component, available pSA columns
    sim_ims = load_im_file(args.sim, comp=args.comp)
    obs_ims = load_im_file(args.obs, comp=args.comp)
    # common IMs
    im_names = np.intersect1d(obs_ims.dtype.names[2:], sim_ims.dtype.names[2:])

    # common stations
    os_idx = argsearch(obs_ims.station, sim_ims.station)
    ls_idx = argsearch(name_rrup["f0"], sim_ims.station)
    os_idx.mask += np.isin(os_idx, ls_idx.compressed(), invert=True)
    obs_idx = np.where(os_idx.mask == False)[0]
    sim_idx = os_idx.compressed()
    stations = obs_ims.station[obs_idx]
    name_rrup = name_rrup[argsearch(stations, name_rrup["f0"]).compressed()]

    for im in im_names:
        print_name = get_print_name(im, args.comp)
        im_ratios = np.log(obs_ims[im][obs_idx].tolist()) - np.log(
            sim_ims[im][sim_idx].tolist()
        )
        bias_string = (
            f"median={np.mean(im_ratios):.2g}, sigma={np.std(im_ratios, ddof=1):.2g}"
        )

        # plot
        fig = plt.figure(figsize=(14, 7.5), dpi=100)
        plt.rcParams.update({"font.size": 18})
        plt.semilogx(
            name_rrup["f1"],
            im_ratios,
            linestyle="None",
            color="blue",
            marker="s",
            markersize=12,
            label=bias_string,
        )

        # plot formatting
        plt.minorticks_on()
        plt.grid(b=True, axis="y", which="major")
        plt.grid(b=True, axis="x", which="minor")
        fig.set_tight_layout(True)
        plt.legend(loc="best", numpoints=1)
        plt.ylabel(f"ln(obs/sim)-{print_name}", fontsize=14)
        plt.xlabel("Source-to-site distance, $R_{rup}$ (km)", fontsize=14)
        plt.title(args.run_name, fontsize=16)
        if not (np.max(im_ratios) < -2.5 or np.min(im_ratios) > 2.5):
            plt.ylim([-2.5, 2.5])
        plt.xlim(
            [0.7, 45]
        )  # manually adjusted x-axis to strip blank space on plot, TODO Should be autoed

        plt.savefig(
            os.path.join(
                args.out_dir, f"{print_name}_ObsSimRatio_withRrup_{args.run_name}.png"
            )
        )
        plt.close()
