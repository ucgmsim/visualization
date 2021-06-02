#!/usr/bin/env python
"""
Compare spectral acceleration with vibration period.
"""

import matplotlib as mpl

mpl.use("Agg")

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np

from qcore.formats import load_im_file_pd
from qcore.nputil import argsearch
from visualization.util import intersection

NOT_FOUND = np.ma.masked
np_endswith = np.core.defchararray.endswith
np_startswith = np.core.defchararray.startswith
np_lstrip = np.core.defchararray.lstrip


def load_args():
    """
    Process command line arguments.
    """

    # read
    parser = ArgumentParser()
    parser.add_argument(
        "--imcsv",
        required=True,
        nargs=2,
        help="path to IM file, repeat as required",
        action="append",
    )
    parser.add_argument(
        "-d", "--out-dir", default=".", help="output folder to place plots"
    )
    parser.add_argument(
        "--run-name",
        help="run_name",
        default="event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm",
    )
    parser.add_argument("--comp", help="component", default="geom")
    parser.add_argument("--stations", help="limit stations to plot", nargs="+")
    args = parser.parse_args()

    # validate
    for imcsv in args.imcsv:
        assert os.path.isfile(imcsv[0])
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args


if __name__ == "__main__":
    args = load_args()

    # load im tables
    ims = []
    psas = []
    for imcsv in args.imcsv:
        ims.append(load_im_file_pd(imcsv[0], all_ims=True, comp=args.comp))
        im_names = ims[-1].columns.values.astype(str)
        psas.append(
            im_names[
                np_startswith(im_names, "pSA_")
                & np.invert(np_endswith(im_names, "_sigma"))
            ]
        )

    # only common pSAs
    psa_names = intersection(psas)
    psa_vals = np_lstrip(psa_names, chars="pSA_").astype(np.float32)
    # sorted
    sort_idx = np.argsort(psa_vals)
    psa_names = psa_names[sort_idx]
    psa_vals = psa_vals[sort_idx]
    # value range
    y_max = max([ims[i][psa_names].max().max() for i in range(len(ims))])
    y_min = min([ims[i][psa_names].min().min() for i in range(len(ims))])

    # each station is a plot containing imcsvs as series
    for station in [index[0] for index in ims[0].index]:
        if args.stations is not None and station not in args.stations:
            continue
        fig = plt.figure(figsize=(7.6, 7.5), dpi=100)
        plt.rcParams.update({"font.size": 14})

        # add a series for each csv
        for i, imcsv in enumerate(args.imcsv):
            if station not in ims[i].index:
                continue
            plt.loglog(
                psa_vals,
                ims[i].loc[(station, args.comp), psa_names].values,
                label=f"{station} {imcsv[1]}",
                linewidth=3,
            )

        # plot formatting
        plt.legend(loc="best")
        plt.ylabel("Spectral acceleration (g)", fontsize=14)
        plt.xlabel("Vibration period, T (s)", fontsize=14)
        plt.title(args.run_name)
        # plt.xlim([x_min, x_min * 10e4])
        plt.xlim([psa_vals[0], psa_vals[-1]])
        plt.ylim([max(0.001, y_min), min(5, y_max)])
        # plt.ylim([0.001, 5])
        plt.savefig(
            os.path.join(
                args.out_dir,
                f"pSA_comp_{args.comp}_vs_Period_{args.run_name}_{station}.png",
            )
        )
        plt.close()
