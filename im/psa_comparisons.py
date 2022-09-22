#!/usr/bin/env python
"""
Compare spectral acceleration with vibration period.
"""

import matplotlib as mpl
from qcore import shared

mpl.use("Agg")

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qcore.formats import load_im_file_pd
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
        help="Path to IM file and label. Each file will be plotted together. "
             "Repeated labels will have the log space mean plotted along with the individual sites.",
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
    parser.add_argument(
        "--real_only",
        help="limit stations to plot to only 'real' ones",
        action="store_true",
    )
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
    ims = {}
    psas = []
    for imcsv, label in args.imcsv:
        if label not in ims:
            ims[label] = []
        ims[label].append(load_im_file_pd(imcsv, all_ims=True, comp=args.comp))
        im_names = ims[label][-1].columns.values.astype(str)
        psas.append(
            im_names[
                np_startswith(im_names, "pSA_")
                & np.invert(np_endswith(im_names, "_sigma"))
                ]
        )
    ims = {
        key: pd.concat(
            value, keys=range(1, len(value) + 1), names=["rel"] + value[0].index.names
        )
        for key, value in ims.items()
    }

    # only common pSAs
    psa_names = intersection(psas)
    psa_vals = np_lstrip(psa_names, chars="pSA_").astype(np.float32)
    # sorted
    sort_idx = np.argsort(psa_vals)
    psa_names = psa_names[sort_idx]
    psa_vals = psa_vals[sort_idx]
    # value range
    y_max = max([df[psa_names].max().max() for df in ims.values()])
    y_min = min([df[psa_names].min().min() for df in ims.values()])

    stations = intersection([[index[1] for index in df.index] for df in ims.values()])

    # each station is a plot containing imcsvs as series
    for station in stations:
        if args.stations is not None and station not in args.stations:
            continue
        if args.real_only and shared.is_virtual_station(station):
            continue
        fig = plt.figure(figsize=(7.6, 7.5), dpi=100)
        plt.rcParams.update({"font.size": 14})

        # add a series for each csv
        for label, df in ims.items():
            colour = None
            df: pd.DataFrame
            rels = df.index.get_level_values(0).unique()
            multi_rel = len(rels) > 1
            for rel in rels:
                line: plt.Line2D = plt.loglog(
                    psa_vals,
                    df.loc[(rel, station, args.comp), psa_names].values,
                    label=f"{station} {label}",
                    linewidth=3,
                    color=colour,
                    alpha=0.3 if multi_rel else 1.0,
                )[0]
                if colour is None:
                    colour = line.get_color()
            if multi_rel:
                log_mean_vals = np.exp(
                    np.log(df.xs(station, axis=0, level=1)).groupby(level=1).mean()
                )[psa_names].values.squeeze()
                plt.loglog(
                    psa_vals,
                    log_mean_vals,
                    label=f"{station} {label} mean",
                    linewidth=5,
                    color=colour,
                )

        # plot formatting
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="best")
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
                f"pSA_comp_{args.comp}_vs_Period_{args.run_name.replace(' ', '_')}_{station}.png",
            )
        )
        plt.close()
