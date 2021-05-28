#!/usr/bin/env python
"""
IM vs RRUP plot.
Plots error bars or Empirical range and scatter of values.
"""

import matplotlib as mpl
from empirical.util.classdef import TectType

mpl.use("Agg")

from argparse import ArgumentParser
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from qcore.formats import load_im_file_pd
from qcore.nputil import argsearch
from qcore.utils import setup_dir

from empirical.scripts import calculate_empirical
from empirical.util import empirical_factory, classdef

from visualization.util import intersection

N_BUCKETS = 10
# matplotlib point style for each imcsv
MARKERS = ["o", "P", "s", "X", "v", "^", "D"]


def load_args():
    """
    Process command line arguments.
    """
    # read
    parser = ArgumentParser()
    parser.add_argument("rrup", help="path to RRUP file", type=os.path.abspath)
    parser.add_argument(
        "--imcsv",
        nargs=2,
        required=True,
        help="path to IM file and label",
        action="append",
    )
    parser.add_argument(
        "--config",
        help="path to .yaml empirical config file (requires SRF info)",
        type=os.path.abspath,
    )
    parser.add_argument("--srf", help="path to srf info file", type=os.path.abspath)
    parser.add_argument(
        "--dist_min", default=0.1, type=float, help="GMPE param DistMin, default 0.1 km"
    )
    parser.add_argument(
        "--dist-max",
        default=100.0,
        type=float,
        help="GMPE param DistMax, default 100.0 km",
    )
    parser.add_argument(
        "--n_val", default=51, type=int, help="GMPE param n_val, default 51"
    )
    parser.add_argument("--bars", help="also plot error bars", action="store_true")
    parser.add_argument(
        "--out-dir",
        help="output folder to place plot",
        default=".",
        type=os.path.abspath,
    )
    parser.add_argument(
        "--run-name",
        help="run_name for title and filename",
        default="event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm",
    )
    parser.add_argument("--comp", help="component", default="geom")
    args = parser.parse_args()

    validate_args(args)

    setup_dir(args.out_dir)

    return args


def get_print_name(im, comp):
    """
    Takes in an IM and component and creates a filename friendly version.
    samples: pSA_0.02 -> pSA(0p02), pSA_10.0 -> pSA(10)
    """
    if im.startswith("pSA_"):
        whole, decimal = im.split("_")[-1].split(".")
        if int(decimal) == 0:
            decimal = ""
        else:
            decimal = f"p{decimal}"
        im = f"pSA({whole}{decimal})"
    return f"{im}_comp_{comp}"


def validate_args(args):
    """
    Validates command line input args.
    """
    assert os.path.isfile(args.rrup)
    for imcsv in args.imcsv:
        assert os.path.isfile(imcsv[0])

    if args.srf is not None:
        assert os.path.isfile(args.srf)
        if args.config is not None:
            assert os.path.isfile(args.config)
    elif args.config is not None:
        # srf file required if config given
        raise AssertionError("SRF required if config given")


def get_empirical_values(fault, im, model_dict, r_rup_vals, period):
    gmm = empirical_factory.determine_gmm(fault, im, model_dict)[0]
    # https://github.com/ucgmsim/post-processing/blob/master/im_processing/computations/GMPE.py
    # line 145
    r_jbs_vals = np.sqrt(np.maximum(0, r_rup_vals ** 2 - fault.ztor ** 2))
    e_medians = []
    e_sigmas = []
    for i in range(len(r_rup_vals)):
        site = classdef.Site()
        site.Rrup = r_rup_vals[i]
        site.Rjb = r_jbs_vals[i]
        value = empirical_factory.compute_gmm(fault, site, gmm, im, period)
        if isinstance(value, tuple):
            e_medians.append(value[0])
            e_sigmas.append(value[1][0])
        elif isinstance(value, list):
            for v in value:
                e_medians.append(v[0])
                e_sigmas.append(v[1][0])

    return np.array(e_medians), np.array(e_sigmas)


if __name__ == "__main__":
    args = load_args()

    # station name, rrup
    name_rrup = np.loadtxt(
        args.rrup, dtype="|U7,f", usecols=(0, 3), skiprows=1, delimiter=","
    )

    # load im files for component, available pSA columns
    ims = []
    for imcsv in args.imcsv:
        ims.append(load_im_file_pd(imcsv[0], comp=args.comp))
    im_names = intersection([im.columns[2:] for im in ims])

    stations = [index[0] for index in ims[0].index]
    name_rrup = name_rrup[argsearch(stations, name_rrup["f0"]).compressed()]
    # limit stations to those with rrups
    stations = name_rrup["f0"]
    rrups = name_rrup["f1"]

    logged_rrups = np.log(rrups)
    min_logged_rrup = np.min(logged_rrups)
    bucket_range = (np.max(logged_rrups) - min_logged_rrup) / N_BUCKETS

    masks = [
        (min_logged_rrup + (i + 1) * bucket_range >= logged_rrups)
        * (logged_rrups > min_logged_rrup + i * bucket_range)
        for i in range(N_BUCKETS)
    ]
    masks[0] = masks[0] + (logged_rrups == min_logged_rrup)
    bucket_rrups = np.exp(
        [min_logged_rrup + (i + 0.5) * bucket_range for i in range(N_BUCKETS)]
    )

    # empirical calc
    if args.srf is not None:
        model_dict = empirical_factory.read_model_dict(args.config)
        fault = calculate_empirical.create_fault_parameters(args.srf)
        fault.tect_type = TectType.SUBDUCTION_INTERFACE
        r_rup_vals = np.exp(
            np.linspace(
                np.log(args.dist_min),
                np.log(max(args.dist_max, max(rrups))),
                args.n_val,
            )
        )

    # plot
    for im in im_names:
        # skip sigma
        if im.endswith("_sigma"):
            continue

        print_name = get_print_name(im, args.comp)
        fig = plt.figure(figsize=(14, 7.5), dpi=100)
        plt.rcParams.update({"font.size": 18})
        values = []
        for im_set in ims:
            values.append(im_set[im][stations].values)

        # plot points
        for i, series in enumerate(values):
            plt.loglog(
                rrups,
                series,
                linestyle="None",
                markeredgewidth=None,
                markersize=0.5,
                marker=MARKERS[i % len(MARKERS)],
                label=args.imcsv[i][1],
            )

        # emp +- range plot
        if args.srf is not None:
            if "pSA" in im:
                im_type, p = im.split("_")
                period = [float(p)]
            else:
                period = None

            e_medians, e_sigmas = get_empirical_values(
                fault, im_type, model_dict, r_rup_vals, period
            )

            if np.size(e_medians) != 0:  # MMI does not have emp
                plt.plot(
                    r_rup_vals,
                    e_medians,
                    color="black",
                    marker=None,
                    linewidth=3,
                    label="Empirical",
                )
                plt.plot(
                    r_rup_vals,
                    e_medians * np.exp(-e_sigmas),
                    color="black",
                    marker=None,
                    linestyle="dashed",
                    linewidth=3,
                )
                plt.plot(
                    r_rup_vals,
                    e_medians * np.exp(e_sigmas[:]),
                    color="black",
                    marker=None,
                    linestyle="dashed",
                    linewidth=3,
                )

        if args.bars:
            # plot error bars
            means = np.asarray(
                [
                    np.mean(np.log(ims[0][im].loc[stations[mask]].values))
                    for mask in masks
                ]
            )
            stddevs = np.asarray(
                [
                    np.std(np.log(ims[0][im].loc[stations[mask]].values))
                    for mask in masks
                ]
            )
            plt.errorbar(
                bucket_rrups,
                np.exp(means),
                np.vstack(
                    (
                        np.exp(means) - np.exp(means - stddevs),
                        np.exp(means + stddevs) - np.exp(means),
                    )
                ),
                fmt="o",
                zorder=50,
                color="black",
                capsize=6,
            )

        # plot formatting
        plt.legend(loc="best", fontsize=9, numpoints=1)
        plt.ylabel(print_name)
        plt.xlabel("Source-to-site distance, $R_{rup}$ (km)")
        plt.minorticks_on()
        plt.title(args.run_name, fontsize=12)
        y_max = max([ims[i][im_names].max().max() for i in range(len(ims))])
        y_min = min([ims[i][im_names].min().min() for i in range(len(ims))])
        plt.ylim(top=y_max * 1.27)
        plt.xlim(1e-1, max(1e2, np.max(rrups) * 1.1))
        fig.set_tight_layout(True)
        plt.savefig(
            os.path.join(args.out_dir, f"{print_name}_with_Rrup_{args.run_name}.png"),
            dpi=400,
        )
        plt.close()
