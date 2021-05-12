#!/usr/bin/env python

import matplotlib as mpl

mpl.use("Agg")

from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import numpy as np

from qcore.formats import load_im_file
from qcore.nputil import argsearch

np_startswith = np.core.defchararray.startswith
np_endswith = np.core.defchararray.endswith
np_lstrip = np.core.defchararray.lstrip


def load_args():
    """
    Process command line arguments.
    """

    # read
    parser = ArgumentParser()
    parser.add_argument("imcsv1", help="path to first IM file")
    parser.add_argument("imcsv2", help="path to second IM file to compare with first")
    parser.add_argument("--imcsv3", help="path to third IM file to compare with first")
    parser.add_argument(
        "-d", "--out-dir", default=".", help="output folder to place plot"
    )
    # TODO: automatically retrieved default
    parser.add_argument(
        "--run-name",
        help="run_name - should automate?",
        default="event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm",
    )
    parser.add_argument("--comp", help="component", default="geom")
    args = parser.parse_args()

    # validate
    assert os.path.isfile(args.imcsv1)
    assert os.path.isfile(args.imcsv2)
    if args.imcsv3 is not None:
        assert os.path.isfile(args.imcsv3)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args


def get_ims_psa(arg_sim):
    sim_ims = load_im_file(arg_sim, all_psa=True)
    sim_ims = sim_ims[sim_ims.component == args.comp]
    sim_psa = [
        sim_ims.dtype.names[col_i]
        for col_i in np.where(
            np.logical_and(
                np_startswith(sim_ims.dtype.names, "pSA_"),
                np.logical_not(np_endswith(sim_ims.dtype.names, "sigma")),
            )
        )[0]
    ]
    return sim_ims, sim_psa


def calc_ratio(arg_im1, arg_im2):
    ims_2, psa_2 = get_ims_psa(arg_im2)
    ims_1, psa_1 = get_ims_psa(arg_im1)

    psa_names = np.intersect1d(psa_1, psa_2)
    psa_vals = np_lstrip(psa_names, chars="pSA_").astype(np.float32)
    # sorted
    sort_idx = np.argsort(psa_vals)
    psa_names = psa_names[sort_idx]
    psa_vals = psa_vals[sort_idx]

    # common stations
    obs_in_sim = argsearch(ims_1.station, ims_2.station)
    obs_idx = np.where(obs_in_sim.mask == False)[0]
    sim_idx = obs_in_sim.compressed()
    del obs_in_sim

    # plotting data
    psa_ratios = np.log(ims_1[psa_names][obs_idx].tolist()) - np.log(
        ims_2[psa_names][sim_idx].tolist()
    )
    psa_means = np.mean(psa_ratios, axis=0)
    psa_std = np.std(psa_ratios, axis=0)

    return psa_vals, psa_means, psa_std


###
### MAIN
###

args = load_args()

psa_vals, psa_means, psa_std = calc_ratio(args.imcsv1, args.imcsv2)
if args.imcsv3 is not None:
    psa_vals_emp, psa_means_emp, psa_std_emp = calc_ratio(args.imcsv1, args.imcsv3)

# plot
fig = plt.figure(figsize=(14, 7.5), dpi=100)
plt.rcParams.update({"font.size": 18})
plt.fill_between(
    psa_vals,
    psa_means - psa_std,
    psa_means + psa_std,
    facecolor=[1, 0.8, 0.8],
    edgecolor=[1, 0.2, 0.2],
    linestyle="dashed",
    linewidth=0.5,
    alpha=0.5,
)
plt.plot(
    psa_vals,
    psa_means,
    color="red",
    linestyle="solid",
    linewidth=5,
    label="Physics-based",
)
if args.imcsv3 is not None:
    plt.fill_between(
        psa_vals_emp,
        psa_means_emp - psa_std_emp,
        psa_means_emp + psa_std_emp,
        facecolor=[0.8, 0.8, 1],
        edgecolor=[0.2, 0.2, 1],
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.5,
    )
    plt.plot(
        psa_vals_emp,
        psa_means_emp,
        color="blue",
        linestyle="solid",
        linewidth=5,
        label="Empirical",
    )
plt.plot(
    psa_vals, np.zeros_like(psa_vals), color="black", linestyle="dashed", linewidth=3
)

# plot formatting
plt.gca().set_xscale("log")
plt.minorticks_on()
plt.grid(b=True, axis="y", which="major")
plt.grid(b=True, axis="x", which="minor")
fig.set_tight_layout(True)
plt.legend(loc="best")
plt.ylabel("pSA residual, ln(obs)-ln(GMM)", fontsize=14)
plt.xlabel("Vibration period, T (s)", fontsize=14)
plt.title(args.run_name, fontsize=16)
plt.xlim([0.01, 10])
if not (np.max(psa_means) < -2.5 or np.min(psa_means) > 2.5):
    plt.ylim([-2.5, 2.5])
plt.savefig(
    os.path.join(args.out_dir, "pSAWithPeriod_comp_{args.comp}_{args.run_name}.png")
)
plt.close()
