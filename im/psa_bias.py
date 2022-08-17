#!/usr/bin/env python
"""
Plot pSA residual with vibration period.
"""

import matplotlib as mpl

mpl.use("Agg")

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from qcore.constants import Components
from qcore.formats import load_im_file
from qcore.nputil import argsearch

np_startswith = np.core.defchararray.startswith
np_endswith = np.core.defchararray.endswith
np_lstrip = np.core.defchararray.lstrip

# default colours for each imcsv ratio
FACE_EDGE_COLOURS = [
    [[1, 0.8, 0.8], [1, 0.2, 0.2], "red"],
    [[0.8, 0.8, 1], [0.2, 0.2, 1], "blue"],
    [[0.8, 1, 0.8], [0.2, 1, 0.2], "green"],
    [[1, 1, 0.8], [1, 1, 0.2], "yellow"],
    [[1, 0.8, 1], [1, 0.2, 1], "magenta"],
    [[0.8, 1, 1], [0.2, 1, 1], "cyan"],
]


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
        help="path to IM file and label. if more than one is supplied, will be compared to first one",
        action="append",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default=Path.cwd(),
        help="output folder to place plot",
        type=Path,
    )
    parser.add_argument(
        "--run_name",
        help="run_name (plot title)",
        default="event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm",
    )
    parser.add_argument(
        "--comp",
        help="component",
        choices=[d.str_value for d in Components],
        default="geom",
    )
    args = parser.parse_args()

    # validate
    assert len(args.imcsv[0]) > 1

    for imcsv in args.imcsv:
        assert Path(imcsv[0]).is_file()

    args.out_dir.mkdir(exist_ok=True)

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

    # plotting data
    psa_ratios = np.log(ims_1[psa_names][obs_idx].tolist()) - np.log(
        ims_2[psa_names][sim_idx].tolist()
    )
    psa_means = np.mean(psa_ratios, axis=0)
    psa_std = np.std(psa_ratios, axis=0)

    return psa_vals, psa_means, psa_std


if __name__ == "__main__":
    args = load_args()

    # plot
    fig = plt.figure(figsize=(14, 7.5), dpi=100)
    plt.rcParams.update({"font.size": 18})

    # data values
    for i in range(1, len(args.imcsv)):
        psa_vals, psa_means, psa_std = calc_ratio(args.imcsv[0][0], args.imcsv[i][0])
        name = f"{args.imcsv[0][1]}/{args.imcsv[i][1]}"
        plt.fill_between(
            psa_vals,
            psa_means - psa_std,
            psa_means + psa_std,
            facecolor=FACE_EDGE_COLOURS[i - 1][0],
            edgecolor=FACE_EDGE_COLOURS[i - 1][1],
            linestyle="dashed",
            linewidth=0.5,
            alpha=0.5,
        )
        plt.plot(
            psa_vals,
            psa_means,
            color=FACE_EDGE_COLOURS[i - 1][2],
            linestyle="solid",
            linewidth=5,
            label=f"ln({name})",
        )
    plt.plot(
        psa_vals,
        np.zeros_like(psa_vals),
        color="black",
        linestyle="dashed",
        linewidth=3,
    )

    # plot formatting
    plt.gca().set_xscale("log")
    plt.minorticks_on()
    plt.grid(b=True, axis="y", which="major")
    plt.grid(b=True, axis="x", which="minor")
    fig.set_tight_layout(True)
    plt.legend(loc="best")
    plt.ylabel(f"pSA residual", fontsize=14)
    plt.xlabel("Vibration period, T (s)", fontsize=14)
    plt.title(args.run_name, fontsize=16)
    plt.xlim([0.01, 10])
    if not (np.max(psa_means) < -2.5 or np.min(psa_means) > 2.5):
        plt.ylim([-2.5, 2.5])
    plt.savefig(args.out_dir / f"pSAWithPeriod_comp_{args.comp}_{args.run_name}.png")
    plt.close()
