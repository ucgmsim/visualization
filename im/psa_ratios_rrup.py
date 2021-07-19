#!/usr/bin/env python
"""
Ratios for each IM.
"""

import matplotlib as mpl
from qcore.im import IM

mpl.use("Agg")

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from qcore.formats import load_im_file
from qcore.nputil import argsearch

np_startswith = np.core.defchararray.startswith


def load_args():
    """
    Process command line arguments.
    """

    # read

    parser = ArgumentParser()
    parser.add_argument("stats", help="ll or rrup file for locations", type=Path)
    parser.add_argument(
        "--imcsv",
        nargs=2,
        required=True,
        help="path to IM file and label",
        action="append",
    )

    parser.add_argument(
        "-d",
        "--out_dir",
        default=".",
        help="output folder to place xyz file",
        type=Path,
    )
    parser.add_argument(
        "--run_name",
        help="run_name",
        default="event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm",
    )
    parser.add_argument("--comp", help="component", default="geom")
    args = parser.parse_args()

    # validate
    assert len(args.imcsv[0]) > 1
    for imcsv in args.imcsv:
        assert Path(imcsv[0]).is_file()

    assert args.stats.is_file()

    args.out_dir.mkdir(exist_ok=True)

    return args


def get_print_name(im, comp):
    im_name = IM.from_im_name(im).get_im_name().replace(".", "p")
    return f"{im_name}_comp_{comp}"


if __name__ == "__main__":
    args = load_args()

    # load rrups - station_name,lon,lat,rrup...
    name_rrup = np.loadtxt(
        args.stats, dtype="|U7,f", usecols=(0, 3), skiprows=1, delimiter=","
    )

    # load im files (slow) for component, available pSA columns
    imcsv = [load_im_file(x[0], comp=args.comp) for x in args.imcsv]

    for i in range(1, len(imcsv)):
        # common IMs
        im_names = np.intersect1d(imcsv[0].dtype.names[2:], imcsv[i].dtype.names[2:])

        # common stations
        os_idx = argsearch(imcsv[0].station, imcsv[i].station)
        ls_idx = argsearch(name_rrup["f0"], imcsv[i].station)
        os_idx.mask += np.isin(os_idx, ls_idx.compressed(), invert=True)
        obs_idx = np.where(os_idx.mask == False)[0]
        sim_idx = os_idx.compressed()
        stations = imcsv[0].station[obs_idx]
        name_rrup = name_rrup[argsearch(stations, name_rrup["f0"]).compressed()]

        for im in im_names:
            if im.endswith("_sigma"):
                print(f"{im} skipping")
                continue
            print(im)

            print_name = get_print_name(im, args.comp)
            im_ratios = np.log(imcsv[0][im][obs_idx].tolist()) - np.log(
                imcsv[i][im][sim_idx].tolist()
            )
            bias_string = f"median={np.mean(im_ratios):.2g}, sigma={np.std(im_ratios, ddof=1):.2g}"

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
            ylabel = f"ln({args.imcsv[0][1]}/{args.imcsv[i][1]})\n-{print_name}"
            plt.ylabel(ylabel, fontsize=14)

            plt.xlabel("Source-to-site distance, $R_{rup}$ (km)", fontsize=14)
            plt.title(args.run_name, fontsize=16)
            if not (np.max(im_ratios) < -2.5 or np.min(im_ratios) > 2.5):
                plt.ylim([-2.5, 2.5])
            plt.xlim(
                [0.7, 45]
            )  # manually adjusted x-axis to strip blank space on plot, TODO Should be autoed

            plt.savefig(
                args.out_dir
                / f"{print_name}_{args.imcsv[0][1]}_{args.imcsv[i][1]}_Ratio_withRrup_{args.run_name}.png"
            )
            plt.close()
