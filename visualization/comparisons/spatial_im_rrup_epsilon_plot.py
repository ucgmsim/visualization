#!/usr/bin/env python3
"""
IM epsilon value calculation script. Generates a file ot be passed to plot_items.py
Takes IM and Rrup files and uses them to calculate how far above or below the empirical median each simulated value is.

To see help message:
python spatial_im_rrup_epsilon_plot.py -h

Sample command:
python spatial_im_rrup_epsilon_plot.py ~/darfield_obs/rrups.csv  ~/darfield_sim/darfield_sim.csv /nesi/project/nesi00213/dev/impp_datasets/Darfield/source.info --out_file darfield_emp_new_rrup4.xyz
"""

from argparse import ArgumentParser
import os

import pandas as pd
import numpy as np

from qcore.formats import load_im_file
from qcore.nputil import argsearch
from qcore.utils import setup_dir

from empirical.scripts import calculate_empirical
from empirical.util import empirical_factory, classdef


def load_args():
    """
    Process command line arguments.
    """
    # read
    parser = ArgumentParser()
    parser.add_argument("rrup", help="path to RRUP file", type=os.path.abspath)
    parser.add_argument("sim", help="path to SIMULATED IM file", type=os.path.abspath)
    parser.add_argument("srf", help="path to srf info file", type=os.path.abspath)
    parser.add_argument(
        "--config", help="path to .yaml empirical config file", type=os.path.abspath
    )
    parser.add_argument(
        "--dist_min", default=0.1, type=float, help="GMPE param DistMin, default 0.1 km"
    )
    parser.add_argument(
        "--dist_max",
        default=100.0,
        type=float,
        help="GMPE param DistMax, default 100.0 km",
    )
    parser.add_argument(
        "--n_val", default=51, type=int, help="GMPE param n_val, default 51"
    )
    parser.add_argument(
        "--out_file",
        help="output folder to place plot",
        default="./epsilons.xyz",
        type=os.path.abspath,
    )
    parser.add_argument("--comp", help="component", default="geom")
    args = parser.parse_args()

    validate_args(args)

    setup_dir(os.path.dirname(args.out_file))
    return args


def get_print_name(im, comp):
    """Takes in an im and component and creates a printable name from them
    In the case of pSA ims the period is processed such that the letter p is used in place of a decimal point and any
    trailing 0s are trimmed.
    pSA_0.02 -> pSA(0p02)
    pSA_0.5 -> pSA(0p5)
    pSA_1.0 -> pSA(1)
    pSA_10.0 -> pSA(10)"""
    if im.startswith("pSA_"):
        whole, decimal = im.split("_")[-1].split(".")
        if int(decimal) == 0:
            decimal = ""
        else:
            decimal = "p{}".format(decimal)
        im = "pSA({}{})".format(whole, decimal)
    return "{}_comp_{}".format(im, comp)


def validate_args(args):
    """
       validates all input args;
       config arg exists if and only if srf arg exists
    """
    assert os.path.isfile(args.rrup)
    assert os.path.isfile(args.sim)

    assert os.path.isfile(args.srf)
    if args.config is not None:
        assert os.path.isfile(args.config)


def get_empirical_values(fault, im, model_dict, r_rup_vals, period):
    gmm = empirical_factory.determine_gmm(fault, im, model_dict)
    if gmm is None:
        return np.asarray([]), np.asarray([])
    gmm, comp = gmm
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


def main():
    args = load_args()

    name_rrup = pd.read_csv(args.rrup)

    # load im files (slow) for component, available pSA columns
    sim_ims = load_im_file(args.sim, comp=args.comp)

    # im_names = ["pSA_5.0"]
    im_names = sim_ims.dtype.names[2:]

    indexes = argsearch(sim_ims.station, name_rrup["station"].values)
    name_rrup = name_rrup.iloc[indexes]
    # empirical calc
    model_dict = empirical_factory.read_model_dict(args.config)
    fault = calculate_empirical.create_fault_parameters(args.srf)
    # fault.tect_type = classdef.TectType.SUBDUCTION_INTERFACE # Use to manually override subduction faults
    r_rup_vals = np.exp(
        np.linspace(
            np.log(args.dist_min),
            np.log(max(args.dist_max, max(name_rrup["r_rup"]))),
            args.n_val,
        )
    )

    data = name_rrup[["lon", "lat"]]
    data["lon"] = ((data["lon"] + 180) % 360) - 180

    # plot
    for im in im_names:
        print_name = get_print_name(im, args.comp)

        log_ys = np.log(sim_ims[im])

        # emp plot
        if "pSA" in im:
            im, p = im.split("_")
            period = [float(p)]
        else:
            period = None

        e_medians, e_sigmas = get_empirical_values(
            fault, im, model_dict, r_rup_vals, period
        )

        if np.size(e_medians) == 0:  # MMI does not have emp
            continue

        interp_e_medians = np.interp(
            name_rrup["r_rup"], r_rup_vals, e_medians.flatten()
        )
        interp_e_sigmas = np.interp(name_rrup["r_rup"], r_rup_vals, e_sigmas.flatten())

        log_medians = np.log(interp_e_medians)
        log_sigmas = np.log(interp_e_sigmas)

        data.loc[:, print_name] = (log_ys - log_medians) / np.abs(log_sigmas)

    data.to_csv(args.out_file, index=False, sep=" ", header=False)


if __name__ == "__main__":
    main()
