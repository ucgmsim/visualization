#!/usr/bin/env python

import matplotlib as mpl

mpl.use("Agg")

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np

from qcore.formats import load_im_file
from qcore.nputil import argsearch

NOT_FOUND = np.ma.masked
np_startswith = np.core.defchararray.startswith
np_lstrip = np.core.defchararray.lstrip


def load_args():
    """
    Process command line arguments.
    """

    # read
    parser = ArgumentParser()
    parser.add_argument("-s", "--sim", help="path to SIMULATED IM file")
    parser.add_argument("-o", "--obs", help="path to OBSERVED IM file")
    parser.add_argument(
        "-d", "--out-dir", default=".", help="output folder to place plots"
    )
    # TODO: automatically retrieved default
    parser.add_argument(
        "--run-name",
        help="run_name - should automate?",
        default="event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm",
    )
    parser.add_argument("--comp", help="component", default="geom")
    args = parser.parse_args()

    args.have_sim = args.sim is not None
    args.have_obs = args.obs is not None
    args.have_both = args.sim is not None and args.obs is not None

    # validate
    assert args.have_sim or args.have_obs
    if args.have_sim:
        assert os.path.isfile(args.sim)
    if args.have_obs:
        assert os.path.isfile(args.obs)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args


###
### MAIN
###

args = load_args()
if args.have_sim:
    sim_ims = load_im_file(args.sim, all_psa=True)
    sim_ims = sim_ims[sim_ims.component == args.comp]
    sim_psa = [
        sim_ims.dtype.names[col_i]
        for col_i in np.where(np_startswith(sim_ims.dtype.names, "pSA_"))[0]
    ]
if args.have_obs:
    obs_ims = load_im_file(args.obs, all_psa=True)
    obs_ims = obs_ims[obs_ims.component == args.comp]
    obs_psa = [
        obs_ims.dtype.names[col_i]
        for col_i in np.where(np_startswith(obs_ims.dtype.names, "pSA_"))[0]
    ]
# only common pSA
if args.have_both:
    psa_names = np.intersect1d(obs_psa, sim_psa)
elif args.have_obs:
    psa_names = np.array(obs_psa)
else:
    psa_names = np.array(sim_psa)
psa_vals = np_lstrip(psa_names, chars="pSA_").astype(np.float32)

# get xlim
x_min = min(psa_vals)
x_max = max(psa_vals)

# sorted
sort_idx = np.argsort(psa_vals)
psa_names = psa_names[sort_idx]
psa_vals = psa_vals[sort_idx]
# pSA arrays
if args.have_sim:
    sim_psa = np.array(
        sim_ims.getfield(
            np.dtype({name: sim_ims.dtype.fields[name] for name in psa_names})
        ).tolist()
    )
    sim_stations = sim_ims.station
    del sim_ims
if args.have_obs:
    obs_psa = np.array(
        obs_ims.getfield(
            np.dtype({name: obs_ims.dtype.fields[name] for name in psa_names})
        ).tolist()
    )
    obs_stations = obs_ims.station
    del obs_ims

if args.have_both:
    stat_idx = enumerate(argsearch(obs_stations, sim_stations))
    stations = sim_stations
elif args.have_obs:
    stat_idx = enumerate(range(len(obs_stations)))
    stations = obs_stations
else:
    stat_idx = enumerate(range(len(sim_stations)))
    stations = sim_stations

# get ylim
y_max = max(np.max(obs_psa), np.max(sim_psa))
y_min = min(np.min(obs_psa), np.min(sim_psa))


# in the case of only sim or obs: both indexes are the same
for obs_idx, sim_idx in stat_idx:
    if sim_idx is NOT_FOUND:
        # obs station not found in sim
        continue
    station = stations[sim_idx]

    # plot data
    # fig = plt.figure(figsize = (14, 7.5), dpi = 100)
    fig = plt.figure(figsize=(7.6, 7.5), dpi=100)  # fig square
    plt.rcParams.update({"font.size": 14})
    if args.have_sim:
        plt.loglog(
            psa_vals,
            sim_psa[sim_idx],
            color="red",
            label=f"{station} Sim",
            linewidth=3,
        )
    if args.have_obs:
        plt.loglog(
            psa_vals,
            obs_psa[obs_idx],
            color="black",
            label=f"{station} Obs",
            linewidth=3,
        )

    # plot formatting
    plt.legend(loc="best")
    plt.ylabel("Spectral acceleration (g)", fontsize=14)
    plt.xlabel("Vibration period, T (s)", fontsize=14)
    plt.title(args.run_name)
    # plt.xlim([x_min, x_min * 10e4])
    plt.xlim([x_min, x_max])
    plt.ylim([max(0.001, y_min), min(5, y_max)])
    # plt.ylim([0.001, 5])
    plt.savefig(
        os.path.join(
            args.out_dir,
            f"pSA_comp_{args.comp}_vs_Period_{args.run_name}_{station}.png",
        )
    )
    plt.close()
