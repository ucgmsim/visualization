#!/usr/bin/env python2

from argparse import ArgumentParser
import os

import numpy as np

from qcore.formats import load_im_file
from qcore.nputil import argsearch


np_startswith = np.core.defchararray.startswith
# np_lstrip = np.core.defchararray.lstrip


def load_args():
    """
    Process command line arguments.
    """

    # read
    parser = ArgumentParser()
    parser.add_argument("obs", help="path to OBSERVED IM file")
    parser.add_argument("sim", help="path to SIMULATED IM file")
    parser.add_argument("stats", help="ll or rrup file for locations")
    parser.add_argument(
        "-d", "--out-dir", default=".", help="output folder to place xyz file"
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
    assert os.path.isfile(args.obs)
    assert os.path.isfile(args.sim)
    assert os.path.isfile(args.stats)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args


###
### MAIN
###

args = load_args()

# load station locations
if args.stats.endswith(".csv"):
    # rrup file - station_name,lon,lat,...
    lon_lat_name = np.loadtxt(
        args.stats, dtype="f,f,|S7", usecols=(1, 2, 0), skiprows=1, delimiter=","
    )
else:
    # ll file - lon lat name
    lon_lat_name = np.loadtxt(args.stats, dtype="f,f,|S7")

# load im files (slow) for component, available pSA columns
sim_ims = load_im_file(args.sim, all_psa=True)
sim_ims = sim_ims[sim_ims.component == args.comp]
# sim_psa = [sim_ims.dtype.names[col_i] for col_i in np.where(np_startswith(sim_ims.dtype.names, 'pSA_'))[0]]
sim_psa = np.array(sim_ims.dtype.names[2:])  # all_ims
obs_ims = load_im_file(args.obs, all_psa=True)
obs_ims = obs_ims[obs_ims.component == args.comp]
# obs_psa = [obs_ims.dtype.names[col_i] for col_i in np.where(np_startswith(obs_ims.dtype.names, 'pSA_'))[0]]
obs_psa = np.array(obs_ims.dtype.names[2:])
print(obs_psa)
# common pSAs
psa_names = np.intersect1d(obs_psa, sim_psa)

# psa_vals = np_lstrip(psa_names, chars='pSA_').astype(np.float32)

# common stations
os_idx = argsearch(obs_ims.station, sim_ims.station)
ls_idx = argsearch(lon_lat_name["f2"], sim_ims.station)
os_idx.mask += np.isin(os_idx, ls_idx.compressed(), invert=True)
obs_idx = np.where(os_idx.mask == False)[0]
sim_idx = os_idx.compressed()
stations = obs_ims.station[obs_idx]
lon_lat_name = lon_lat_name[argsearch(stations, lon_lat_name["f2"]).compressed()]

# all data
xyz = np.zeros((stations.size, 2 + psa_names.size))
xyz[:, 0] = lon_lat_name["f0"]
xyz[:, 1] = lon_lat_name["f1"]
xyz[:, 2:] = np.log(obs_ims[psa_names][obs_idx].tolist()) - np.log(
    sim_ims[psa_names][sim_idx].tolist()
)

new_psa_names = []
for name in psa_names:
    if name.startswith("pSA"):
        name = name.replace("_", " (") + "s)"
    new_psa_names.append(name)

filename = os.path.join(args.out_dir, "stat_ratios_all_ims_%s.xyz" % (args.run_name))
np.savetxt(
    filename,
    xyz,
    fmt="%.5e",
    comments="",
    header="""Residual Plot
IM residual
polar:fg-80/0/0,bg-0/0/80 0.2
-1.5 1.5 0.25 0.5
%d white
%s"""
    % (psa_names.size, ", ".join(new_psa_names)),
)
