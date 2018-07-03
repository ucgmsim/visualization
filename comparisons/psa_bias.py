#!/usr/bin/env python2

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np

from qcore.formats import load_im_file
from qcore.nputil import argsearch

np_startswith = np.core.defchararray.startswith
np_lstrip = np.core.defchararray.lstrip

def load_args():
    """
    Process command line arguments.
    """

    # read
    parser = ArgumentParser()
    parser.add_argument('obs', help = 'path to OBSERVED IM file')
    parser.add_argument('sim', help = 'path to SIMULATED IM file')
    parser.add_argument('-d', '--out-dir', default = '.', \
                        help = 'output folder to place plot')
    # TODO: automatically retrieved default
    parser.add_argument('--run-name', help = 'run_name - should automate?', \
                        default = 'event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm')
    parser.add_argument('--comp', help = 'component', default = 'geom')
    args = parser.parse_args()

    # validate
    assert(os.path.isfile(args.obs))
    assert(os.path.isfile(args.sim))
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args

###
### MAIN
###

args = load_args()

# load im files (slow) for component, available pSA columns
sim_ims = load_im_file(args.sim, all_psa = True)
sim_ims = sim_ims[sim_ims.component == args.comp]
sim_psa = [sim_ims.dtype.names[col_i] for col_i in \
           np.where(np_startswith(sim_ims.dtype.names, 'pSA_'))[0]]
obs_ims = load_im_file(args.obs, all_psa = True)
obs_ims = obs_ims[obs_ims.component == args.comp]
obs_psa = [obs_ims.dtype.names[col_i] for col_i in \
           np.where(np_startswith(obs_ims.dtype.names, 'pSA_'))[0]]

# common pSAs
psa_names = np.intersect1d(obs_psa, sim_psa)
psa_vals = np_lstrip(psa_names, chars='pSA_').astype(np.float32)
# sorted
sort_idx = np.argsort(psa_vals)
psa_names = psa_names[sort_idx]
psa_vals = psa_vals[sort_idx]
del obs_psa, sim_psa, sort_idx

# common stations
obs_in_sim = argsearch(obs_ims.station, sim_ims.station)
obs_idx = np.where(obs_in_sim.mask == False)[0]
sim_idx = obs_in_sim.compressed()
del obs_in_sim

# plotting data
psa_ratios = np.log(obs_ims[psa_names][obs_idx].tolist()) / \
             np.log(sim_ims[psa_names][sim_idx].tolist())
psa_means = np.mean(psa_ratios, axis = 0)
psa_std = np.std(psa_ratios, axis = 0)
del obs_ims, sim_ims, obs_idx, sim_idx, psa_names, psa_ratios

# plot
