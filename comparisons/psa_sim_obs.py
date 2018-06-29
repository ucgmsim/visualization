#!/usr/bin/env python2

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
    parser.add_argument('sim', help = 'path to SIMULATED IM file')
    parser.add_argument('obs', help = 'path to OBSERVED IM file')
    parser.add_argument('-o', '--out-dir', default = '.', \
                        help = 'output folder to place plots')
    # TODO: automatically retrieved default
    parser.add_argument('--run-name', help = 'run_name - should automate?', \
                        default = 'event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm')
    parser.add_argument('--comp', help = 'component', default = 'geom')
    args = parser.parse_args()

    # validate
    assert(os.path.isfile(args.sim))
    assert(os.path.isfile(args.obs))
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args

###
### MAIN
###

args = load_args()
sim_ims = load_im_file(args.sim, all_psa = True)
obs_ims = load_im_file(args.obs, all_psa = True)
# keep only wanted component
sim_ims = sim_ims[sim_ims.component == args.comp]
obs_ims = obs_ims[obs_ims.component == args.comp]
# only common pSA
psa_names = np.intersect1d([obs_ims.dtype.names[col_i] for col_i in \
                            np.where(np_startswith(obs_ims.dtype.names, 'pSA_'))[0]], \
                           sim_ims.dtype.names)
psa_vals = np_lstrip(psa_names, chars='pSA_').astype(np.float32)
x_min = min(psa_vals)
# sorted
sort_idx = np.argsort(psa_vals)
psa_names = psa_names[sort_idx]
psa_vals = psa_vals[sort_idx]
# pSA arrays
sim_psa = np.array(sim_ims.getfield(np.dtype({name: sim_ims.dtype.fields[name] \
                                              for name in psa_names})).tolist())
obs_psa = np.array(obs_ims.getfield(np.dtype({name: obs_ims.dtype.fields[name] \
                                              for name in psa_names})).tolist())
y_max = max(np.max(sim_psa), np.max(obs_psa))
# duplicated data
sim_stations = sim_ims.station
obs_stations = obs_ims.station
del sim_ims
del obs_ims

for obs_idx, sim_idx in enumerate(argsearch(obs_stations, sim_stations)):
    if sim_idx is NOT_FOUND:
        # obs station not found in sim
        continue
    station = sim_stations[sim_idx]

    # plot data
    fig = plt.figure(figsize = (14, 7.5), dpi = 100)
    plt.rcParams.update({'font.size': 18})
    plt.loglog(psa_vals, sim_psa[sim_idx], color='red', \
               label='%s Sim' % (station))
    plt.loglog(psa_vals, obs_psa[obs_idx], color='black', \
               label='%s Obs' % (station))

    # plot formatting
    plt.legend(loc='best')
    plt.ylabel('Spectral acceleration (g)', fontsize=14)
    plt.xlabel('Vibration period, T (s)', fontsize=14)
    plt.title(args.run_name)
    plt.xlim([x_min, x_min * 10e4])
    plt.ylim([0.001, 5])
    plt.savefig(os.path.join(args.out_dir, 'pSA_comp_%s_vs_Period_%s_%s.png' \
                                           % (args.comp, args.run_name, station)))
    plt.close()
