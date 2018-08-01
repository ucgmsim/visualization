#!/usr/bin/env python2
"""
IM vs RRUP plot - basic edition

Must have only exactly 2 IM inputs: sim, obs (in that order)
"""

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np


import sys
sys.path.insert(0, '../../qcore/')
from qcore.formats import load_im_file
from qcore.nputil import argsearch

#from qcore.formats import load_im_file
#from qcore.nputil import argsearch

from collections import Counter

colours = ['red', [0, 0.5, 0]]
labels = ['Physics-based', 'Observed']
markers = ['o', '+']
edges = [None, 5]


def load_args():
    """
    Process command line arguments.
    """

    # read
    parser = ArgumentParser()
    parser.add_argument('rrup', help = 'path to RRUP file')
    parser.add_argument('sim', help='path to SIMULATED IM file')
    parser.add_argument('obs', help='path to OBSERVED IM file')
    parser.add_argument('--emp', help='path to empirical IM file')
    parser.add_argument('--out-dir', help = 'output folder to place plot', \
                        default = '.')
    parser.add_argument('--run-name', help = 'run_name - should automate?', \
                        default = 'event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm')
    parser.add_argument('--comp', help = 'component', default = 'geom')
    args = parser.parse_args()

    # validate
    assert (os.path.isfile(args.obs))
    assert (os.path.isfile(args.sim))
    # TODO: currently fixed as 'sim', 'obs'
    assert(os.path.isfile(args.rrup))
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args


def get_print_name(im, comp):
    if im.startswith('pSA_'):
        im = 'pSA(%dp%s' % (float(im.split('_')[-1]), im.split('.')[-1])
        im = '%s)' % (im.rstrip('p0'))
    return '%s_comp_%s' % (im, comp)

###
### MAIN
###

args = load_args()
name_rrup = np.loadtxt(args.rrup, dtype='|S7,f', \
                       usecols=(0,3), skiprows=1, delimiter=',')

# load im files (slow) for component, available pSA columns
sim_ims = load_im_file(args.sim, comp=args.comp)
obs_ims = load_im_file(args.obs, comp=args.comp)

print(type(sim_ims))
im_names = np.intersect1d(obs_ims.dtype.names[2:], sim_ims.dtype.names[2:])
print(im_names)

os_idx = argsearch(obs_ims.station, sim_ims.station)
ls_idx = argsearch(name_rrup['f0'], sim_ims.station)
os_idx.mask += np.isin(os_idx, ls_idx.compressed(), invert=True)
obs_idx = np.where(os_idx.mask == False)[0]
sim_idx = os_idx.compressed()
stations = obs_ims.station[obs_idx]
name_rrup = name_rrup[argsearch(stations, name_rrup['f0']).compressed()]
rrups = name_rrup['f1']

for im in im_names:
    print_name = get_print_name(im, args.comp)
    fig = plt.figure(figsize=(14, 7.5), dpi=100)
    plt.rcParams.update({'font.size': 18})
    sim_ys = sim_ims[im][sim_idx]
    obs_ys = obs_ims[im][obs_idx]
    plt.loglog(rrups, sim_ys, linestyle='None', color=colours[0],
               marker=markers[0], markeredgewidth=edges[0], markersize=10, \
               markeredgecolor=colours[0], label=labels[0])
    plt.loglog(rrups, obs_ys, linestyle='None', color=colours[1],
               marker=markers[1], markeredgewidth=edges[1], markersize=10, \
               markeredgecolor=colours[1], label=labels[1])


    # plot formatting
    plt.legend(loc='best', fontsize=9, numpoints=1)
    plt.ylabel(print_name)
    plt.xlabel('Source-to-site distance, $R_{rup}$ (km)')
    plt.minorticks_on()
    plt.title(args.run_name, fontsize=12)
    ymax = max(np.max(sim_ys), np.max(obs_ys))
    ymin = min(np.min(sim_ys), np.min(obs_ys))
    xmax = np.max(rrups)
    xmin = np.min(rrups)
    print(ymax)
    plt.ylim(ymax=ymax * 1.27)
    # plt.xlim(1e-1, 1e2)
    fig.set_tight_layout(True)
    plt.savefig(os.path.join(args.out_dir, '%s_with_Rrup_%s.png' \
                                           % (print_name, args.run_name)))
    plt.close()