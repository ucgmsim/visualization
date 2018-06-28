#!/usr/bin/env python2
"""
IM vs RRUP plot - basic edition

Must have only exactly 2 IM inputs: sim, obs (in that order)
"""

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np

from qcore.formats import load_im_file
from qcore.nputil import argsearch

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
    parser.add_argument('--out-dir', help = 'output folder to place plot', \
                        default = '.')
    # TODO: automatically retrieved default
    parser.add_argument('--run-name', help = 'run_name - should automate?', \
                        default = 'event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm')
    parser.add_argument('--im', help = 'path to IM file, repeat as necessary', \
                        action = 'append')
    parser.add_argument('--comp', help = 'component', default = 'geom')
    args = parser.parse_args()

    # validate
    assert(args.im is not None)
    # TODO: currently fixed as 'sim', 'obs'
    assert(len(args.im) == 2)
    assert(min([os.path.isfile(im_file) for im_file in args.im]))
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
rrups = np.loadtxt(args.rrup, dtype = [('station', '|S7'), ('rrup', 'f')], \
                    delimiter = ',', skiprows = 1, usecols = (0, 3))
im_data_list = [load_im_file(im_csv) for im_csv in args.im]

for im_col in im_data_list[0].dtype.names[2:]:
    print_name = get_print_name(im_col, args.comp)

    # plot data
    fig = plt.figure(figsize = (14, 7.5), dpi = 100)
    plt.rcParams.update({'font.size': 18})
    for i, im_data in enumerate(im_data_list):
        im_data = im_data[im_data.component == args.comp]
        r_rups = rrups['rrup'][argsearch(im_data.station, rrups['station'])]
        plt.loglog(r_rups, im_data[im_col], linestyle='None', color=colours[i],
                    marker=markers[i], markeredgewidth=edges[i], markersize=10, \
                    markeredgecolor=colours[i], label=labels[i])

    # plot formatting
    plt.legend(loc='best', fontsize=9)
    plt.ylabel(print_name)
    plt.xlabel('Source-to-site distance, $R_{rup}$ (km)')
    plt.minorticks_on()
    plt.title(args.run_name, fontsize=12)
    fig.set_tight_layout(True)
    plt.savefig(os.path.join(args.out_dir, '%s_with_Rrup_%s.png' \
                                           % (print_name, args.run_name)))
    plt.close()
