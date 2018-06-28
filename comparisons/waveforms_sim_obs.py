#!/usr/bin/env python2
"""
Plots 3 components for simulated and observed seismograms.

USAGE: run with -h parameter
"""

from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
import os

import numpy as np
import matplotlib.pyplot as plt

from qcore.timeseries import BBSeis, read_ascii

# files that contain the 3 components (text based)
# must be in same order as binary results (x, y, z)
extensions = ['.090', '.000', '.ver']

def load_args():
    """
    Process command line arguments.
    """

    # read
    parser = ArgumentParser()
    parser.add_argument('sim', help = 'path to binary file or text dir for simulated seismograms')
    parser.add_argument('obs', help = 'path to text dir for observed seismograms')
    parser.add_argument('out', help = 'output folder to place plots')
    parser.add_argument('--sim-prefix', default = '', \
                        help = 'sim text files are named <prefix>station.comp')
    parser.add_argument('--obs-prefix', default = '', \
                        help = 'obs text files are named <prefix>station.comp')
    parser.add_argument('-v', help = 'verbose messages', action = 'store_true')
    parser.add_argument('-n', '--nproc', help = 'number of processes to use', \
                        type = int, default = 1)
    args = parser.parse_args()

    # validate
    if os.path.isfile(args.sim):
        args.binary_sim = True
    elif os.path.isdir(args.sim):
        args.binary_sim = False
    else:
        raise ValueError('sim location not found')
    if args.v:
        print('sim data is binary: %r' % (bool(args.binary_sim)))
    assert(os.path.isdir(args.obs))
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    return args

def load_stations(args, sim_bb = None):
    """
    Determine stations available for plotting.
    returns numpy array of intersecting station names
    """
    def station_from_filename(path, n_prefix, n_suffix):
        return os.path.basename(path)[n_prefix : - n_suffix]

    # stations available in sim
    if args.binary_sim:
        sim_stations = sim_bb.stations.name
    else:
        sim_stations = glob(os.path.join(args.sim, '%s*%s' % (args.sim_prefix, \
                                                               extensions[0])))
        sim_stations = np.array([station_from_filename(s, len(args.sim_prefix), \
                                                       len(extensions[0])) \
                                 for s in sim_stations])

    # stations available in obs
    obs_stations = glob(os.path.join(args.obs, '%s*%s' % (args.obs_prefix, \
                                                           extensions[0])))
    obs_stations = [station_from_filename(o, len(args.obs_prefix), \
                                          len(extensions[0])) \
                    for o in obs_stations]

    # interested only if station available in both
    both = np.isin(sim_stations, obs_stations)
    if args.v:
        print('n_stations: %d sim, %d obs, intersection: %d' \
                % (sim_stations.size, len(obs_stations), np.sum(both)))
    return sim_stations[both]

def plot_station(args, name, sim_bb = None):
    if args.v:
        print('Plotting station: %s...' % (name))

    # load data
    def load_txt(folder, prefix):
        sim = [read_ascii(os.path.join(folder, \
                                       '%s%s%s' % (prefix, name, \
                                                   extensions[i])), meta = True) \
               for i in xrange(len(extensions))]
        return [s[0] for s in sim], \
               np.arange(sim[0][1]['nt']) * sim[0][1]['dt'] + sim[0][1]['sec']
    if args.binary_sim:
        sim_yx = sim_bb.vel(name).T, \
                 np.arange(sim_bb.nt) * sim_bb.dt + sim_bb.start_sec
    else:
        sim_yx = load_txt(args.sim, args.sim_prefix)
    obs_yx = load_txt(args.obs, args.obs_prefix)

    # get axis max, assume that within sim and obs, nt are equal
    y_min = min(np.min(sim_yx[0]), np.min(obs_yx[0]))
    y_max = max(np.max(sim_yx[0]), np.max(obs_yx[0]))
    all_y = np.concatenate((np.array(sim_yx[0], dtype = np.float32), \
                            np.array(obs_yx[0], dtype = np.float32)), axis = 1)
    pgvs = np.max(np.abs(all_y), axis = 1)
    ppgvs = np.max(all_y, axis = 1)
    npgvs = np.min(all_y, axis = 1)
    y_diff = y_max - y_min
    x_max = max(sim_yx[1][-1], obs_yx[1][-1])
    scale_length = max(int(round(x_max / 25.)) * 5, 5)

    # start plot
    colours = ['black', 'red']
    f, axis = plt.subplots(2, 3, sharex = True, sharey = True, \
                           figsize = (20, 4), dpi = 96)
    f.subplots_adjust(left = 0.08, bottom = 0.12, right = 0.98, top = None, \
                      wspace = 0.08, hspace = 0)
    plt.suptitle(name, fontsize = 20, x = 0.02, y = 0.5, \
                 horizontalalignment = 'left', verticalalignment = 'center')

    # subplots
    for i, s in enumerate([obs_yx, sim_yx]):
        for j in xrange(3):
            ax = axis[i, j]
            ax.set_axis_off()
            ax.plot(s[1], s[0][j] * min(y_max / ppgvs[j], y_min / npgvs[j]), \
                    color = colours[i], linewidth = 1)
            ax.set_ylim([y_min, y_max])
            if i and not j:
                ax.plot([0, scale_length], [y_min] * 2, color = 'black', \
                        linewidth = 2)
                ax.text(0, y_min - y_diff * 0.05, '0', size = 12, \
                        verticalalignment = 'top', \
                        horizontalalignment = 'center')
                ax.text(scale_length, y_min - y_diff * 0.05, \
                        str(scale_length), size = 12, \
                        verticalalignment = 'top', \
                        horizontalalignment = 'center')
                ax.text(scale_length / 2.0, y_min - y_diff * 0.125, \
                        'sec', size = 12, verticalalignment = 'top', \
                        horizontalalignment = 'center')
            elif not i:
                ax.set_title(extensions[j][1:], fontsize = 18)
                ax.text(s[1][-1], y_max, '%.1f' % (pgvs[j]), fontsize = 14)

    plt.savefig(os.path.join(args.out, '%s.png' % (name)))
    plt.close()

if __name__ == '__main__':
    args = load_args()

    sim_bb = None
    if args.binary_sim:
        sim_bb = BBSeis(args.sim)

    stations = load_stations(args, sim_bb)

    # multiprocessing or serial (debug friendly)
    if args.nproc > 1:
        def plot_station_star(params):
            return plot_station(*params)
        p = Pool(args.nproc)
        msgs = [(args, s, sim_bb) for s in stations]
        p.map(plot_station_star, msgs)
    else:
        [plot_station(args, s, sim_bb) for s in stations]
