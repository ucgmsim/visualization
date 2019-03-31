#!/usr/bin/env python2
"""
IM vs RRUP plot

To see help message:
python im_rrup.py -h

Sample command:
python im_rrup.py ~/darfield_obs/rrups.csv  ~/darfield_sim/darfield_sim.csv ~/darfield_obs/darfield_obs.csv --config ~/Empirical_Engine/model_config.yaml --srf /nesi/projects/nesi00213/dev/impp_datasets/Darfield/source.info --out_dir darfield_emp_new_rrup4 --run_name 20100904_Darfield_m7p1_201705011613
"""

import matplotlib as mpl
mpl.use('Agg')

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np

from qcore.formats import load_im_file
from qcore.nputil import argsearch
from qcore.utils import setup_dir

from empirical.scripts import calculate_empirical
from empirical.util import empirical_factory
from empirical.GMM_models import classdef


def load_args():
    """
    Process command line arguments.
    """
    # read
    parser = ArgumentParser()
    parser.add_argument('rrup', help='path to RRUP file')
    parser.add_argument('sim', help='path to SIMULATED IM file')
    parser.add_argument('obs', help='path to OBSERVED IM file')
    parser.add_argument('--config', help='path to .yaml empirical config file')
    parser.add_argument('--srf', help='path to srf info file')
    parser.add_argument('--dist_min', default=0.1, type=float, help='GMPE param DistMin, default 0.1 km')
    parser.add_argument('--dist_max', default=100.0, type=float, help='GMPE param DistMax, default 100.0 km')
    parser.add_argument('--n_val', default=51.0, type=float, help='GMPE param n_val, default 51.0')
    parser.add_argument('--out_dir', help='output folder to place plot', default='.')
    parser.add_argument('--run_name', help='run_name - should automate?', default='event-yyyymmdd_location_mMpM_sim-yyyymmddhhmm')
    parser.add_argument('--comp', help='component', default='geom')
    args = parser.parse_args()

    validate_args(args)

    setup_dir(args.out_dir)

    return args


def get_print_name(im, comp):
    if im.startswith('pSA_'):
        im = 'pSA(%dp%s' % (float(im.split('_')[-1]), im.split('.')[-1])
        im = '%s)' % (im.rstrip('p0'))
    return '%s_comp_%s' % (im, comp)


def validate_args(args):
    """
       validates all input args;
       config arg exists if and only if srf arg exists
    """
    assert os.path.isfile(args.rrup)
    assert os.path.isfile(args.obs)
    assert os.path.isfile(args.sim)

    if args.srf is not None:
        assert os.path.isfile(args.srf)
        if args.config is not None:
            assert os.path.isfile(args.config)
    else:
        if args.config is not None:
            sys.exit("Please also provide the path to an srf info file for empirical calculation.")


def get_empirical_values(fault, im, model_dict, r_rup_vals, period):
    gmm = empirical_factory.determine_gmm(fault, im, model_dict)
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

###
### MAIN
###

args = load_args()
name_rrup = np.loadtxt(args.rrup, dtype='|S7,f', usecols=(0, 3), skiprows=1, delimiter=',')

# load im files (slow) for component, available pSA columns
sim_ims = load_im_file(args.sim, comp=args.comp)
obs_ims = load_im_file(args.obs, comp=args.comp)

im_names = np.intersect1d(obs_ims.dtype.names[2:], sim_ims.dtype.names[2:])

os_idx = argsearch(obs_ims.station, sim_ims.station)
ls_idx = argsearch(name_rrup['f0'], sim_ims.station)
os_idx.mask += np.isin(os_idx, ls_idx.compressed(), invert=True)
obs_idx = np.where(os_idx.mask == False)[0]
sim_idx = os_idx.compressed()
stations = obs_ims.station[obs_idx]
name_rrup = name_rrup[argsearch(stations, name_rrup['f0']).compressed()]
rrups = name_rrup['f1']

# empirical calc
if args.srf is not None:
    model_dict = empirical_factory.read_model_dict(args.config)
    fault = calculate_empirical.create_fault_parameters(args.srf)
    r_rup_vals = np.exp(np.linspace(np.log(args.dist_min), np.log(args.dist_max), args.n_val))

# plot
for im in im_names:
    print_name = get_print_name(im, args.comp)
    fig = plt.figure(figsize=(14, 7.5), dpi=100)
    plt.rcParams.update({'font.size': 18})
    sim_ys = sim_ims[im][sim_idx]
    obs_ys = obs_ims[im][obs_idx]

    # sim obs plots
    plt.loglog(rrups, sim_ys, linestyle='None', color='red',
               marker='o', markeredgewidth=None, markersize=10, markeredgecolor='red', label='Physics-based')
    plt.loglog(rrups, obs_ys, linestyle='None', color='green',
               marker='+', markeredgewidth=5, markersize=10, markeredgecolor='green', label='Observed')

    # emp plot
    if args.srf is not None:
        if 'pSA' in im:
            im, p = im.split('_')
            period = [float(p)]
        else:
            period = None

        e_medians, e_sigmas = get_empirical_values(fault, im, model_dict, r_rup_vals, period)

        if np.size(e_medians) != 0:  # MMI does not have emp
            plt.loglog(r_rup_vals, e_medians, color='black', marker=None, linewidth=3, label='Empirical')
            plt.loglog(r_rup_vals, e_medians * np.exp(-e_sigmas), color='black', marker=None, linestyle='dashed',
                       linewidth=3)
            plt.loglog(r_rup_vals, e_medians * np.exp(e_sigmas[:]), color='black', marker=None, linestyle='dashed',
                       linewidth=3)

    # plot formatting
    plt.legend(loc='best', fontsize=9, numpoints=1)
    plt.ylabel(print_name)
    plt.xlabel('Source-to-site distance, $R_{rup}$ (km)')
    plt.minorticks_on()
    plt.title(args.run_name, fontsize=12)
    ymax = max(np.max(sim_ys), np.max(obs_ys))
    ymin = min(np.min(sim_ys), np.min(obs_ys))
    plt.ylim(ymax=ymax * 1.27)
    plt.xlim(1e-1, 1e2)
    fig.set_tight_layout(True)
    plt.savefig(os.path.join(args.out_dir, '%s_with_Rrup_%s.png'% (print_name, args.run_name)))
    plt.close()
