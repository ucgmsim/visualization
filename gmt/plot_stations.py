#!/usr/bin/env python2

"""
Created: 4 January 2017
Purpose: Generate visualisations of obs/sim ratios, PGA, PGV, PSA.
Authors: Viktor Polak <viktor.polak@canterbury.ac.nz>

USAGE:
Execute with python: "$ ./plot_stations.py" or "$ python2 plot_stations.py"
First parameter is the file to plot.
Second parameter is optional override for output folder.

INPUT FORMAT:
File to plot must be in the following format:
Note numbers are lines of the file.
1. Plot Title (blank line for no title)
2. Legend Title for the colour palette scale
3. cpt source, station size
cpt source and station size can be followed by comma separated properties after a ':'
..::cpt source examples::..
hot:invert,t-40 will invert the hot palette and display with 40% transparency
hot:fg-black,bg-white set foreground and background colour
..::station size examples::..
0.2:shape-c will make the station size 0.2 with a circle shape
1k:g-nearneighbor will make grid spacing 1km and use the nearneighbor algorithm
4. min cpt, max cpt, cpt inc, cpt legend tick
all optionoal but cpt min and max must both be provided or none at all
optional parameters must be in order
5. number of data colums excluding longitude and latitude, optional label colour
6. comma separated column labels. placed inside top left corner of map.
Optional but must either be of length 0 or number of columns

7 - END. Data are longitude, latitude, col_1, col_2 ... col_N

ISSUES:
"""

from glob import glob
from multiprocessing import Pool
import os
from shutil import copy, rmtree
import sys
from tempfile import mkdtemp
from time import time, sleep

import numpy as np


import sys
sys.path.insert(0, '../../qcore/')
import qcore.gmt as gmt
import qcore.geo as geo
from qcore.shared import get_corners
from qcore.srf import srf2corners
from qcore.config import qconfig
CPT_DIR = os.path.join(qconfig['GMT_DATA'], 'cpt')

# process file header
def load_file(station_file):
    # load values
    val_pool = np.atleast_2d( \
            np.loadtxt(station_file, dtype = 'f', skiprows = 6)[:, 2:].T)
    ncol = val_pool.shape[0]

    with open(station_file) as statf:
        head = [next(statf).strip() for _ in xrange(6)]
        # 1st - title
        title = head[0]

        # 2nd line - legend title
        legend = head[1]

        # 3rd line - cpt description 1
        # src, point size, foreground colour, background colour
        cpt_info = head[2].split()
        cpt = cpt_info[0].split(':')[0]
        # default properties
        transparency = 0
        cpt_fg = None
        cpt_bg = None
        cpt_gap = ''
        cpt_topo = None
        cpt_overlays = 'black'
        if os.path.exists(cpt):
            # assuming it is a built in cpt if not matching filename
            cpt = os.path.abspath(cpt)
        try:
            # src:invert will add the 'invert' property to invert cpt
            cpt_properties = cpt_info[0].split(':')[1].split(',')
            for p in cpt_properties:
                if p[:2] == 't-':
                    transparency = p[2:]
                elif p[:3] == 'fg-':
                    cpt_fg = p[3:]
                elif p[:3] == 'bg-':
                    cpt_bg = p[3:]
                elif p[:4] == 'gap-':
                    cpt_gap = p[4:]
                elif p[:5] == 'topo-':
                    cpt_topo = p[5:]
                elif p[:9] == 'overlays-':
                    cpt_overlays = p[9:]
        except IndexError:
            cpt_properties = []
        if len(cpt_info) > 1:
            stat_size = cpt_info[1].split(':')[0]
            # also default search radius
            search = stat_size
            # stat size can be in km but surface search can only be in min|sec
            user_search = False
            # default properties
            shape = 't'
            grid = None
            grd_mask_dist = None
            landmask = False
            contours = False
            try:
                stat_properties = cpt_info[1].split(':')[1].split(',')
                for p in stat_properties:
                    if p[:6] == 'shape-':
                        shape = p[6]
                    elif p[:2] == 'g-':
                        grid = p[2:]
                    elif p[:4] == 'nns-':
                        search = p[4:]
                        user_search = True
                    elif p[:6] == 'gmask-':
                        grd_mask_dist = p[6:]
                    elif p[:8] == 'landmask':
                        landmask = True
                    elif p[:8] == 'contours':
                        contours = True
            except IndexError:
                stat_properties = []
        if not user_search and grid == 'surface':
            search = None

        # 4th line - cpt description 2
        # cpt_min, cpt_max, cpt_inc, cpt_tick
        cpt_info2 = head[3].split()
        if len(cpt_info2) > 1:
            usr_min, usr_max = map(float, cpt_info2[:2])
            cpt_min = [usr_min] * ncol
            cpt_max = [usr_max] * ncol
        else:
            cpt_min = []
            cpt_max = np.percentile(val_pool, 99.5, axis = 1)
            cpt_inc = []
            cpt_tick = []
            for i in xrange(len(cpt_max)):
                if cpt_max[i] > 115:
                    # 2 significant figures
                    cpt_max[i] = round(cpt_max[i], \
                            1 - int(np.floor(np.log10(abs(cpt_max[i])))))
                else:
                    # 1 significant figures
                    cpt_max[i] = round(cpt_max[i], \
                            - int(np.floor(np.log10(abs(cpt_max[i])))))
                if val_pool[i].min() < 0:
                    cpt_min.append(-cpt_max)
                else:
                    cpt_min.append(0)
                cpt_inc.append(cpt_max[i] / 10.)
                cpt_tick.append(cpt_inc[i] * 2.)
        if len(cpt_info2) > 2:
            cpt_inc = [float(cpt_info2[2])] * ncol
        if len(cpt_info2) > 3:
            cpt_tick = [float(cpt_info2[3])] * ncol

        # 5th line ncols and optional column label prefix
        col_info = head[4].split()
        ncol = int(col_info[0])
        if len(col_info) > 1:
            label_colour = col_info[1]
        else:
            label_colour = 'black'

        # 6th line - column labels
        col_labels = map(str.strip, head[5].split(','))
        if col_labels == ['']:
            col_labels = []
        if len(col_labels) != ncol and len(col_labels) != 0:
            print('%d column labels found when there are %d columns.' \
                    % (len(col_labels), ncol))
            exit(1)

    return {'title':title, 'legend':legend, 'stat_size':stat_size, \
            'search':search, 'shape':shape, 'grid':grid, \
            'grd_mask_dist':grd_mask_dist, 'cpt':cpt, 'cpt_fg':cpt_fg, \
            'cpt_bg':cpt_bg, 'cpt_min':cpt_min, 'cpt_max':cpt_max, \
            'cpt_inc':cpt_inc, 'cpt_tick':cpt_tick, 'cpt_properties':cpt_properties, \
            'transparency':transparency, 'ncol':ncol, 'cpt_gap':cpt_gap, \
            'label_colour':label_colour, 'col_labels':col_labels, \
            'cpt_topo':cpt_topo, 'landmask':landmask, 'overlays':cpt_overlays, \
            'contours':contours}

###
### boundaries
###
def determine_sizing(args, meta):
    # retrieve simulation domain if available
    if args.model_params is not None:
        corners, cnr_str = get_corners(args.model_params, gmt_format = True)
        meta['corners'] = corners
        meta['cnr_str'] = cnr_str

    if args.region is not None:
        x_min, x_max, y_min, y_max = args.region
    elif 'corners' in meta.keys():
        # path following sim domain curved on mercator like projections
        fine_path = geo.path_from_corners(corners = meta['corners'], output = None)
        # fit simulation region
        x_min = min([xy[0] for xy in fine_path])
        x_max = max([xy[0] for xy in fine_path])
        y_min = min([xy[1] for xy in fine_path])
        y_max = max([xy[1] for xy in fine_path])
    else:
        # fit all values
        xy = np.loadtxt(args.station_file, skiprows = 6, \
                        usecols = (0, 1), dtype = 'f')
        x_min, y_min = np.min(xy, axis = 0) - 0.1
        x_max, y_max = np.max(xy, axis = 0) + 0.1
    # combined region
    meta['region'] = (x_min, x_max, y_min, y_max)
    # avg lon/lat (midpoint of plotting region)
    meta['ll_avg'] = sum(meta['region'][:2]) / 2, sum(meta['region'][2:]) / 2

    # create masking if using grid overlay
    if meta['grid'] is not None:
        try:
            corners = meta['corners']
        except KeyError:
            corners = [[x_min, y_min], [x_max, y_min], \
                       [x_max, y_max], [x_min, y_max]]
        geo.path_from_corners(corners = corners, \
                              min_edge_points = 100, \
                              output = '%s/sim.modelpath_hr' % (meta['gmt_temp']))
        gmt.grd_mask('%s/sim.modelpath_hr' % (meta['gmt_temp']), \
                '%s/mask.grd' % (meta['gmt_temp']), dx = meta['stat_size'], \
                dy = meta['stat_size'], region = meta['region'])

    # work out an ideal tick increment (ticks per inch)
    # x axis is more constrainig
    if args.tick_major is None:
        args.tick_major, args.tick_minor = gmt.auto_tick(x_min, x_max, args.width)
    elif args.tick_minor is None:
        args.tick_minor = args.tick_major / 5.
    # cities/locations to plot
    if args.sites is 'none':
        args.sites = []
    elif args.sites == 'auto':
        if x_max - x_min > 3:
            args.sites = gmt.sites_major
        else:
            args.sites = gmt.sites.keys()
    elif args.sites == 'major':
        args.sites = gmt.sites_major
    elif args.sites == 'all':
        args.sites = gmt.sites.keys()

def template(args, meta):
    """
    Creates template (baselayer) file and prepares recources.
    """

    # incomplete template for common working GMT conf/history files
    t = gmt.GMTPlot('%s/template.ps' % (meta['gmt_temp']))
    # background can be larger as whitespace is later cropped
    t.background(11, 15)
    t.spacial('M', meta['region'], sizing = args.width, \
            x_shift = 1, y_shift = 2.5)

    # topo, water, overlay cpt scale (slow)
    if meta['title'] == 'DRAFT':
        t.basemap(road = None, highway = None, topo = None, res = 'f')
    else:
        if meta['cpt_topo'] is None:
            t.basemap()
        else:
            t.basemap(topo_cpt = meta['cpt_topo'])
    # simulation domain
    try:
        t.path(meta['cnr_str'], is_file = False, split = '-', \
                close = True, width = '0.4p', colour = 'black')
    except KeyError:
        pass
    t.leave()

    # fault path - boundaries already available
    if args.srf_cnrs is not None:
        copy(args.srf_cnrs, '%s/srf_cnrs.txt' % (meta['gmt_temp']))
    # fault path - determine boundaries (slow)
    elif args.srf is not None:
        srf2corners(args.srf, cnrs = '%s/srf_cnrs.txt' % (meta['gmt_temp']))

def column_overlay(args_meta_n):
    """
    Produces map for a column of data.
    """
    args, meta, n = args_meta_n

    # prepare resources in separate folder
    # prevents GMT IO errors on its conf/history files
    swd = '%s/c%.3dwd' % (meta['gmt_temp'], n)
    os.makedirs(swd)
    # name of slice postscript
    ps = '%s/c%.3d.ps' % (swd, n)

    # copy GMT setup and append top layer to blank file
    copy('%s/gmt.conf' % (meta['gmt_temp']), swd)
    copy('%s/gmt.history' % (meta['gmt_temp']), swd)
    p = gmt.GMTPlot(ps, append = True)

    if meta['cpt'].split('/')[0] == '<REPO>':
        meta['cpt'] = os.path.join('%s/%s' % (CPT_DIR, meta['cpt'][7:]))
    if 'fixed' in meta['cpt_properties']:
        cpt_stations = meta['cpt']
    else:
        # prepare cpt
        cpt_stations = '%s/stations.cpt' % (swd)
        # overlay colour scale
        gmt.makecpt(meta['cpt'], cpt_stations, meta['cpt_min'][n], \
                meta['cpt_max'][n], inc = meta['cpt_inc'][n], \
                invert = 'invert' in meta['cpt_properties'], \
                fg = meta['cpt_fg'], bg = meta['cpt_bg'], \
                transparency = meta['transparency'], wd = swd)

    # common title
    if len(meta['title']):
        p.text(meta['ll_avg'][0], meta['region'][3], meta['title'], \
                colour = 'black', align = 'CB', size = 28, dy = 0.2)

    # add ratios to map
    mask = '%s/mask.grd' % (meta['gmt_temp'])
    if meta['grid'] == None:
        if meta['landmask']:
            # start clip - TODO: allow different resolutions including GSHHG
            p.clip(path = gmt.LINZ_COAST['150k'], is_file = True)
        p.points(args.station_file, shape = meta['shape'], \
                size = meta['stat_size'], fill = None, line = None, \
                cpt = cpt_stations, cols = '0,1,%d' % (n + 2), header = 6)
        if meta['landmask']:
            # apply clip to intermediate items
            p.clip()
    else:
        grd_file = '%s/overlay.grd' % (swd)
        if meta['grd_mask_dist'] != None:
            col_mask = '%s/column_mask.grd' % (swd)
            mask = col_mask
        else:
            col_mask = None
        if meta['grid'] == 'surface':
            station_tmp = '%s/blocked.xyz' % (swd)
            gmt.table2block(args.station_file, station_tmp, header = 6, \
                    dx = meta['stat_size'], region = meta['region'], wd = swd, \
                    cols = '0,1,%d' % (n + 2))
            args.station_file = station_tmp
            n_header = 1
            cols = None
        else:
            n_header = 6
            cols = '0,1,%d' % (n + 2)
        gmt.table2grd(args.station_file, grd_file, file_input = True, \
                grd_type = meta['grid'], region = meta['region'], \
                dx = meta['stat_size'], climit = meta['cpt_inc'][n] * 0.5, \
                wd = swd, geo = True, sectors = 4, min_sectors = 1, \
                search = meta['search'], cols = cols, \
                header = n_header, automask = col_mask, \
                mask_dist = meta['grd_mask_dist'])
        p.clip(path = '%s/sim.modelpath_hr' % (meta['gmt_temp']), is_file = True)
        if meta['landmask']:
            # start clip - TODO: allow different resolutions including GSHHG
            p.clip(path = gmt.LINZ_COAST['150k'], is_file = True)
        p.overlay(grd_file, cpt_stations, dx = meta['stat_size'], \
                dy = meta['stat_size'], land_crop = False, \
                transparency = meta['transparency'])
        if meta['contours']:
            # use set increments if we have scaled the CPT
            #if cpt_stations == '%s/stations.cpt' % (swd):
            #    interval = meta['cpt_tick'][n]
            #else:
            interval = cpt_stations
            p.contours(grd_file, interval = interval)
        # apply clip to intermediate items
        p.clip()
    # add locations to map
    p.sites(args.sites)

    # title for this data column
    if len(meta['col_labels']):
        p.text(meta['region'][0], meta['region'][3], \
                meta['col_labels'][n], colour = meta['label_colour'], \
                align = 'LB', size = '18p', dx = 0.2, dy = -0.35)

    # ticks on top otherwise parts of map border may be drawn over
    p.ticks(major = args.tick_major, minor = args.tick_minor, sides = 'ws')

    # colour scale
    if 'categorical' in meta['cpt_properties']:
        p.cpt_scale(3, -0.5, cpt_stations, label = meta['legend'], \
                arrow_f = False, arrow_b = False, gap = meta['cpt_gap'], \
                intervals = 'intervals' in meta['cpt_properties'], \
                categorical = True)
    else:
        p.cpt_scale(3, -0.5, cpt_stations, meta['cpt_tick'][n], \
                meta['cpt_inc'][n], label = meta['legend'], \
                arrow_f = meta['cpt_max'][n] > 0, arrow_b = meta['cpt_min'][n] < 0)

    # fault planes
    if os.path.exists('%s/srf_cnrs.txt' % (meta['gmt_temp'])):
        p.fault('%s/srf_cnrs.txt' % (meta['gmt_temp']), is_srf = False, \
                plane_width = 0.5, top_width = 1, hyp_width = 0.5, \
                top_colour = meta['overlays'], \
                plane_colour = meta['overlays'], hyp_colour = meta['overlays'])
    p.leave()

    ###
    ### save as png
    ###

    # have to combine multiple postscript layers
    bottom = '%s/template.ps' % (meta['gmt_temp'])
    top = '%s%sc%.3d.ps' % (swd, os.sep, n)
    combined = '%s/final.ps' % (swd)
    copy(bottom, combined)
    with open(combined, 'a') as cf:
        with open(top, 'r') as tf:
            cf.write(tf.read())
    p = gmt.GMTPlot(combined, append = True)
    # actual rendering (slow)
    p.finalise()
    p.png(dpi = args.dpi, clip = True, out_name = \
            os.path.join(args.out_dir, \
                         os.path.splitext(os.path.basename(top))[0]))

def load_args():
    """
    Load command line arguments.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('station_file', help = 'data input file')
    parser.add_argument('--out_dir', help = 'folder where outputs are saved', \
                        default = 'PNG_stations')
    parser.add_argument('--srf', \
            help = 'srf file, will use corners file instead if available')
    parser.add_argument('--srf_cnrs', help = 'standard srf fault corners file')
    parser.add_argument('--model_params', \
            help = 'model_params file for simulation domain')
    parser.add_argument('--region', help = 'plot region. xmin xmax ymin ymax.', \
                        type = float, nargs = 4)
    parser.add_argument('-n', '--nproc', help = 'number of processes to run', \
            type = int, default = int(os.sysconf('SC_NPROCESSORS_ONLN')))
    parser.add_argument('--width', help = 'map width in default units', \
                        type = float, default = 6.0)
    parser.add_argument('--tick_major', help = 'major map tick increment', \
                        type = float)
    parser.add_argument('--tick_minor', help = 'minor map tick increment', \
                        type = float)
    parser.add_argument('--sites', help = 'locations to label', \
                        default = 'auto')
    parser.add_argument('--dpi', help = 'output resolution', \
                        type = int, default = 300)
    args = parser.parse_args()

    args.station_file = os.path.abspath(args.station_file)
    assert(os.path.exists(args.station_file))
    args.out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return args

###
### MASTER
###
if __name__ == '__main__':
    # command line arguments
    args = load_args()
    pool = Pool(args.nproc)
    msgs = []

    # station file metadata
    meta = load_file(args.station_file)
    meta['gmt_temp'] = mkdtemp()
    # calculate other parameters
    determine_sizing(args, meta)
    # start plot
    template(args, meta)
    # finish plot per column
    if args.nproc > 1:
        msgs.extend([(args, meta, i) for i in xrange(meta['ncol'])])
        pool.map(column_overlay, msgs)
    else:
        # debug friendly version
        [column_overlay((args, meta, i)) for i in xrange(meta['ncol'])]

    # clear all working files
    rmtree(meta['gmt_temp'])
