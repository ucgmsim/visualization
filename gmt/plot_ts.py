#!/usr/bin/env python2
"""
"""

from argparse import ArgumentParser
from multiprocessing import Pool
import os
from shutil import rmtree
from tempfile import mkdtemp
from time import time

import qcore.geo as geo
import qcore.gmt as gmt
from qcore.xyts import XYTSFile

PAGE_WIDTH = 16
PAGE_HEIGHT = 9
# space around map for titles, tick labels and scales etc
MARGIN_TOP = 1.0
MARGIN_BOTTOM = 0.4
MARGIN_LEFT = 1.0
MARGIN_RIGHT = 1.7

parser = ArgumentParser()
arg = parser.add_argument
arg('xyts', help='path to xyts.e3d file')
arg('--cpt', help='xyts overlay cpt', default='hot')
arg('-r', '--dpi', help='dpi 80: 720p, [120]: 1080p, 240: 4k', \
    type=int, default=120)
arg('--title', help='main title', default='Automatically Generated Event')
arg('--subtitle1', help='top subtitle', \
    default='Automatically generated source model')
arg('--subtitle2', help='bottom subtitle', default='NZVM v1.65 h=<HH>km')
arg('--legend', help='colour scale legend text', default='ground motion (cm/s)')
arg('-n', '--nproc', help='number of processes to use', type=int, default=1)
arg('--borders', help='do not show map behind map margins', action='store_true')
#srf_cnrs
args = parser.parse_args()
assert(os.path.isfile(args.xyts))
assert(args.nproc > 0)
if args.nproc == 1:
    print('warning: only using 1 process, use more by setting nproc parameter')

#
gmt_temp = mkdtemp()
png_dir = os.path.join(gmt_temp, 'TS_PNG')
os.makedirs(png_dir)

map_width = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
map_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

xyts = XYTSFile(args.xyts)
# xyts derivatives
corners, cnr_str = xyts.corners(gmt_format=True)
ll_region = xyts.region(corners=corners)
# extend region to fill view window
map_width, map_height, ll_region = gmt.fill_space(map_width, map_height, \
                                                  ll_region, proj='M', \
                                                  dpi=args.dpi, wd=gmt_temp)
# region midpoint
ll_avg = sum(ll_region[:2]) / 2.0, sum(ll_region[2:]) / 2.0
# extend map to cover margins
if not args.borders:
    map_width_a, map_height_a, borderless_region = gmt.fill_margins( \
            ll_region, map_width, args.dpi, left=MARGIN_LEFT, \
            right=MARGIN_RIGHT, top=MARGIN_TOP, bottom=MARGIN_BOTTOM)

###
### PLOTTING STARTS HERE - TEMPLATE
###
######################################################

cpt_overlay = '%s/motion.cpt' % (gmt_temp)
template_bottom = '%s/bottom.ps' % (gmt_temp)
template_top = '%s/top.ps' % (gmt_temp)

###
### create resources that are used throughout the process
###
print('========== CREATING TEMPLATE ==========')
t0 = time()
# AUTOPARAMS - sites, TODO: allow setting major, all as arg
if ll_region[1] - ll_region[0] > 3:
    region_sites = gmt.sites_major
else:
    region_sites = gmt.sites.keys()
# AUTOPARAMS - tick labels, TODO: allow setting as arg
major_tick, minor_tick = gmt.auto_tick(ll_region[0], ll_region[1], map_width)
# AUTOPARAMS - overlay spacing
grd_dxy = '%sk' % (xyts.dx / 2.0)
# AUTOPARAMS - colour scale
pgv_path = '%s/PGV.bin' % (gmt_temp)
xyts.pgv(pgvout=pgv_path)
cpt_inc, cpt_max = gmt.xyv_cpt_range(pgv_path)[1:3]
# AUTOPARAMS - convergence limit
convergence_limit = cpt_inc * 0.2
# AUTOPARAMS - low cutoff
lowcut = cpt_max * 0.02
# AUTOPARAMS - title text generation
args.subtitle2 = args.subtitle2.replace('<HH>', str(xyts.hh))
# overlay colour scale
gmt.makecpt(args.cpt, cpt_overlay, 0, cpt_max, inc=cpt_inc, invert=True, \
            bg=None, fg=None)
# simulation area mask
geo.path_from_corners(corners=corners, min_edge_points=100, \
                      output='%s/sim.modelpath_hr' % (gmt_temp))
print('Created resources (%.2fs)' % (time() - t0))

def create_shared_pngs():
    t0 = time()
    b = gmt.GMTPlot(template_bottom)
    gmt.gmt_defaults(wd=gmt_temp, \
                     ps_media='Custom_%six%si' % (PAGE_WIDTH, PAGE_HEIGHT))
    if args.borders:
        b.background(PAGE_WIDTH, PAGE_HEIGHT, colour='white')
    else:
        b.spacial('M', borderless_region, sizing=map_width_a)
        # topo, water, overlay cpt scale
        b.basemap(topo_cpt='grey1')
        # map margins are semi-transparent
        b.background(map_width_a, map_height_a, \
                    colour='white@25', spacial=True, \
                    window=(MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM))
    # leave space for left tickmarks and bottom colour scale
    b.spacial('M', ll_region, sizing=map_width, \
            x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM)
    if args.borders:
        # topo, water, overlay cpt scale
        b.basemap(topo_cpt='grey1')
    # title, fault model and velocity model subtitles
    b.text(ll_avg[0], ll_region[3], args.title, size=20, dy=0.6)
    b.text(ll_region[0], ll_region[3], args.subtitle1, size=14, align='LB', dy=0.3)
    b.text(ll_region[0], ll_region[3], args.subtitle2, size=14, align='LB', dy=0.1)
    b.cpt_scale('R', 'B', cpt_overlay, cpt_inc, cpt_inc, label=args.legend, \
                length=map_height, horiz=False, pos='rel_out', align='LB', \
                thickness=0.3, dx=0.3, \
                arrow_f=cpt_max > 0, arrow_b=0 < 0)
    # stations - split into real and virtual
    #with open(stat_file, 'r') as sf:
    #    stations = sf.readlines()
    #stations_real = []
    #stations_virtual = []
    #for i in xrange(len(stations)):
    #    if len(stations[i].split()[-1]) == 7:
    #        stations_virtual.append(stations[i])
    #    else:
    #        stations_real.append(stations[i])
    #b.points(''.join(stations_real), is_file = False, \
    #        shape = 't', size = 0.08, fill = None, \
    #        line = 'white', line_thickness = 0.8)
    #b.points(''.join(stations_virtual), is_file = False, \
    #        shape = 'c', size = 0.02, fill = 'black', line = None)
    b.finalise()
    b.png(dpi=args.dpi, clip=False)
    print('bottom template completed in %.2fs' % (time() - t0))

    ###
    ### create map data which all maps will have on top
    ###
    t0 = time()
    t = gmt.GMTPlot(template_top, reset=False)
    t.spacial('M', ll_region, sizing=map_width, \
              x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM)
    # locations in NZ
    t.sites(region_sites)
    t.coastlines()
    # simulation domain
    t.path(cnr_str, is_file=False, split='-', close=True, width='0.4p', \
        colour='black')
    # fault file - creating direct from SRF is slower
    #t OK if only done in template - more reliable
    #t.fault(srf_cnrs, is_srf = False, plane_width = 0.5, \
    #        top_width = 1, hyp_width = 0.5)
    # ticks on top otherwise parts of map border may be drawn over
    t.ticks(major=major_tick, minor=minor_tick, sides='ws')
    t.finalise()
    t.png(dpi=args.dpi, clip=False)
    print('top template completed in %.2fs' % (time() - t0))

def render_slice(n):
    t0 = time()

    # process working directory
    swd = '%s/ts%.4d' % (gmt_temp, n)
    os.makedirs(swd)

    s = gmt.GMTPlot('%s/ts%.4d.ps' % (swd, n), reset=False)
    gmt.gmt_defaults(wd=swd, \
                     ps_media='Custom_%six%si' % (PAGE_WIDTH, PAGE_HEIGHT))
    s.spacial('M', ll_region, sizing=map_width, \
              x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM)

    # timestamp text
    s.text(ll_region[1], ll_region[3], 't=%.2fs' % (n * xyts.dt), \
           align='RB', size='14p', dy=0.1)
    # overlay
    xyts.tslice_get(n, comp=-1, outfile='%s/ts.bin' % (swd))
    s.clip('%s/sim.modelpath_hr' % (gmt_temp), is_file=True)
    s.overlay('%s/ts.bin' % (swd), cpt_overlay, dx=grd_dxy, dy=grd_dxy, \
              climit=convergence_limit, min_v=lowcut, \
              contours=cpt_inc, land_crop=True)
    s.clip()

    # add seismograms if wanted
    #if os.path.exists(os.path.abspath(tsplot.seis_data)):
    #    s.seismo(os.path.abspath(tsplot.seis_data), n, \
    #            fmt = tsplot.seis_fmt, \
    #            colour = tsplot.seis_colour, \
    #            width = tsplot.seis_line)

    # create PNG
    s.finalise()
    s.png(dpi=args.dpi, clip=False, out_dir=png_dir)
    # cleanup
    rmtree(swd)
    print('timeslice %.4d completed in %.2fs' % (n, time() - t0))

def combine_slice(n):
    """
    Sandwitch midde layer (time dependent) between basemap and top (labels etc).
    """
    png = '%s/ts%.4d.png' % (png_dir, n)
    gmt.overlay('%s/bottom.png' % (gmt_temp), png, png)
    gmt.overlay(png, '%s/top.png' % (gmt_temp), png)

###
### start rendering
###
ts0 = time()
pool = Pool(args.nproc)
# shared bottom and top layers
templates = pool.apply_async(create_shared_pngs, ())
# middle layers
pool.map(render_slice, xrange(xyts.t0, xyts.nt - xyts.t0))
# wait for bottom and top layers
print('waiting for templates to finish...')
templates.get()
print('templates finished, combining layers...')
# combine layers
pool.map(combine_slice, xrange(xyts.t0, xyts.nt - xyts.t0))
print('layers combined, creating animation...')
# images -> animation
gmt.make_movie('%s/ts%%04d.png' % (png_dir), \
               os.path.join('.', 'animation.m4v'), fps=20, codec='libx264')
print('finished.')
# cleanup
rmtree(gmt_temp)
