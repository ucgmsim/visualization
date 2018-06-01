#!/usr/bin/env python2
"""
plot_items.py plots items given as parameters.

add srf files from 2 locations:
-s "/folder1/*/Srf/*.srf" -s "location2/*.srf"

add vm corners created with srfinfo2vm:
-v "autovm/*/VeloModCorners.txt"

for more information on parameters, use -h
"""
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
import os
from shutil import rmtree
import sys
from tempfile import mkdtemp

from h5py import File as h5open
import numpy as np

from qcore import geo
from qcore import gmt
from qcore import srf

script_dir = os.path.abspath(os.path.dirname(__file__))

# parameters
parser = ArgumentParser()
arg = parser.add_argument
arg('-t', '--title', help = 'title text', default = '')
arg('-f', '--filename', default = 'plot_items', \
    help = 'output filename excluding extention')
arg('-s', '--srf-files', action = 'append', \
    help = 'SRF files to plot, use wildcards, repeat as needed')
arg('--slip-max', help = 'maximum slip (cm/s) on colour scale', \
    type = float, default = 1000.0)
arg('-v', '--vm-corners', action = 'append', \
    help = 'VeloModCorners.txt to plot, use wildcards, repeat as needed')
arg('-b', '--bb-scale', help = 'beachball scale', type = float, default = 0.05)
arg('-n', '--nproc', help = 'max number of processes', type = int, default = 1)
arg('-d', '--dpi', help = 'render DPI', type = int, default = 300)
args = parser.parse_args()

# gather resources
srf_files = []
vm_corners = []
if args.srf_files is not None:
    for ex in args.srf_files:
        srf_files.extend(glob(ex))
if args.vm_corners is not None:
    for ex in args.vm_corners:
        vm_corners.extend(glob(ex))
if len(srf_files) == 0 and len(vm_corners) == 0:
    sys.exit('nothing found to plot')

gmt_temp = mkdtemp()
print gmt_temp

# slip cpt
slip_cpt = '%s/slip.cpt' % (gmt_temp)
gmt.makecpt(gmt.CPTS['slip'], slip_cpt, 0, args.slip_max)
# load srf files
def load_srf(i_srf):
    # point source - save beachball data
    if not srf.is_ff(i_srf[1]):
        info = '%s.info' % os.path.splitext(i_srf[1])[0]
        if not os.path.exists(info):
            print('ps SRF missing .info, using 5.0 for magnitude: %s' \
                  % (i_srf[1]))
            mag = 5.0
            hypocentre = srf.get_hypo(i_srf[1], depth = True)
            strike, dip, rake = srf.ps_params(i_srf[1])
        else:
            with h5open(info) as h:
                mag = h.attrs['mag']
                hypocentre = h.attrs['hlon'], h.attrs['hlat'], h.attrs['cd']
                strike = h.attrs['strike'][0]
                dip = h.attrs['dip'][0]
                rake = h.attrs['rake']
        with open('%s/beachball%d.bb' % (gmt_temp, i_srf[0]), 'w') as bb:
            bb.write('%s %s %s %s %s %s %s %s %s\n' % \
                     (hypocentre[0], hypocentre[1], hypocentre[2], \
                      strike, dip, rake, mag, hypocentre[0], hypocentre[1]))
        return
    # finite fault - save outline and slip distributions
    srf.srf2corners(i_srf[1], cnrs = '%s/srf%d.cnrs' % (gmt_temp, i_srf[0]))
    proc_tmp = '%s/srf2map_%d' % (gmt_temp, i_srf[0])
    os.makedirs(proc_tmp)
    try:
        srf_data = gmt.srf2map(i_srf[1], gmt_temp, \
                               prefix = 'srf%d' % (i_srf[0]), wd = proc_tmp)
        return i_srf[0], srf_data
    except ValueError:
        # vertical dip
        return
if len(srf_files) > 0:
    pool = Pool(args.nproc)
    i_srf_data = pool.map(load_srf, zip(range(len(srf_files)), srf_files))

# load vm corners
vm_corners = '\n>\n'.join(['\n'.join([' '.join(map(str, v)) for v in \
                np.loadtxt(c, skiprows = 2, dtype = np.float32).tolist()]) \
                for c in vm_corners])


ps_file = '%s/plot_items.ps' % (gmt_temp)
map_width = 9
p = gmt.GMTPlot(ps_file)
p.spacial('M', gmt.nz_region, sizing = '%si' % (map_width), x_shift = 2, y_shift = 2)
p.basemap(topo_cpt = 'grey1', land = 'lightgray', topo = None, road = None, highway = None)

# plot velocity model corners
p.path(vm_corners, is_file = False, close = True, width = '0.5p', split = '-')

# loop through srfs and planes
for i_s in i_srf_data:
    if i_s is None:
        continue
    for plane in xrange(len(i_s[1][1])):
        p.overlay('%s/srf%d_%d_slip.bin' % (gmt_temp, i_s[0], plane), slip_cpt, \
                dx = i_s[1][0][0], dy = i_s[1][0][1], climit = 2, \
                crop_grd = '%s/srf%d_%d_mask.grd' % (gmt_temp, i_s[0], plane), \
                land_crop = False, transparency = 35, \
                custom_region = i_s[1][1][plane])
for c in glob('%s/srf*.cnrs' % (gmt_temp)):
    p.fault(c, is_srf = False, \
            hyp_size = 0, plane_width = 0.2, top_width = 0.4, \
            hyp_width = 0.2, plane_colour = 'blue', \
            top_colour = 'blue', hyp_colour = 'blue')
for bb in glob('%s/beachball*.bb' % (gmt_temp)):
    p.beachballs(bb, is_file = True, fmt = 'a', scale = args.bb_scale)

# border tick labels
p.ticks(major = 2, minor = 0.2)
# QuakeCoRE logo
p.image('L', 'T', '%s/quakecore-logo.png' % (script_dir), \
        width = '3i', pos = 'rel')
# slip scale
p.cpt_scale('C', 'B', slip_cpt, pos = 'rel_out', dy = '0.5i', \
        label = 'Slip (cm)', length = map_width * 0.618)
# output
p.finalise()
p.png(out_name = os.path.abspath(args.filename), dpi = args.dpi, \
      background = 'white')
rmtree(gmt_temp)
