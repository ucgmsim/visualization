#!/usr/bin/env python2

from argparse import ArgumentParser
from math import ceil, floor, log10, log
import os
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from qcore import gmt
from qcore import srf

parser = ArgumentParser()
arg = parser.add_argument
arg('srf_file', help='path to srf file')
arg('--out-dir', help='save in a different dir')
arg('--title', help='give a plot title')
arg('--rake-spacing', help='set rake arrow spacing (subfaults)', type=float)
arg('--rake-scale', help='maximum length of rake arrows', type=float, default=0.4)
arg('--rake-average', help='use averaging in rake arrow area', action='store_true')
args = parser.parse_args()

if args.out_dir is None:
    args.out_dir = os.path.dirname(args.srf_file)
if args.out_dir == '':
    args.out_dir = '.'
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

gmt_temp = mkdtemp(prefix='_GMT_WD_SRF_')

# modified base colour palettes
plot_cpt = [gmt.CPTS['slip'], gmt.CPTS['trise']]

labels = ['Slip (cm)', 'Rise Time (s)', 'Rake (deg)']

# retrieve hypocentre information
hyp_seg, hyp_s, hyp_d = srf.get_hypo(args.srf_file, lonlat=False)
with open(args.srf_file, 'r') as s:
    planes = srf.read_header(s, idx=True)
seg_len = [plane['length'] for plane in planes]
seg_wid = [plane['width'] for plane in planes]
seg_sub = [plane['dhyp'] < 0 for plane in planes]
# planes are plotted on a row, slip tinit and rake on 3 rows
ncol = len(planes) - sum(seg_sub)
nrow = 3
# use optimal grid resolution to prevent interpolation or grid artifacts
srf_dx, srf_dy = srf.srf_dxy(args.srf_file)

###
### NOMINAL KM / INCH PAPER
###
# points per inch (for fonts)
PTPI = 72.
# width x height of PS_MEDIA defined above
media = [11.7, 8.3]
# top space for title
top_margin = 0.65
if args.title is None:
    top_margin = 0.4
# bottom space for tick labels
bottom_margin = 0.55
# left space for scale, margin
left_margin = 0.65
# right margin for colour scales
right_margin = 1.2
# vertical space between plots (leave room for headings)
row_spacing = 0.2
# horizontal space between different planes
col_spacing = 0.2

def kminch_scale():
    # max Y axis plot space per plot after all margins
    plot_y = (media[1] - (nrow - 1) * row_spacing \
            - top_margin - bottom_margin) / nrow
    # kilometres per unit (inch) of paper
    # based on Y axis limitations
    km_inch = max(seg_wid) / plot_y
    # required X axis space considering above
    plot_x_total = left_margin + (ncol - 1) * col_spacing \
            + sum(seg_len) / km_inch + right_margin
    # check if enough X axis space
    if plot_x_total > media[0]:
        # max X axis plot space
        plot_x = media[0] - plot_x_total + sum(seg_len) / km_inch
        # recalculate based on more limited X axis limitations
        km_inch = sum(seg_len) / plot_x
        # amount of space on edges
        x_space = 0
    else:
        x_space = media[0] - plot_x_total
    return km_inch, x_space
km_inch, x_space = kminch_scale()

# font scaling
# XXX: using km_inch not best metric.
# TODO: use something like min or avg region width instead
scale_factor = 1. / max(log(km_inch), 2)
base_size = scale_factor * 24
annot_size = 4. / 5. * base_size
label_size = 4. / 3. * base_size
# then re-adjust spacing dependent on fonts
# 0.04 should be major tick length
row_spacing = annot_size / PTPI * 2 + 0.04 * 2
col_spacing = annot_size / PTPI * 2 + 0.04 * 2
base_gap = base_size / PTPI * 1.2
annot_gap = annot_size / PTPI * 1.2
if args.title is None:
    top_margin = 0.04 + annot_gap
# between lines, below x tick labels, bottom margin update
base_gap = base_size / PTPI * 1.2
x_text_gap = 0.04 * 2 + base_size / PTPI * 1.5
bottom_margin = x_text_gap + 3 * base_gap
# repeat above for y labels
y_text_gap = 0.04 * 2 + base_gap * 2
left_margin = y_text_gap + base_gap
# finally adjust km_inch slightly as needed
km_inch, x_space = kminch_scale()

# automatic rake arrow decimation
if args.rake_spacing is None:
    subfault_size = planes[0]['length'] / planes[0]['nstrike'] / km_inch
    args.rake_spacing = int(round(0.5 / subfault_size * scale_factor ** 2))
    if args.rake_spacing < 1:
        args.rake_spacing = 1

# pre-calculate distances using result km_inch
seg_wid_d = [seg / km_inch for seg in seg_wid]
seg_len_d = [seg / km_inch for seg in seg_len]
# ideally spaced tick labels
if max(seg_len) < 100:
    tick_factor = 5
else:
    tick_factor = 3
major_tick = 0.05
tick_helper = 0
while km_inch / major_tick > tick_factor:
    if tick_helper != 2:
        major_tick *= 2
        tick_helper += 1
    else:
        major_tick *= 2.5
        tick_helper = 0

# start plot
p = gmt.GMTPlot('%s/%s_square.ps' % (gmt_temp, \
        os.path.splitext(os.path.basename(args.srf_file))[0]))
# override GMT defaults
gmt.gmt_defaults(wd = gmt_temp, font_label = label_size, \
        map_tick_length_primary = '0.03i', ps_media = 'A4', \
        font_annot_primary = base_size, ps_page_orientation = 'landscape', \
        map_frame_pen = '%sp,black' % (scale_factor * 1.5))
p.background(media[0], media[1])

# prepare data and CPTs
slips = srf.srf2llv_py(args.srf_file, seg = -1, value = 'slip', lonlat = False)
tinits = srf.srf2llv_py(args.srf_file, seg = -1, value = 'tinit', lonlat = False)
trises = srf.srf2llv_py(args.srf_file, seg = -1, value = 'trise', lonlat = False)
rakes = srf.srf2llv_py(args.srf_file, seg = -1, value = 'rake', \
        flip_rake = True, lonlat = False)
slip_values = np.concatenate((slips))[:, 2]
trise_values = np.concatenate((trises))[:, 2]
tinit_max = max(np.concatenate((tinits))[:, 2])
acontour = 0.5
if tinit_max > 20:
    acontour = 2
elif tinit_max > 10:
    acontour = 1
cpt_out = ['%s/slip.cpt' % (gmt_temp), \
        '%s/trise.cpt' % (gmt_temp)]
cpt_max = []
for i, values in enumerate([slip_values, trise_values]):
    cpt_max.append(np.percentile(values, 99))
    # 2 sf
    cpt_max[i] = round(cpt_max[i], 1 - int(floor(log10(abs(cpt_max[i])))))
    gmt.makecpt(plot_cpt[i], cpt_out[i], 0, cpt_max[i], cpt_max[i] / 1000, \
            continuous = True)

# overlays
grd_file = '%s/tmp.grd' % (gmt_temp)
for s, seg in enumerate(planes):
    for r, row_data in enumerate([slips[s], trises[s], rakes[s]]):
        # shift to plot origin (bottom left corner)
        if s == 0 and r == 0:
            # reason for x_space is to make sure long titles fit
            x_shift = left_margin + x_space / 2
            y_shift = media[1] - top_margin - max(seg_wid_d) \
                    + (max(seg_wid_d) - seg_wid_d[s])
        elif r != 0:
            x_shift = 0
            y_shift = - row_spacing - max(seg_wid_d)
        else:
            x_shift = seg_len_d[s - 1]
            if not seg_sub[s]:
                x_shift += col_spacing
            y_shift = (nrow - 1) * (row_spacing + max(seg_wid_d)) \
                    - (max(seg_wid_d) - seg_wid_d[s - 1]) \
                    + (max(seg_wid_d) - seg_wid_d[s])

        # only show tick labels on edges
        tick_labels = '%s%s' % ('w' * (s == 0), 's' * (r == nrow - 1))

        # rake angles have static background
        if r == 2:
            fill = '255/255/200'
        else:
            fill = None
        # setup mapping region, - in sizing inverts axis
        p.spacial('X', (0, seg['length'], 0, seg['width']), \
                sizing = '%s/-%s' % (seg_len_d[s], seg_wid_d[s]), \
                x_shift = x_shift, y_shift = y_shift, fill = fill)
        # for labels to be less likely to overlap
        align = 'L'
        if s % 2:
            align = 'R'
        # strike (x) label
        if r == 2:
            p.text(seg['length'] * (s % 2), seg['width'], 'L (km)', \
                    dy = - x_text_gap, align = '%sT' % align, size = base_size)
            p.text(seg['length'] * (s % 2), seg['width'], \
                    'strike %s\260' % (seg['strike']), \
                    dy = - x_text_gap - base_gap, \
                    align = '%sT' % align, size = base_size)
            p.text(seg['length'] * (s % 2), seg['width'], \
                    'dip %s\260' % (seg['dip']), \
                    dy = - x_text_gap - base_gap * 2, \
                    align = '%sT' % align, size = base_size)
        # dip (y) label
        if s == 0:
            p.text(0, max(seg_wid) / 2., 'W (km)', \
                    dx = - y_text_gap, align = 'CB', angle = 90, \
                    size = base_size)
        # min, average, max labels
        mn = round(min(row_data[:, 2]), 1)
        avg = round(np.average(row_data[:, 2]), 1)
        mx = round(max(row_data[:, 2]), 1)
        if r == 2:
            mn, avg, mx = map(int, [mn, avg, mx])
        p.text(seg['length'] * (s % 2), 0, '%s / %s / %s' % (mn, avg, mx), \
                dy = 0.04, align = '%sB' % align, size = annot_size)

        # overlay data
        if r != 2:
            grd_table = '\n'.join(['%f %f %f' % tuple(row) \
                    for row in row_data])
            gmt.table2grd(grd_table, grd_file, file_input = False, \
                    grd_type = 'xyz2grd', dx = srf_dx, dy = srf_dy, geo = False)
            p.overlay(grd_file, cpt_out[r], dx = srf_dx, dy = srf_dy, \
                transparency = 0)
        else:
            # rake angles
            # only show arrows at dx, dy resolution
            dx_rake = args.rake_spacing * srf_dx
            dy_rake = args.rake_spacing * srf_dy
            # first arrow is at midpoint (0.5 dx and dy rake)
            nx = int((seg_len[s] + 0.5 * dx_rake) / dx_rake)
            ny = int((seg_wid[s] + 0.5 * dy_rake) / dy_rake)
            grid = np.mgrid[dx_rake / 2 : dx_rake * nx : dx_rake, \
                    dy_rake / 2 : dy_rake * ny : dy_rake] \
                    .reshape(2, -1, order = 'F').T
            gridpoints = grid.shape[0]
            # rake angle and slip amount
            rk = np.zeros(gridpoints)
            sp = np.zeros(gridpoints)

            # decimated x, y and grid array positons
            ix = (rakes[s][:, 0] / dx_rake).astype('i4')
            iy = (rakes[s][:, 1] / dy_rake).astype('i4')
            ip = ix + iy * nx

            # use average value of all positions in grid square
            if args.rake_average:
                for x in np.nditer(np.arange(gridpoints)):
                    rk[x] = np.average(rakes[s][ip == x, 2])
                    sp[x] = np.average(slips[s][ip == x, 2])
            # use closest position to middle of grid square
            else:
                # distances from midpoint
                mr = ((ix + 0.5) * dx_rake - rakes[s][:, 0]) ** 2 \
                        + ((iy + 0.5) * dy_rake - rakes[s][:, 1]) ** 2
                for x in xrange(gridpoints):
                    try:
                        # position with the smallest distance to midpoint
                        mid = np.where(ip == x)[0][np.argmin(mr[ip == x])]
                    except ValueError:
                        print('DX / DY rake too small.')
                        exit(1)
                    rk[x] = rakes[s][mid, 2]
                    sp[x] = slips[s][mid, 2]

            output = []
            for x in xrange(gridpoints):
                if not np.isnan(rk[x]):
                    output.append('%f %f %f %f\n' % (grid[x][0], grid[x][1], \
                            -rk[x], \
                            args.rake_scale * sp[x] / max(slip_values)))
            # RWG default is +a45+ea
            p.points(''.join(output), is_file = False, shape = 'v', \
                    size = '%sp+a45+eA+g-' % (scale_factor * 6.), \
                    line = 'black', \
                    line_thickness = '%sp' % (scale_factor / 2.))

        # also show tinit on top of slip
        if r == 0:
            grd_table = '\n'.join(['%f %f %f' % tuple(row) \
                    for row in tinits[s]])
            gmt.table2grd(grd_table, grd_file, file_input = False, \
                    grd_type = 'xyz2grd', dx = srf_dx, dy = srf_dy, geo = False)
            p.overlay(grd_file, None, dx = srf_dx, dy = srf_dy, \
                    acontours = acontour, font_size = annot_size, \
                    contour_thickness = scale_factor, contour_colour = 'black')
            # and hypocentre if first segment
            if s == hyp_seg:
                p.points('%s %s' % (hyp_s, hyp_d), is_file = False, \
                        shape = 'a', size = (scale_factor * 0.6), \
                        line = 'red', line_thickness = '%sp' % (scale_factor))

        # no more tick labels than neccessary
        p.ticks(major = major_tick, minor = major_tick / 4, sides = tick_labels)

        # scale
        if s < len(planes) - 1:
            continue
        if r == 2:
            p.text(seg_len[-1], max(seg_wid) / 2, labels[r], \
                    size = label_size, dx = col_spacing, \
                    align = 'CT', angle = 90)
            continue
        # alignment override required due to possibility of different heights
        p.cpt_scale('R', 'T', cpt_out[r], cpt_max[r] / 4., cpt_max[r] / 16., \
                arrow_f = True, dx = col_spacing, pos = 'rel_out', \
                cross_tick = cpt_max[r] / 4., length = max(seg_wid_d), \
                thickness = scale_factor / 5., horiz = False, \
                align = 'LT', label = labels[r])

# main title
if args.title is not None:
    p.text((sum(seg_len) + col_spacing * (ncol - 1) * km_inch) / -2 \
            + seg_len[-1], \
            -(max(seg_wid) * 2 + row_spacing * 2 * km_inch), \
            args.title, font = 'Helvetica-Bold', size = 20, \
            dy = 0.4, align = 'CB')

p.finalise()
p.png(dpi=600, portrait=True, out_dir=args.out_dir)
rmtree(gmt_temp)
