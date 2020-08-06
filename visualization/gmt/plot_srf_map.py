#!/usr/bin/env python
"""
"""

from argparse import ArgumentParser
from math import floor, log10
import os
from shutil import rmtree
from subprocess import call
import sys
from tempfile import mkdtemp

import numpy as np

import qcore.geo as geo
import qcore.gmt as gmt
import qcore.srf as srf


def get_args():
    parser = ArgumentParser()
    arg = parser.add_argument

    arg("srf_file", help="srf file to plot")
    arg("--dpi", help="render dpi", type=int, default=300)
    arg("--active-faults", help="show active faults", action="store_true")
    arg("--cpt", help="CPT for SRF slip", default=gmt.CPTS["slip"])

    args = parser.parse_args()
    args.srf_file = os.path.abspath(args.srf_file)
    if not os.path.exists(args.srf_file):
        sys.exit("SRF file not found.")

    return args


faults = "/nesi/project/nesi00213/PlottingData/Paths/faults/FAULTS_20161219.ll"

args = get_args()
# output directory for srf resources
gmt_tmp = os.path.abspath(mkdtemp())

# whether we are plotting a finite fault or point source
finite_fault = srf.is_ff(args.srf_file)
if finite_fault:
    dx, dy = srf.srf_dxy(args.srf_file)
    text_dx = "%s km" % (dx)
    text_dy = "%s km" % (dy)
    # plot at greater resolution to increase smoothness
    # also considering rotation and roughness, grid will not be exactly matching
    plot_dx = "%sk" % (dx * 0.3)
    plot_dy = "%sk" % (dy * 0.3)
    # output for plane data
    os.makedirs(os.path.join(gmt_tmp, "PLANES"))
else:
    text_dx = "N/A"
    text_dy = "N/A"

###
### OUTPUT 1: binary file for GMT grid plotting
###
if finite_fault:
    # get all corners
    bounds = srf.get_bounds(args.srf_file)
    # get all tinit values, set a sane countour interval
    # contour interval should probably also depend on area
    tinit = srf.srf2llv_py(args.srf_file, value="tinit")
    tinit_max = max([np.max(tinit[p][:, 2]) for p in range(len(bounds))])
    contour_int = 2
    if tinit_max < 10:
        contour_int = 1
    if tinit_max < 2:
        contour_int = 0.3
    # gridding is computationally expensive
    # each segment should include minimal region
    seg_regions = []
    # for percentile to automatically calculate colour palette range
    values = np.array([], dtype=np.float32)
    # find total extremes
    np_bounds = np.array(bounds)
    x_min, y_min = np.min(np.min(np_bounds, axis=0), axis=0)
    x_max, y_max = np.max(np.max(np_bounds, axis=0), axis=0)
    plot_region = (x_min - 0.1, x_max + 0.1, y_min - 0.1, y_max + 0.1)
    # read all max slip values (all at once is much faster)
    seg_llslips = srf.srf2llv_py(args.srf_file, value="slip")
    seg_lldepths = srf.srf2llv_py(args.srf_file, value="depth")
    depth_max = 0.0
    for seg in range(len(bounds)):
        # create binary llv file for GMT overlay
        seg_llslips[seg].astype(np.float32).tofile(
            "%s/PLANES/slip_map_%d.bin" % (gmt_tmp, seg)
        )
        seg_lldepths[seg].astype(np.float32).tofile(
            "%s/PLANES/depth_map_%d.bin" % (gmt_tmp, seg)
        )
        values = np.append(values, seg_llslips[seg][:, -1])
        depth_max = max(depth_max, np.max(seg_lldepths[seg][:, -1]))
        # also store tinit values retrieved previously
        tinit[seg].astype(np.float32).tofile(
            "%s/PLANES/tinit_map_%d.bin" % (gmt_tmp, seg)
        )
        # create a mask path for GMT overlay
        geo.path_from_corners(
            corners=bounds[seg],
            min_edge_points=100,
            output="%s/PLANES/plane_%d.bounds" % (gmt_tmp, seg),
        )
        # create mask from path
        x_min, y_min = np.min(np_bounds[seg], axis=0)
        x_max, y_max = np.max(np_bounds[seg], axis=0)
        seg_regions.append((x_min, x_max, y_min, y_max))
        gmt.grd_mask(
            "%s/PLANES/plane_%d.bounds" % (gmt_tmp, seg),
            "%s/PLANES/plane_%d.mask" % (gmt_tmp, seg),
            dx=plot_dx,
            dy=plot_dy,
            region=seg_regions[seg],
        )
    percentile = np.percentile(values, 95)
    maximum = np.max(values)
    average = np.average(values)
    subfaults = len(values)
    # round percentile significant digits for colour pallete
    if percentile < 1000:
        # 1 sf
        cpt_max = round(percentile, -int(floor(log10(abs(percentile)))))
    else:
        # 2 sf
        cpt_max = round(percentile, 1 - int(floor(log10(abs(percentile)))))
else:
    bounds = []


###
### OUTPUT 2: corners file for fault plane and hypocentre plot
###
if not finite_fault:
    hypocentre = srf.get_hypo(args.srf_file, depth=True)
    plot_region = (
        hypocentre[0] - 0.2,
        hypocentre[0] + 0.2,
        hypocentre[1] - 0.1,
        hypocentre[1] + 0.1,
    )
    subfaults = 1
    maximum = srf.srf2llv_py(args.srf_file, value="slip")[0][0][-1]
    percentile = maximum
    average = maximum
    # arbitrary, only changes size of beachball which is relative anyway
    mw = 8
    strike, dip, rake = srf.ps_params(args.srf_file)
# for plotting region on NZ-wide map
plot_bounds = "%f %f\n%f %f\n%f %f\n%f %f\n" % (
    plot_region[0],
    plot_region[2],
    plot_region[1],
    plot_region[2],
    plot_region[1],
    plot_region[3],
    plot_region[0],
    plot_region[3],
)


###
### OUTPUT 3: GMT MAP
###
perimeters, top_edges = srf.get_perimeter(args.srf_file)
nz_region = gmt.nz_region
if finite_fault:
    gmt.makecpt(args.cpt, "%s/slip.cpt" % (gmt_tmp), 0, cpt_max, 1)
    gmt.makecpt("gray", "%s/depth.cpt" % (gmt_tmp), 0, depth_max, 0.1, invert=True)
gmt.gmt_defaults(wd=gmt_tmp)
# gap on left of maps
gap = 1
# width of NZ map, if changed, other things also need updating
# including tick font size and or tick increment for that map
full_width = 4

### PART A: zoomed in map
p = gmt.GMTPlot(
    "%s/%s_map.ps" % (gmt_tmp, os.path.splitext(os.path.basename(args.srf_file))[0])
)
# this is how high the NZ map will end up being
full_height = gmt.mapproject(
    nz_region[0],
    nz_region[3],
    region=nz_region,
    projection="M%s" % (full_width),
    wd=gmt_tmp,
)[1]
# match height of zoomed in map with full size map
zoom_width, zoom_height = gmt.map_width("M", full_height, plot_region, wd=gmt_tmp)
p.spacial("M", plot_region, sizing=zoom_width, x_shift=gap, y_shift=2.5)
p.basemap(topo=os.path.join(gmt.GMT_DATA, "Topo/srtm_NZ_1s.grd"), land="lightgray", topo_cpt="grey1")
if args.active_faults:
    p.path(faults, is_file=True, close=False, width="0.4p", colour="red")
for seg in range(len(bounds)):
    gmt_outline = "\n".join(" ".join(list(map(str, x))) for x in perimeters[seg])
    gmt_top_edge = "\n".join(" ".join(list(map(str, x))) for x in top_edges[seg])
    p.clip(path=gmt_outline)
    gmt.table2grd(
        "%s/PLANES/depth_map_%d.bin" % (gmt_tmp, seg),
        "%s/PLANES/depth_map_%d.grd" % (gmt_tmp, seg),
        region=seg_regions[seg],
        dx=plot_dx,
        wd=gmt_tmp,
        climit=2,
    )
    gmt.table2grd(
        "%s/PLANES/slip_map_%d.bin" % (gmt_tmp, seg),
        "%s/PLANES/slip_map_%d.grd" % (gmt_tmp, seg),
        region=seg_regions[seg],
        dx=plot_dx,
        wd=gmt_tmp,
        climit=2,
    )
    p.overlay(
        "%s/PLANES/depth_map_%d.bin" % (gmt_tmp, seg),
        "%s/depth.cpt" % (gmt_tmp),
        dx=plot_dx,
        dy=plot_dy,
        climit=0.1,
        land_crop=False,
        custom_region=seg_regions[seg],
        transparency=0,
    )
    # TODO: fix working directory
    call([
        "gmt",
        "grdgradient",
        "%s/PLANES/depth_map_%d.grd" % (gmt_tmp, seg),
        "-G%s/PLANES/illu_map_%d.grd" % (gmt_tmp, seg),
        "-Ne.5",
        "-A100"])
    p.topo(
        "%s/PLANES/slip_map_%d.grd" % (gmt_tmp, seg),
        topo_file_illu="%s/PLANES/illu_map_%d.grd" % (gmt_tmp, seg),
        cpt="%s/slip.cpt" % (gmt_tmp),
        transparency=30,
    )
    p.overlay(
        "%s/PLANES/tinit_map_%d.bin" % (gmt_tmp, seg),
        None,
        dx=plot_dx,
        dy=plot_dy,
        climit=2,
        land_crop=False,
        custom_region=seg_regions[seg],
        transparency=30,
        contours=contour_int,
    )
    p.clip()
    p.clip(path=gmt_outline, invert=True)
    p.path(gmt_outline, is_file=False, colour="black", split="-", width="2p")
    p.path(gmt_top_edge, is_file=False, colour="black", width="4p")
    p.clip()
if finite_fault:
    hypocentre = srf.get_hypo(args.srf_file, depth=False)
    p.points("{} {}".format(*hypocentre), is_file=False, shape="a", size="0.3i", line="red", line_thickness="1p")
else:
    p.beachballs(
        "%s %s %s %s %s %s %s %s %s\n"
        % (
            hypocentre[0],
            hypocentre[1],
            hypocentre[2],
            strike,
            dip,
            rake,
            mw,
            hypocentre[0],
            hypocentre[1],
        ),
        is_file=False,
        fmt="a",
        scale=0.4,
    )

p.sites(list(gmt.sites.keys()))
major_tick, minor_tick = gmt.auto_tick(plot_region[0], plot_region[1], zoom_width)
major_tick = max(0.1, major_tick)
p.ticks(major="%sd" % (major_tick), minor="%sd" % (minor_tick), sides="ws")

### PART B: NZ map
# draw NZ wide map to show rough location in country
p.spacial("M", nz_region, sizing=full_width, x_shift=zoom_width + gap)
# height of NZ map
full_height = gmt.mapproject(nz_region[0], nz_region[3], wd=gmt_tmp)[1]
p.basemap(land="lightgray", topo=gmt.TOPO_LOW, topo_cpt="grey1", road=None)
if args.active_faults:
    p.path(faults, is_file=True, close=False, width="0.1p", colour="red")
p.path(plot_bounds, is_file=False, close=True, colour="black")
# get displacement of box to draw zoom lines later
window_bottom = gmt.mapproject(plot_region[1], plot_region[2], wd=gmt_tmp)
window_top = gmt.mapproject(plot_region[1], plot_region[3], wd=gmt_tmp)
if finite_fault:
    for seg in range(len(bounds)):
        p.path("\n".join(" ".join(list(map(str, x))) for x in perimeters[seg]), is_file=False)
else:
    p.beachballs(
        "%s %s %s %s %s %s %s %s %s\n"
        % (
            hypocentre[0],
            hypocentre[1],
            hypocentre[2],
            strike,
            dip,
            rake,
            mw,
            hypocentre[0],
            hypocentre[1],
        ),
        is_file=False,
        fmt="a",
        scale=0.05,
    )
p.ticks(major="2d", minor="30m", sides="ws")

### PART C: zoom lines
# draw zoom lines that extend from view box to original plot
p.spacial(
    "X",
    (0, window_bottom[0] + gap, 0, max(zoom_height, full_height)),
    x_shift=-gap,
    sizing="%s/%s" % (window_top[0] + gap, max(zoom_height, full_height)),
)
p.path(
    "%f %f\n%f %f\n" % (0, 0, window_bottom[0] + gap, window_bottom[1]),
    width="0.6p",
    is_file=False,
    split="-",
    straight=True,
    colour="black",
)
p.path(
    "%f %f\n%f %f\n" % (0, zoom_height, window_top[0] + gap, window_top[1]),
    width="0.6p",
    is_file=False,
    split="-",
    straight=True,
    colour="black",
)

### PART D: surrounding info
# add text and colour palette
# position to enclose both plots
total_width = zoom_width + gap + full_width
total_height = max(zoom_height, full_height)
p.spacial(
    "X",
    (0, total_width, 0, total_height + 2),
    sizing="%s/%s" % (total_width, total_height + 2),
    x_shift=-zoom_width,
)
# SRF filename
p.text(
    total_width / 2.0,
    total_height,
    os.path.basename(args.srf_file),
    align="CB",
    size="20p",
    dy=0.8,
)
# max slip
p.text(zoom_width / 2.0, total_height, "Maximum slip: ", align="RB", size="14p", dy=0.5)
p.text(
    zoom_width / 2.0 + 0.1,
    total_height,
    "%.1f cm" % (maximum),
    align="LB",
    size="14p",
    dy=0.5,
)
# 95th percentile
p.text(
    zoom_width / 2.0, total_height, "95th percentile: ", align="RB", size="14p", dy=0.3
)
p.text(
    zoom_width / 2.0 + 0.1,
    total_height,
    "%.1f cm" % (percentile),
    align="LB",
    size="14p",
    dy=0.3,
)
# average slip
p.text(zoom_width / 2.0, total_height, "Average slip: ", align="RB", size="14p", dy=0.1)
p.text(
    zoom_width / 2.0 + 0.1,
    total_height,
    "%.1f cm" % (average),
    align="LB",
    size="14p",
    dy=0.1,
)
# planes
p.text(total_width - 4 / 2.0, total_height, "Planes: ", align="RB", size="14p", dy=0.5)
p.text(
    total_width - 4 / 2.0 + 0.1,
    total_height,
    len(bounds),
    align="LB",
    size="14p",
    dy=0.5,
)
# dx and dy
p.text(total_width - 4 / 2.0, total_height, "dX, dY: ", align="RB", size="14p", dy=0.3)
p.text(
    total_width - 4 / 2.0 + 0.1,
    total_height,
    "%s, %s" % (text_dx, text_dy),
    align="LB",
    size="14p",
    dy=0.3,
)
# subfaults
p.text(
    total_width - 4 / 2.0, total_height, "Subfaults: ", align="RB", size="14p", dy=0.1
)
p.text(
    total_width - 4 / 2.0 + 0.1, total_height, subfaults, align="LB", size="14p", dy=0.1
)
if finite_fault:
    # scale
    p.cpt_scale(
        zoom_width / 2.0,
        -0.5,
        "%s/slip.cpt" % (gmt_tmp),
        cpt_max / 4.0,
        cpt_max / 8.0,
        label="Slip (cm)",
        length=zoom_width,
    )

p.finalise()
p.png(dpi=args.dpi * 4, downscale=4, background="white", out_dir=".")
rmtree(gmt_tmp)
