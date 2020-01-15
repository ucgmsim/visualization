#!/usr/bin/env python

from argparse import ArgumentParser
import os
from shutil import copyfile, rmtree
from tempfile import mkdtemp

from qcore import geo
from qcore import gmt

GRID = "/nesi/nobackup/nesi00213/seistech/sites/18p6/non_uniform_whole_nz_with_real_stations-hh400_v18p6_land.ll"
VS30 = "/nesi/nobackup/nesi00213/seistech/vs30/19p1/nz_vs30_nz-specific-v19p1_100m.grd"

parser = ArgumentParser()
parser.add_argument("closest_lon", type=float)
parser.add_argument("closest_lat", type=float)
parser.add_argument("closest_vs30", type=float)
parser.add_argument("lon", type=float)
parser.add_argument("lat", type=float)
parser.add_argument("--site-vs30", type=float)
parser.add_argument("--distance", type=float, default=8)
args = parser.parse_args()


max_lat = geo.ll_shift(args.lat, args.lon, args.distance, 0)[0]
min_lon = geo.ll_shift(args.lat, args.lon, args.distance, -90)[1]
min_lat = args.lat - (max_lat - args.lat)
max_lon = args.lon + (args.lon - min_lon)
region = (min_lon, max_lon, min_lat, max_lat)

# automatic label positioning, doesn't work over geographic quadrants
if abs(args.lat - args.closest_lat) > abs(args.lon - args.closest_lon):
    # labels above/below
    dx = 0
    if args.lat > args.closest_lat:
        # site label above, closest site label below
        site_align = "CB"
        closest_align = "CT"
        dy = 0.12
    else:
        # opposite
        site_align = "CT"
        closest_align = "CB"
        dy = -0.12
else:
    # labels to the side
    dy = 0
    if args.lon > args.closest_lon:
        # site label to right, closest site label to left
        site_align = "LM"
        closest_align = "RM"
        dx = -0.12
    else:
        # opposite
        site_align = "RM"
        closest_align = "LM"
        dx = 0.12

wd = mkdtemp()
img = os.path.join(wd, "snapped_station")
cpt = os.path.join(wd, "vs30.cpt")
p = gmt.GMTPlot(img + ".ps")
p.spacial("M", region, sizing=9, x_shift=1, y_shift=2)
gmt.makecpt("rainbow", cpt, 100, 800, continuing=True)
p.overlay(VS30, cpt=cpt)
p.points(GRID, shape="s", size=0.2, line_thickness="2p", line="black")

p.points(
    "{} {}\n".format(args.lon, args.lat),
    is_file=False,
    shape="c",
    fill="black",
    size=0.1,
    line="white",
    line_thickness="1p",
)
p.points(
    "{} {}\n".format(args.closest_lon, args.closest_lat),
    is_file=False,
    shape="c",
    size=0.2,
    line_thickness="2p",
    line="white",
)
p.text(
    args.lon,
    args.lat,
    "site",
    dx=-dx,
    dy=dy,
    align=site_align,
    size="14p",
    box_fill="white@40",
)
p.text(
    args.closest_lon,
    args.closest_lat,
    "closest site",
    dx=dx * 1.5,
    dy=-dy * 1.5,
    align=closest_align,
    size="14p",
    box_fill="white@40",
)
p.text(
    min_lon,
    min_lat,
    "Site Vs30: {} {}".format(args.site_vs30, "m/s" * (args.site_vs30 is not None)),
    size="20p",
    align="LB",
    dx=0.2,
    dy=0.8,
    box_fill="white@40",
)
p.text(
    min_lon,
    min_lat,
    "Closest Site Vs30: {} m/s".format(args.closest_vs30),
    size="20p",
    align="LB",
    dx=0.2,
    dy=0.5,
    box_fill="white@40",
)
p.text(
    min_lon,
    min_lat,
    "Distance: {:.2f} km".format(
        geo.ll_dist(args.closest_lon, args.closest_lat, args.lon, args.lat)
    ),
    size="20p",
    align="LB",
    dx=0.2,
    dy=0.2,
    box_fill="white@40",
)

p.ticks(major="0.05d", minor="0.01d")
p.cpt_scale(
    "R",
    "M",
    cpt,
    pos="rel_out",
    dx=0.2,
    label="Vs30 (m/s)",
    major=100,
    minor=10,
    horiz=False,
)
p.finalise()
p.png(background="white")

# allow setting output location?
copyfile(img + ".png", "./snapped_station.png")
rmtree(wd)
