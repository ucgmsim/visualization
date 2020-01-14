#!/usr/bin/env python

from argparse import ArgumentParser

from qcore import geo
from qcore import gmt

GRID = "/nesi/nobackup/nesi00213/seistech/sites/18p6/non_uniform_whole_nz_with_real_stations-hh400_v18p6_land.ll"

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

p = gmt.GMTPlot("snapped_station.ps")
p.spacial("M", region, sizing=9, x_shift=1, y_shift=2)
gmt.makecpt("rainbow", "vs30.cpt", 100, 800, continuing=True)
p.overlay("nz_vs30_nz-specific-v19p1_100m.grd", cpt="vs30.cpt")
p.points(GRID, shape="x", size=0.4, line_thickness="2p", line="black")

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
    size=0.4,
    line_thickness="2p",
    line="white",
)
p.text(
    args.lon, args.lat, "site", dy=-0.12, align="CT", size="14p", box_fill="white@40"
)
p.text(
    min_lon,
    min_lat,
    "Site Vs30: {}".format(args.site_vs30),
    size="20p",
    align="LB",
    dx=0.2,
    dy=0.2,
    box_fill="white@40",
)
p.text(
    max_lon,
    min_lat,
    "Closest Site Vs30: {}".format(args.closest_vs30),
    size="20p",
    align="RB",
    dx=-0.2,
    dy=0.2,
    box_fill="white@40",
)

p.ticks(major="0.05d", minor="0.01d")
p.cpt_scale(
    "R",
    "M",
    "vs30.cpt",
    pos="rel_out",
    dx=0.2,
    label="Vs30 (m/s)",
    major=100,
    minor=10,
    horiz=False,
)
p.finalise()
p.png(background="white")
