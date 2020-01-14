#!/usr/bin/env python

from argparse import ArgumentParser

from qcore import geo
from qcore import gmt

GRID = "/nesi/nobackup/nesi00213/seistech/sites/18p6/non_uniform_whole_nz_with_real_stations-hh400_v18p6_land.ll"

parser = ArgumentParser()
parser.add_argument("lon", type=float)
parser.add_argument("lat", type=float)
parser.add_argument("closest_lon", type=float)
parser.add_argument("closest_lat", type=float)
parser.add_argument("--distance", type=float, default=8)
args = parser.parse_args()


max_lat = geo.ll_shift(args.lat, args.lon, args.distance, 0)[0]
min_lon = geo.ll_shift(args.lat, args.lon, args.distance, -90)[1]

region = (min_lon, args.lon + (args.lon - min_lon), args.lat - (max_lat - args.lat), max_lat)

p = gmt.GMTPlot("snapped_station.ps")
p.spacial("M", region, sizing=9, x_shift=1, y_shift=2)
gmt.makecpt("rainbow", "vs30.cpt", 100, 800, continuing=True)
p.overlay("nz_vs30_nz-specific-v19p1_100m.grd", cpt="vs30.cpt")
p.points(GRID, shape='x', size=0.4, line_thickness="2p", line="black")

p.points("{} {}\n".format(args.lon, args.lat), is_file=False, shape="c", fill="black", size=0.1, line="white", line_thickness="1p")
p.text(args.lon, args.lat, "site", dy=-0.12, align="CT", size="14p", box_fill="white@40")

p.ticks(major="0.05d", minor="0.01d")
p.cpt_scale("M", "B", "vs30.cpt", pos="rel_out", dy=0.6, label="Vs30 (m/s)", major=100, minor=10)
p.finalise()
p.png(background="white")
