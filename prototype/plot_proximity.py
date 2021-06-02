#!/usr/bin/env python

from argparse import ArgumentParser
import math
import os
import pkg_resources

import numpy as np

from qcore import gmt

corners = pkg_resources.resource_filename("visualization", "data/SimAtlasFaults.csv")
balls = pkg_resources.resource_filename("visualization", "data/gmt.bb")

mom2mag = lambda mom: (2 / 3.0 * math.log(mom) / math.log(10.0)) - 10.7

parser = ArgumentParser()
arg = parser.add_argument
arg("lat", help="Latitude", type=float)
arg("lon", help="Longitude", type=float)
args = parser.parse_args()

p = gmt.GMTPlot("proximity_plot.ps")
# in a future release of GMT, this might be possible
# p.spacial("M" + str(args.lon) + "/" + str(args.lat) + "/", ("-200", "200", "-200", "200+uk"), sizing=8, x_shift=2, y_shift=2)
p.spacial(
    "M",
    (args.lon - 1.3, args.lon + 1.3, args.lat - 1, args.lat + 1),
    sizing=8,
    x_shift=1,
    y_shift=1,
)
p.basemap()

paths = []
with open(corners, "r") as c:
    c.readline()
    for l in c:
        paths.append(
            l.split(",")[9]
            .replace("]|", "\n")
            .replace("|", " ")
            .replace("[[", ">\n")
            .replace("[", "")
            .replace("]", "")
            .replace("\n ", "\n")
        )
paths = "".join(paths)
p.path(
    paths,
    is_file=False,
    close=True,
    colour="black",
    width="1.0p",
    cols="1,0",
    split="-",
)
paths = "\n".join([">\n" + "\n".join(x.split("\n")[1:3]) for x in paths.split(">")])
p.path(paths, is_file=False, colour="black", width="1.5p", cols="1,0")


p.ticks()

# beachballs by magnitude
b5 = []
b56 = []
b6 = []
with open(balls, "r") as b:
    for l in b:
        man, exp = map(float, l.split()[9:11])
        mag = mom2mag(man * 10 ** exp)
        if mag < 5:
            b5.append(l)
        elif mag < 6:
            b56.append(l)
        else:
            b6.append(l)
if len(b5) > 0:
    p.beachballs("\n".join(b5), scale=0.2, colour="blue")
if len(b56) > 0:
    p.beachballs("\n".join(b56), scale=0.2, colour="orange")
if len(b6) > 0:
    p.beachballs("\n".join(b6), scale=0.2, colour="red")

p.points(
    "{} {}\n".format(args.lon, args.lat),
    is_file=False,
    shape="c",
    fill="black",
    size=0.1,
    line="white",
    line_thickness="1p",
)
p.text(
    args.lon, args.lat, "site", dy=-0.12, align="CT", size="14p", box_fill="white@40"
)
p.dist_scale("R", "B", "25", pos="rel", dx=0.5, dy=0.5)

p.finalise()
p.png(clip=True)
