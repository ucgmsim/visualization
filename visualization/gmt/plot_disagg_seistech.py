#!/usr/bin/env python2
"""
Plots deagg data.

Requires:
numpy
gmt from qcore
"""

from argparse import ArgumentParser
from io import BytesIO
import json
import math
import os
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from qcore import gmt

X_LEN = 4.5
Y_LEN = 4.0
Z_LEN = 2.5
ROT = 30
TILT = 60
LEGEND_SPACE = 0.7
EPSILON_LEGEND_EXPAND = 1.25
EPSILON_COLOURS = [
    "215/38/3",
    "252/94/62",
    "252/180/158",
    "254/220/210",
    "217/217/255",
    "151/151/255",
    "0/0/255",
    "0/0/170",
]
EPSILON_LABELS = [
    "@~e@~<-2",
    "-2<@~e@~<-1",
    "-1<@~e@~<-0.5",
    "-0.5<@~e@~<0",
    "0<@~e@~<0.5",
    "0.5<@~e@~<1",
    "1<@~e@~<2",
    "2<@~e@~",
]
TYPE_LEGEND_EXPAND = 0.35
TYPE_COLOURS = ["blue", "red", "green"]
TYPE_LABELS = ["A", "B", "DS"]

###
### LOAD DATA
###
parser = ArgumentParser()
parser.add_argument("disagg_json", help="disagg file to plot")
parser.add_argument("--out-name", help="basename excluding extention", default="disagg")
parser.add_argument("--out-dir", help="directory to store output", default=".")
parser.add_argument("--dpi", help="dpi of raster output", type=int, default=300)
parser.add_argument("-z", help='"epsilon" or "type"', default="epsilon")
args = parser.parse_args()
assert os.path.exists(args.disagg_json)
if not os.path.exists(args.out_dir):
    try:
        os.makedirs(args.out_dir)
    except OSError:
        if not os.path.isdir(args.out_dir):
            raise
with open(args.disagg_json, "rb") as j:
    jdisagg = json.loads(j.read())

# modifications based on plot type selection
if args.z == "type":
    colours = TYPE_COLOURS
    labels = TYPE_LABELS
    legend_expand = TYPE_LEGEND_EXPAND
else:
    colours = EPSILON_COLOURS
    labels = EPSILON_LABELS
    legend_expand = EPSILON_LEGEND_EXPAND

###
### PROCESS DATA
###
# x axis
x_max = max(jdisagg["rrup_edges"])
if x_max < 115:
    x_inc = 10
elif x_max < 225:
    x_inc = 20
elif x_max < 335:
    x_inc = 30
elif x_max < 445:
    x_inc = 40
else:
    x_inc = 50
dx = x_inc / 2.0
x_max = math.ceil(x_max / float(dx)) * dx

# y axis
y_min = jdisagg["mag_edges"][0]
y_max = jdisagg["mag_edges"][-1]
if y_max - y_min < 5:
    y_inc = 0.5
else:
    y_inc = 1.0
dy = y_inc / 2.0

# bins to put data in
# TODO: set bottom limit on x and y (not e)
bins_x = np.array(jdisagg["rrup_edges"][1:])
bins_y = np.array(jdisagg["mag_edges"][1:])
bins_e = np.array([-2, -1, -0.5, 0, 0.5, 1, 2, np.inf])
# XXX: missing data
bins_e = np.array([-2, -1, 0, 1, 2, np.inf])

# build gmt input lines from block data
gmt_in = BytesIO()
if args.z == "type":
    blocks_flt = np.array(jdisagg["flt_bin_contr"])
    blocks_ds = np.array(jdisagg["ds_bin_contr"])
    # sum to 100
    factor = 100 / (np.sum(blocks_flt) + np.sum(blocks_ds))
    blocks_flt *= factor
    blocks_ds *= factor
    for y in range(len(bins_y)):
        for x in range(len(bins_x)):
            if blocks_flt[y, x] > 0:
                base = blocks_flt[y, x]
                gmt_in.write("%s %s %s %s %s\n" % (bins_x[x], bins_y[y], base, 0, 0))
            else:
                base = 0
            if blocks_ds[y, x] > 0:
                gmt_in.write(
                    "%s %s %s %s %s\n"
                    % (bins_x[x], bins_y[y], base + blocks_ds[y, x], 2, base)
                )
    # z axis depends on max contribution tower
    z_inc = int(math.ceil(np.max(np.add.reduce(blocks_flt + blocks_ds, axis=1)) / 5.0))
    z_max = z_inc * 5
    del blocks_flt, blocks_ds
else:
    blocks = np.array(jdisagg["eps_bin_contr"])
    # sum to 100
    blocks *= 100 / np.sum(blocks)
    for z in range(len(bins_e)):
        for y in range(len(bins_y)):
            for x in range(len(bins_x)):
                if blocks[z, y, x] > 0:
                    base = sum(blocks[:z, y, x])
                    gmt_in.write(
                        "%s %s %s %s %s\n"
                        % (bins_x[x], bins_y[y], base + blocks[z, y, x], z, base)
                    )
    # z axis depends on max contribution tower
    z_inc = int(math.ceil(np.max(np.add.reduce(blocks, axis=2)) / 5.0))
    z_max = z_inc * 5
    del blocks

###
### PLOT AXES
###
wd = mkdtemp()
p = gmt.GMTPlot("%s.ps" % os.path.join(wd, args.out_name))
os.remove(os.path.join(wd, "gmt.conf"))
# setup axes
p.spacial(
    "X",
    (0, x_max, y_min, y_max, 0, z_max),
    sizing="%si/%si" % (X_LEN, Y_LEN),
    z="Z%si" % (Z_LEN),
    p="%s/%s" % (180 - ROT, 90 - TILT),
    x_shift="5",
    y_shift=5,
)
p.ticks_multi(
    [
        "xa%s+lRupture Distance (km)" % (x_inc),
        "ya%s+lMagnitude" % (y_inc),
        "za%sg%s+l%%Contribution" % (z_inc, z_inc),
        "wESnZ",
    ]
)
# GMT will not plot gridlines without box, manually add gridlines
gridlines = []
for z in xrange(z_inc, z_max + z_inc, z_inc):
    gridlines.append(
        "0 %s %s\n0 %s %s\n%s %s %s" % (y_min, z, y_max, z, x_max, y_max, z)
    )
gridlines.append("0 %s 0\n0 %s %s" % (y_max, y_max, z_max))
gridlines.append("%s %s 0\n%s %s %s" % (x_max, y_max, x_max, y_max, z_max))
p.path("\n>\n".join(gridlines), is_file=False, width="0.5p", z=True)

###
### PLOT CONTENTS
###
cpt = os.path.join(wd, "z.cpt")
gmt.makecpt(",".join(colours), cpt, 0, len(colours), inc=1, wd=wd)
p.points(
    gmt_in.getvalue(),
    is_file=False,
    z=True,
    line="black",
    shape="o",
    size="%si/%sib"
    % (float(X_LEN) / len(bins_x) - 0.05, float(Y_LEN) / len(bins_x) - 0.05),
    line_thickness="0.5p",
    cpt=cpt,
)

###
### PLOT LEGEND
###
# x y diffs from start to end, alternatively run multiple GMT commands with -X
angle = math.radians(ROT)
map_width = math.cos(angle) * X_LEN + math.sin(angle) * Y_LEN
x_end = (
    (X_LEN + math.cos(angle) * math.sin(angle) * (Y_LEN - math.tan(angle) * X_LEN))
    / X_LEN
    * x_max
    * legend_expand
)
y_end = math.tan(angle) * x_end / x_max * X_LEN * (y_max - y_min) / Y_LEN
# x y diffs at start, alternatively set -D(dz)
x_shift = map_width * (legend_expand - 1) * -0.5
y_shift = (LEGEND_SPACE) / math.cos(math.radians(TILT)) + X_LEN * math.sin(angle)
x0 = (y_shift * math.sin(angle) + x_shift * math.cos(angle)) * (x_max / X_LEN)
y0 = y_min + (-y_shift * math.cos(angle) + x_shift * math.sin(angle)) * (
    (y_max - y_min) / Y_LEN
)
# legend definitions
legend_boxes = []
legend_labels = []
for i, x in enumerate(np.arange(0, 1.01, 1.0 / (len(colours) - 1.0))):
    legend_boxes.append(
        "%s %s %s %s" % (x0 + x * x_end, y0 + x * y_end, z_inc / 2.0, i)
    )
    legend_labels.append("%s 0 %s" % (x, labels[i]))
# cubes and labels of legend
p.points(
    "\n".join(legend_boxes),
    is_file=False,
    z=True,
    line="black",
    shape="o",
    size="%si/%sib0" % (Z_LEN / 10.0, Z_LEN / 10.0),
    line_thickness="0.5p",
    cpt=cpt,
    clip=False,
)
p.spacial(
    "X",
    (0, 1, 0, 1),
    sizing="%si/1i" % (map_width * legend_expand),
    x_shift="%si" % (x_shift),
    y_shift="-%si" % (LEGEND_SPACE + 0.2),
)
p.text_multi("\n".join(legend_labels), is_file=False, justify="CT")

###
### SAVE
###
p.finalise()
p.png(
    portrait=True,
    background="white",
    dpi=args.dpi,
    out_dir=args.out_dir,
    margin=[0.618, 1],
)
rmtree(wd)
