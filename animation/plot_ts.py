#!/usr/bin/env python
"""
TODO:
- allow plotting single component, cpt_min would have to be calculated
- optimise land crop: download lower res, make template with it (png still slower)
"""

from argparse import ArgumentParser
from multiprocessing import Pool
import os
from shutil import rmtree
from tempfile import mkdtemp
from time import time

import numpy as np

import qcore.gmt as gmt
from qcore.xyts import XYTSFile

# size of plotting area
PAGE_WIDTH = 16
PAGE_HEIGHT = 9
# space around map for titles, tick labels and scales etc
MARGIN_TOP = 1.0
MARGIN_BOTTOM = 0.4
MARGIN_LEFT = 1.0
MARGIN_RIGHT = 1.7

parser = ArgumentParser()
arg = parser.add_argument
arg("xyts", help="path to xyts.e3d file")
arg("--xyts2", help="path to second xyts.e3d file to plot (xyts2 - xyts)")
arg("--output", help="path to save animation (no extention)")
arg("--cpt", help="xyts overlay cpt (default hot or polar for double xyts plot)")
arg("-r", "--dpi", help="dpi 80: 720p, [120]: 1080p, 240: 4k", type=int, default=120)
arg("--title", help="main title", default="Automatically Generated Event")
arg("--subtitle1", help="top subtitle", default="Automatically generated source model")
arg("--subtitle2", help="bottom subtitle", default="NZVM v?.?? h=<HH>km")
arg("--legend", help="colour scale legend text")
arg("-n", "--nproc", help="number of processes to use", type=int, default=1)
arg("--borders", help="opaque map margins", action="store_true")
arg("--srf", help="path to srf file")
arg("--stations", help="path to station file")
arg("--seis", help="path to seismogram overlay file")
arg("--land-crop", help="crop to land (slow)", action="store_true")
arg(
    "--scale",
    help="speed of animation. 1.0 is realtime, 2.0 is double time",
    type=float,
    default=2.0,
)
args = parser.parse_args()
assert os.path.isfile(args.xyts)
if args.xyts2 is None:
    if args.cpt is None:
        args.cpt = "hot"
    if args.legend is None:
        args.legend = "ground motion [cm/s]"
else:
    assert os.path.isfile(args.xyts2)
    if args.cpt is None:
        args.cpt = "polar"
    if args.legend is None:
        args.legend = "sim2 - sim1 ground motion [cm/s]"
assert args.nproc > 0
if args.nproc == 1:
    print("warning: only using 1 process, use more by setting nproc parameter")
if args.srf is not None:
    assert os.path.isfile(args.srf)
if args.stations is not None:
    assert os.path.isfile(args.stations)
if args.seis is not None:
    assert os.path.isfile(args.seis)
if args.output is None:
    args.output = os.path.splitext(os.path.basename(args.xyts))[0]

# prepare temp locations
gmt_temp = mkdtemp()
png_dir = os.path.join(gmt_temp, "TS_PNG")
os.makedirs(png_dir)
cpt_file = "%s/motion.cpt" % (gmt_temp)
# load xyts
xyts = XYTSFile(args.xyts)
if args.xyts2 is not None:
    xyts2 = XYTSFile(args.xyts2)
    attrs = [
        "x0",
        "y0",
        "z0",
        "t0",
        "nx",
        "ny",
        "nz",
        "nt",
        "dx",
        "dy",
        "hh",
        "mrot",
        "mlat",
        "mlon",
        "dt",
    ]
    for a in attrs:
        try:
            assert getattr(xyts, a) == getattr(xyts2, a)
        except AssertionError:
            print(
                f"Values for {a} don't match: {getattr(xyts, a)}, {getattr(xyts2, a)}"
            )
            raise
# xyts derivatives
pgv_file = "%s/PGV.bin" % (gmt_temp)
xyts.pgv(pgvout=pgv_file)
cpt_min = 0
cpt_inc, cpt_max = gmt.xyv_cpt_range(pgv_file)[1:3]
if args.xyts2 is not None:
    cpt_max /= 2
    cpt_min = -cpt_max
    cpt_inc /= 2
    lowcut = cpt_max * 0.02
    highcut = cpt_max * -0.02
else:
    lowcut = cpt_max * 0.02
    highcut = None
convergence_limit = cpt_inc * 0.2
corners, cnr_str = xyts.corners(gmt_format=True)
region = xyts.region()
region_code = gmt.get_region(region[0], region[2])
ll_region = xyts.region(corners=corners)
grd_dxy = "%sk" % (xyts.dx / 2.0)
# determine map sizing
map_width = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
map_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
# extend region to fill view window
map_width, map_height, ll_region = gmt.fill_space(
    map_width, map_height, ll_region, proj="M", dpi=args.dpi, wd=gmt_temp
)
# extend map to cover margins
if not args.borders:
    map_width_a, map_height_a, borderless_region = gmt.fill_margins(
        ll_region,
        map_width,
        args.dpi,
        left=MARGIN_LEFT,
        right=MARGIN_RIGHT,
        top=MARGIN_TOP,
        bottom=MARGIN_BOTTOM,
    )
# colour scale
gmt.makecpt(
    args.cpt,
    cpt_file,
    cpt_min,
    cpt_max,
    inc=cpt_inc,
    invert=args.xyts2 is None,
    bg=None,
    fg=None if args.xyts2 is None else "darkred",
)


def bottom_template():
    t0 = time()
    bwd = os.path.join(gmt_temp, "bottom")
    os.makedirs(bwd)
    b = gmt.GMTPlot("%s/bottom.ps" % (bwd))
    gmt.gmt_defaults(wd=bwd, ps_media="Custom_%six%si" % (PAGE_WIDTH, PAGE_HEIGHT))
    if args.borders:
        b.background(PAGE_WIDTH, PAGE_HEIGHT, colour="white")
    else:
        b.spacial("M", borderless_region, sizing=map_width_a)
        # topo, water, overlay cpt scale
        b.basemap(land="lightgray", topo_cpt="grey1", resource_region=region_code)
        # map margins are semi-transparent
        b.background(
            map_width_a,
            map_height_a,
            colour="white@25",
            spacial=True,
            window=(MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM),
        )
    # leave space for left tickmarks and bottom colour scale
    b.spacial(
        "M", ll_region, sizing=map_width, x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM
    )
    if args.borders:
        # topo, water, overlay cpt scale
        b.basemap(land="lightgray", topo_cpt="grey1", resource_region=region_code)
    # title, fault model and velocity model subtitles
    b.text(sum(ll_region[:2]) / 2.0, ll_region[3], args.title, size=20, dy=0.6)
    b.text(ll_region[0], ll_region[3], args.subtitle1, size=14, align="LB", dy=0.3)
    b.text(
        ll_region[0],
        ll_region[3],
        args.subtitle2.replace("<HH>", str(xyts.hh)),
        size=14,
        align="LB",
        dy=0.1,
    )
    # cpt scale
    b.cpt_scale(
        "R",
        "B",
        cpt_file,
        cpt_inc,
        cpt_inc,
        label=args.legend,
        length=map_height,
        horiz=False,
        pos="rel_out",
        align="LB",
        thickness=0.3,
        dx=0.3,
        arrow_f=cpt_max > 0,
        arrow_b=cpt_min < 0,
    )
    # stations - split into real and virtual
    if args.stations is not None:
        with open(args.stations, "r") as sf:
            stations = sf.readlines()
        stations_real = []
        stations_virtual = []
        for stat in stations:
            if len(stat.split()[-1]) == 7:
                stations_virtual.append(stat)
            else:
                stations_real.append(stat)
        b.points(
            "".join(stations_real),
            is_file=False,
            shape="t",
            size=0.08,
            fill=None,
            line="white",
            line_thickness=0.8,
        )
        b.points(
            "".join(stations_virtual),
            is_file=False,
            shape="c",
            size=0.02,
            fill="black",
            line=None,
        )
    # render
    b.finalise()
    b.png(dpi=args.dpi, clip=False, out_dir=gmt_temp)
    rmtree(bwd)
    print("bottom template completed in %.2fs" % (time() - t0))


def top_template():
    t0 = time()
    twd = os.path.join(gmt_temp, "top")
    os.makedirs(twd)
    t = gmt.GMTPlot("%s/top.ps" % (twd))
    gmt.gmt_defaults(wd=twd, ps_media="Custom_%six%si" % (PAGE_WIDTH, PAGE_HEIGHT))
    t.spacial(
        "M", ll_region, sizing=map_width, x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM
    )
    # locations in NZ
    if ll_region[1] - ll_region[0] > 3:
        t.sites(gmt.sites_major)
    else:
        t.sites(list(gmt.sites.keys()))
    t.coastlines()
    # simulation domain
    t.path(cnr_str, is_file=False, split="-", close=True, width="0.4p", colour="black")
    # fault path
    if args.srf is not None:
        t.fault(args.srf, is_srf=True, plane_width=0.5, top_width=1, hyp_width=0.5)
    # ticks on top otherwise parts of map border may be drawn over
    major, minor = gmt.auto_tick(ll_region[0], ll_region[1], map_width)
    t.ticks(major=major, minor=minor, sides="ws")
    # render
    t.finalise()
    t.png(dpi=args.dpi, clip=False, out_dir=gmt_temp)
    rmtree(twd)
    print("top template completed in %.2fs" % (time() - t0))


def render_slice(n):
    t0 = time()

    # process working directory
    swd = "%s/ts%.4d" % (gmt_temp, n)
    os.makedirs(swd)

    s = gmt.GMTPlot("%s/ts%.4d.ps" % (swd, n), reset=False)
    gmt.gmt_defaults(wd=swd, ps_media="Custom_%six%si" % (PAGE_WIDTH, PAGE_HEIGHT))
    s.spacial(
        "M", ll_region, sizing=map_width, x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM
    )

    # timestamp text
    s.text(
        ll_region[1],
        ll_region[3],
        "t=%.2fs" % (n * xyts.dt),
        align="RB",
        size="14p",
        dy=0.1,
    )
    # overlay
    if args.xyts2 is None:
        xyts.tslice_get(n, comp=-1, outfile="%s/ts.bin" % (swd))
    else:
        x1 = xyts.tslice_get(n, comp=-1)
        x2 = xyts2.tslice_get(n, comp=-1)
        x2[:, 2] -= x1[:, 2]
        x2.astype(np.float32).tofile(os.path.join(swd, "ts.bin"))
    s.clip(cnr_str, is_file=False)
    if args.land_crop:
        s.clip(gmt.regional_resource("NZ", resource="coastline"), is_file=True)
    gmt.table2grd(
        os.path.join(swd, "ts.bin"),
        os.path.join(swd, "ts.grd"),
        dx=grd_dxy,
        wd=swd,
        climit=convergence_limit,
    )
    s.overlay(
        os.path.join(swd, "ts.grd"),
        cpt_file,
        dx=grd_dxy,
        dy=grd_dxy,
        min_v=lowcut,
        max_v=highcut,
        contours=cpt_inc,
    )
    s.clip()

    # add seismograms if wanted
    if args.seis is not None:
        # TODO: if used again, look into storing params inside seismo file
        s.seismo(os.path.abspath(args.seis), n, fmt="time", colour="red", width="1p")

    # create PNG
    s.finalise()
    s.png(dpi=args.dpi, clip=False, out_dir=png_dir)
    # cleanup
    rmtree(swd)
    print("timeslice %.4d completed in %.2fs" % (n, time() - t0))


def combine_slice(n):
    """
    Sandwitch midde layer (time dependent) between basemap and top (labels etc).
    """
    png = "%s/ts%.4d.png" % (png_dir, n)
    mid = "%s/bm%.4d.png" % (png_dir, n)
    gmt.overlay("%s/bottom.png" % (gmt_temp), png, mid)
    gmt.overlay(mid, "%s/top.png" % (gmt_temp), png)


###
### start rendering
###
ts0 = time()
pool = Pool(args.nproc)
# shared bottom and top layers
b_template = pool.apply_async(bottom_template, ())
t_template = pool.apply_async(top_template, ())
# middle layers
pool.map(render_slice, range(xyts.t0, xyts.nt - xyts.t0))
# wait for bottom and top layers
print("waiting for templates to finish...")
t_template.get()
b_template.get()
print("templates finished, combining layers...")
# combine layers
pool.map(combine_slice, range(xyts.t0, xyts.nt - xyts.t0))
print("layers combined, creating animation...")
# images -> animation
gmt.make_movie(
    "%s/ts%%04d.png" % (png_dir),
    args.output,
    fps=int(1.0 / xyts.dt * args.scale),
    codec="libx264",
)
print("finished.")
# cleanup
rmtree(gmt_temp)
