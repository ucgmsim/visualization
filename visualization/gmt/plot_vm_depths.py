#!/usr/bin/env python
"""
Plots VM 3D files xy planes over specified distances.
"""

from argparse import ArgumentParser
from math import floor
import os
from shutil import copyfile, rmtree
from tempfile import mkdtemp

import numpy as np
import yaml

from qcore import geo, gmt

# space around map for titles, tick labels and scales etc
MARGIN_TOP = 1.0
MARGIN_BOTTOM = 0.4
MARGIN_LEFT = 1.0
MARGIN_RIGHT = 1.7


def load_args():
    """
    Command line arguments and VM configuration.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "vm_params", help="path to vm_params.yaml", type=os.path.abspath
    )
    parser.add_argument("vm_file", help="binary VM file to plot", type=os.path.abspath)
    parser.add_argument("--depth", nargs="+", type=float, default=[2, 5, 10, 20])
    parser.add_argument("--out-dir", help="output location", default="./vm_depths")
    parser.add_argument("--cpt", help="overlay cpt", default="hot")
    parser.add_argument("--cpt-invert", help="invert cpt range", action="store_false")
    parser.add_argument("--legend", help="colour scale legend text")
    parser.add_argument(
        "--page-height", help="height of figure * dpi = pixels", type=float, default=9
    )
    parser.add_argument(
        "--page-width", help="width of figure * dpi = pixels", type=float, default=16
    )
    parser.add_argument(
        "--dpi", help="dpi 80: 720p, [120]: 1080p, 240: 4k", type=int, default=120
    )
    parser.add_argument("--borders", help="opaque map margins", action="store_true")
    parser.add_argument(
        "--downscale",
        type=int,
        default=8,
        help="downscale factor prevents jitter/improves filtering",
    )
    args = parser.parse_args()

    with open(os.path.join(args.vm_params)) as y:
        vm_conf = yaml.safe_load(y)

    return args, vm_conf


def process_coords(vm_conf):
    """
    Determine datas longitude and latitude positions.
    """
    # create lat, lon grid
    xy = (
        np.vstack(np.mgrid[0 : vm_conf["nx"], 0 : vm_conf["ny"]].T) * vm_conf["hh"]
        - (np.array([vm_conf["extent_x"], vm_conf["extent_y"]]) - vm_conf["hh"]) / 2
    )
    model_mat = geo.gen_mat(
        vm_conf["MODEL_ROT"], vm_conf["MODEL_LON"], vm_conf["MODEL_LAT"]
    )[0]
    xyll = geo.xy2ll(xy, model_mat).reshape(vm_conf["ny"], vm_conf["nx"], 2)

    # determine extents
    xyll_shift = np.copy(xyll)
    xyll_shift[:, :, 0][xyll_shift[:, :, 0] < 0] += 360
    xyll_shift[:, :, 1][xyll_shift[:, :, 1] < 0] += 180
    xmin, ymin = np.min(xyll_shift, axis=(1, 0))
    xmax, ymax = np.max(xyll_shift, axis=(1, 0))
    ll_region = (
        xmin if xmin <= 180 else xmin - 360,
        xmax if xmax <= 180 or xmin <= 180 else xmax - 360,
        ymin if ymin <= 90 else ymin - 180,
        ymax if ymax <= 90 or ymin <= 90 else ymax - 180,
    )
    corners = xyll[
        [0, 0, vm_conf["ny"] - 1, vm_conf["ny"] - 1],
        [0, vm_conf["nx"] - 1, vm_conf["nx"] - 1, 0],
    ]
    return xyll, ll_region, corners


def map_sizing(args, ll_region, gmt_temp):
    """
    Determine map sizing and region.
    """
    map_width = args.page_width - MARGIN_LEFT - MARGIN_RIGHT
    map_height = args.page_height - MARGIN_TOP - MARGIN_BOTTOM
    # extend region to fill view window
    map_width, map_height, ll_region = gmt.fill_space(
        map_width, map_height, ll_region, proj="M", dpi=args.dpi, wd=gmt_temp
    )
    return map_width, map_height, ll_region


def template_gs(args, corners_gmt, map_width, map_height, gmt_temp):
    """
    Create the common background plot.
    """
    gs_file = "%s/template.ps" % (gmt_temp)
    p = gmt.GMTPlot(gs_file)
    gmt.gmt_defaults(
        wd=gmt_temp, ps_media="Custom_%six%si" % (args.page_width, args.page_height)
    )
    if args.borders:
        p.background(args.page_width, args.page_height, colour="white")
    else:
        # extend map to cover margins
        map_width_a, map_height_a, borderless_region = gmt.fill_margins(
            ll_region,
            map_width,
            args.dpi,
            left=MARGIN_LEFT,
            right=MARGIN_RIGHT,
            top=MARGIN_TOP,
            bottom=MARGIN_BOTTOM,
        )
        p.spacial("M", borderless_region, sizing=map_width_a)
        # topo, water, overlay cpt scale
        p.basemap(land="lightgray", topo_cpt="grey1", scale=args.downscale)
        # map margins are semi-transparent
        p.background(
            map_width_a,
            map_height_a,
            colour="white@25",
            spacial=True,
            window=(MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM),
        )
    # leave space for left tickmarks and bottom colour scale
    p.spacial(
        "M", ll_region, sizing=map_width, x_shift=MARGIN_LEFT, y_shift=MARGIN_BOTTOM
    )
    if args.borders:
        # topo, water, overlay cpt scale
        p.basemap(land="lightgray", topo_cpt="grey1", scale=args.downscale)
    # title, fault model and velocity model subtitles
    p.text(sum(ll_region[:2]) / 2.0, ll_region[3], "Velocity Model", size=20, dy=0.6)
    p.text(
        ll_region[0],
        ll_region[3],
        "NZVM v{} h={}km".format(vm_conf["model_version"], vm_conf["hh"]),
        size=14,
        align="LB",
        dy=0.3,
    )
    p.text(
        ll_region[0],
        ll_region[3],
        os.path.basename(args.vm_file),
        size=14,
        align="LB",
        dy=0.1,
    )
    p.leave()
    return gs_file


def finish_gs(
    args,
    vm_conf,
    gs_file,
    depth,
    map_width,
    map_height,
    xyll,
    vm3d,
    corners_gmt,
    ll_region0,
    gmt_temp,
):
    """
    Add features specific to current depth plot.
    """
    depth_ix = floor(0.5 + depth / vm_conf["hh"])
    if depth_ix >= vm_conf["nz"]:
        print("skipping depth", depth, "out of range for VM")
        return
    depth_value = (0.5 + depth_ix) * vm_conf["hh"]
    depth_wd = os.path.join(gmt_temp, str(depth))
    os.makedirs(depth_wd)
    depth_gs = (
        f"{depth_wd}/{os.path.basename(args.vm_file).replace('.', '_')}_{depth}.ps"
    )
    copyfile(gs_file, depth_gs)
    for setup in ["gmt.conf", "gmt.history"]:
        copyfile(os.path.join(gmt_temp, setup), os.path.join(depth_wd, setup))
    p = gmt.GMTPlot(depth_gs, append=True, reset=False)
    xyz_bin = os.path.join(depth_wd, "xyz.bin")
    surface = np.column_stack((xyll.reshape(-1, 2), vm3d[:, depth_ix, :].flatten()))
    surface.astype(np.float32).tofile(xyz_bin)
    cpt_inc, cpt_max = gmt.xyv_cpt_range(xyz_bin)[1:3]
    cpt_file = os.path.join(depth_wd, "cpt.cpt")
    cpt_min = 0
    # colour scale
    gmt.makecpt(
        args.cpt,
        cpt_file,
        cpt_min,
        cpt_max,
        continuous=True,
        invert=args.cpt_invert,
        bg=None,
        fg=None,
    )
    p.clip(path=corners_gmt)
    spacing = "{}k".format(vm_conf["hh"] * 0.4)
    p.overlay(xyz_bin, cpt_file, dx=spacing, dy=spacing, custom_region=ll_region0)
    p.clip()
    p.text(
        ll_region[1],
        ll_region[3],
        f"depth: {depth_value:.2f}km",
        size=14,
        align="RB",
        dy=0.1,
    )
    p.cpt_scale(
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
    p.path(corners_gmt, is_file=False, close=True, width="1p", split="-")

    # ticks on top otherwise parts of map border may be drawn over
    major, minor = gmt.auto_tick(ll_region[0], ll_region[1], map_width)
    p.ticks(major=major, minor=minor, sides="ws")
    # render
    p.finalise()
    p.png(
        dpi=args.dpi * args.downscale,
        downscale=args.downscale,
        clip=False,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    args, vm_conf = load_args()
    xyll, ll_region0, corners = process_coords(vm_conf)
    # locations
    os.makedirs(args.out_dir, exist_ok=True)
    gmt_temp = mkdtemp()

    map_width, map_height, ll_region = map_sizing(args, ll_region0, gmt_temp)
    corners_gmt = "\n".join([" ".join(map(str, point)) for point in corners])
    vm3d = np.memmap(
        args.vm_file, dtype="f4", shape=(vm_conf["ny"], vm_conf["nz"], vm_conf["nx"])
    )
    gs_file = template_gs(args, corners_gmt, map_width, map_height, gmt_temp)
    for depth in args.depth:
        finish_gs(
            args,
            vm_conf,
            gs_file,
            depth,
            map_width,
            map_height,
            xyll,
            vm3d,
            corners_gmt,
            ll_region0,
            gmt_temp,
        )

    rmtree(gmt_temp)
