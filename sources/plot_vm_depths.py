#!/usr/bin/env python
"""
Plots VM 3D files xy planes over specified distances.
"""

from argparse import ArgumentParser
from math import floor
import os
from shutil import copyfile, rmtree
from tempfile import TemporaryDirectory

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
        "--page-height", help="height of figure (inches)", type=float, default=9
    )
    parser.add_argument(
        "--page-width", help="width of figure (inches)", type=float, default=16
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

    # configuration of view window
    window_conf = {
        "borders": args.borders,
        "downscale": args.downscale,
        "page_width": args.page_width,
        "page_height": args.page_height,
        "dpi": args.dpi,
    }
    # configuration of plot contents
    plot_conf = {
        "cpt": args.cpt,
        "cpt_invert": args.cpt_invert,
        "legend": args.legend,
        "depths": args.depth,
    }
    # locations
    path_conf = {
        "out_dir": args.out_dir,
        "vm_file": args.vm_file,
    }

    with open(os.path.join(args.vm_params)) as y:
        vm_conf = yaml.safe_load(y)

    return window_conf, plot_conf, path_conf, vm_conf


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


def map_sizing(window_conf, ll_region, gmt_temp):
    """
    Determine map sizing and region.
    """
    map_width = window_conf["page_width"] - MARGIN_LEFT - MARGIN_RIGHT
    map_height = window_conf["page_height"] - MARGIN_TOP - MARGIN_BOTTOM
    # extend region to fill view window
    map_width, map_height, ll_region = gmt.fill_space(
        map_width, map_height, ll_region, proj="M", dpi=window_conf["dpi"], wd=gmt_temp
    )
    window_conf["map_width"] = map_width
    window_conf["map_height"] = map_height
    window_conf["ll_region"] = ll_region


def template_gs(window_conf, path_conf):
    """
    Create the common background plot.
    """
    gs_file = "%s/template.ps" % (path_conf["gmt_temp"])
    p = gmt.GMTPlot(gs_file)
    gmt.gmt_defaults(
        wd=path_conf["gmt_temp"],
        ps_media="Custom_%six%si"
        % (window_conf["page_width"], window_conf["page_height"]),
    )
    if window_conf["borders"]:
        p.background(
            window_conf["page_width"], window_conf["page_height"], colour="white"
        )
    else:
        # extend map to cover margins
        map_width_a, map_height_a, borderless_region = gmt.fill_margins(
            window_conf["ll_region"],
            window_conf["map_width"],
            window_conf["dpi"],
            left=MARGIN_LEFT,
            right=MARGIN_RIGHT,
            top=MARGIN_TOP,
            bottom=MARGIN_BOTTOM,
        )
        p.spacial("M", borderless_region, sizing=map_width_a)
        # topo, water, overlay cpt scale
        p.basemap(land="lightgray", topo_cpt="grey1", scale=window_conf["downscale"])
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
        "M",
        window_conf["ll_region"],
        sizing=window_conf["map_width"],
        x_shift=MARGIN_LEFT,
        y_shift=MARGIN_BOTTOM,
    )
    if window_conf["borders"]:
        # topo, water, overlay cpt scale
        p.basemap(land="lightgray", topo_cpt="grey1", scale=window_conf["downscale"])
    # title, fault model and velocity model subtitles
    p.text(
        sum(window_conf["ll_region"][:2]) / 2.0,
        window_conf["ll_region"][3],
        "Velocity Model",
        size=20,
        dy=0.6,
    )
    p.text(
        window_conf["ll_region"][0],
        window_conf["ll_region"][3],
        f"NZVM v{vm_conf['model_version']} h={vm_conf['hh']}km",
        size=14,
        align="LB",
        dy=0.3,
    )
    p.text(
        window_conf["ll_region"][0],
        window_conf["ll_region"][3],
        os.path.basename(path_conf["vm_file"]),
        size=14,
        align="LB",
        dy=0.1,
    )
    p.leave()
    return gs_file


def finish_gs(
    window_conf,
    plot_conf,
    path_conf,
    vm_conf,
    gs_file,
    depth,
    xyll,
    vm3d,
):
    """
    Add features specific to current depth plot.
    """
    depth_ix = floor(0.5 + depth / vm_conf["hh"])
    if depth_ix >= vm_conf["nz"]:
        print("skipping depth", depth, "out of range for VM")
        return
    depth_value = (0.5 + depth_ix) * vm_conf["hh"]
    depth_wd = os.path.join(path_conf["gmt_temp"], str(depth))
    os.makedirs(depth_wd)
    depth_gs = f"{depth_wd}/{os.path.basename(path_conf['vm_file']).replace('.', '_')}_{depth}.ps"
    copyfile(gs_file, depth_gs)
    for setup in ["gmt.conf", "gmt.history"]:
        copyfile(
            os.path.join(path_conf["gmt_temp"], setup), os.path.join(depth_wd, setup)
        )
    p = gmt.GMTPlot(depth_gs, append=True, reset=False)
    xyz_bin = os.path.join(depth_wd, "xyz.bin")
    surface = np.column_stack((xyll.reshape(-1, 2), vm3d[:, depth_ix, :].flatten()))
    surface.astype(np.float32).tofile(xyz_bin)
    cpt_inc, cpt_max = gmt.xyv_cpt_range(xyz_bin)[1:3]
    cpt_file = os.path.join(depth_wd, "cpt.cpt")
    cpt_min = 0
    # colour scale
    gmt.makecpt(
        plot_conf["cpt"],
        cpt_file,
        cpt_min,
        cpt_max,
        continuous=True,
        invert=plot_conf["cpt_invert"],
        bg=None,
        fg=None,
    )
    p.clip(path=plot_conf["corners_gmt"])
    spacing = f"{vm_conf['hh'] * 0.4}k"
    p.overlay(
        xyz_bin,
        cpt_file,
        dx=spacing,
        dy=spacing,
        custom_region=plot_conf["ll_region_data"],
    )
    p.clip()
    p.text(
        window_conf["ll_region"][1],
        window_conf["ll_region"][3],
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
        label=plot_conf["legend"],
        length=window_conf["map_height"],
        horiz=False,
        pos="rel_out",
        align="LB",
        thickness=0.3,
        dx=0.3,
        arrow_f=cpt_max > 0,
        arrow_b=cpt_min < 0,
    )
    p.path(plot_conf["corners_gmt"], is_file=False, close=True, width="1p", split="-")

    # ticks on top otherwise parts of map border may be drawn over
    major, minor = gmt.auto_tick(
        *window_conf["ll_region"][0:2], window_conf["map_width"]
    )
    p.ticks(major=major, minor=minor, sides="ws")
    # render
    p.finalise()
    p.png(
        dpi=window_conf["dpi"] * window_conf["downscale"],
        downscale=window_conf["downscale"],
        clip=False,
        out_dir=path_conf["out_dir"],
    )


if __name__ == "__main__":
    window_conf, plot_conf, path_conf, vm_conf = load_args()

    # extend configurations with derived values
    os.makedirs(path_conf["out_dir"], exist_ok=True)
    temp_object = TemporaryDirectory()
    path_conf["gmt_temp"] = temp_object.name
    xyll, plot_conf["ll_region_data"], corners = process_coords(vm_conf)
    map_sizing(window_conf, plot_conf["ll_region_data"], path_conf["gmt_temp"])
    plot_conf["corners_gmt"] = "\n".join(
        [" ".join(map(str, point)) for point in corners]
    )
    vm3d = np.memmap(
        path_conf["vm_file"],
        dtype="f4",
        shape=(vm_conf["ny"], vm_conf["nz"], vm_conf["nx"]),
    )

    # create plots
    gs_file = template_gs(window_conf, path_conf)
    for depth in plot_conf["depths"]:
        finish_gs(
            window_conf,
            plot_conf,
            path_conf,
            vm_conf,
            gs_file,
            depth,
            xyll,
            vm3d,
        )
