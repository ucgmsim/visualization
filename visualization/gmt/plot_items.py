#!/usr/bin/env python
"""
plot_items.py plots items given as parameters.

add srf files from 2 locations:
-s "/folder1/*/Srf/*.srf" -s "location2/*.srf"

add vm corners created with srfinfo2vm:
-v "autovm/*/VeloModCorners.txt"

for more information on parameters, use -h
"""
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
import os
from shutil import copy, rmtree
import sys
from tempfile import mkdtemp

from h5py import File as h5open
import numpy as np

from qcore import geo
from qcore import gmt
from qcore import srf
from qcore import xyts

script_dir = os.path.abspath(os.path.dirname(__file__))

def get_args():
    parser = ArgumentParser()
    arg = parser.add_argument
    arg("-t", "--title", help="title text", default="")
    arg(
        "-f", "--filename", default="plot_items", help="output filename excluding extention"
    )
    arg("--fast", help="no topography, low resolution coastlines", action="store_true")
    arg(
        "-s",
        "--srf-files",
        action="append",
        help="SRF files to plot, use wildcards, repeat as needed",
    )
    arg(
        "-c",
        "--srf-only-outline",
        action="append",
        help="SRF files to plot only outline, use wildcards, repeat as needed",
    )
    arg("--fault-colour", help="outline colour of faults", default="black")
    arg(
        "--outline-fault-colour",
        help="outline colour of only-outline faults",
        default="blue",
    )
    arg("-b", "--bb-scale", help="beachball scale", type=float, default=0.05)
    arg(
        "--slip-max", help="maximum slip (cm/s) on colour scale", type=float, default=1000.0
    )
    arg("-r", "--region", help="Region to plot in the form xmin/xmax/ymin/ymax.")
    arg(
        "-v",
        "--vm-corners",
        action="append",
        help="VeloModCorners.txt to plot, use wildcards, repeat as needed",
    )
    arg(
        "-x",
        "--xyts-corners",
        action="append",
        help="xyts.e3d to plot outlines for, use wildcards, repeat as needed")
    arg("--xyz", help="path to file containing lon, lat, value_1 .. value_N")
    arg("--xyz-size", help="size of points or grid spacing eg: 1c or 1k")
    arg("--xyz-cpt", help="CPT to use for overlay data", default="hot")
    arg("--xyz-cpt-invert", help="inverts CPT", action="store_true")
    arg("--xyz-cpt-min", help="CPT minimum value")
    arg("--xyz-cpt-max", help="CPT maximum value")
    arg("--xyz-cpt-inc", help="CPT colour increments")
    arg("--xyz-cpt-tick", help="CPT legend annotation spacing")
    arg("--xyz-cpt-bg", help="overlay colour below CPT min")
    arg("--xyz-cpt-fg", help="overlay colour above CPT max")
    arg("--xyz-grid", help="display as grid instead of points", action="store_true")
    arg("--xyz-grid-type", help="interpolation program to use", default="surface")
    arg("--xyz-grid-nns", help="search radius for interpolation")
    arg("-n", "--nproc", help="max number of processes", type=int, default=1)
    arg("-d", "--dpi", help="render DPI", type=int, default=300)

    return parser.parse_args()


# load srf files for plotting
def load_srf(i_srf):
    """
    Prepare data in a format required for plotting.
    i_srf: index, srf path
    """
    # point source - save beachball data
    if not srf.is_ff(i_srf[1]):
        info = "%s.info" % os.path.splitext(i_srf[1])[0]
        if not os.path.exists(info):
            print("ps SRF missing .info, using 5.0 for magnitude: %s" % (i_srf[1]))
            mag = 5.0
            hypocentre = srf.get_hypo(i_srf[1], depth=True)
            strike, dip, rake = srf.ps_params(i_srf[1])
        else:
            with h5open(info) as h:
                mag = h.attrs["mag"]
                hypocentre = h.attrs["hlon"], h.attrs["hlat"], h.attrs["hdepth"]
                strike = h.attrs["strike"][0]
                dip = h.attrs["dip"][0]
                rake = h.attrs["rake"]
        with open("%s/beachball%d.bb" % (gmt_temp, i_srf[0]), "w") as bb:
            bb.write(
                "%s %s %s %s %s %s %s %s %s\n"
                % (
                    hypocentre[0],
                    hypocentre[1],
                    hypocentre[2],
                    strike,
                    dip,
                    rake,
                    mag,
                    hypocentre[0],
                    hypocentre[1],
                )
            )
        return
    # finite fault - only save outline
    if i_srf[0] < 0:
        srf.srf2corners(i_srf[1], cnrs="%s/srf%d.cnrs-X" % (gmt_temp, i_srf[0]))
        return
    # finite fault - save outline and slip distributions
    srf.srf2corners(i_srf[1], cnrs="%s/srf%d.cnrs" % (gmt_temp, i_srf[0]))
    proc_tmp = "%s/srf2map_%d" % (gmt_temp, i_srf[0])
    os.makedirs(proc_tmp)
    try:
        srf_data = gmt.srf2map(
            i_srf[1], gmt_temp, prefix="srf%d" % (i_srf[0]), wd=proc_tmp
        )
        return i_srf[0], srf_data
    except ValueError:
        # vertical dip
        return


def load_xyts_corners(xyts_path):
    x = xyts.XYTSFile(xyts_path[0], meta_only=True)
    return x.corners(gmt_format=True)[1]

def load_xyz(args):
    if args.xyz is None:
        return
    xyz_info = {}
    xy = np.loadtxt(args.xyz, usecols=(0, 1), dtype="f")
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    xyz_info["region"] = x_min, x_max, y_min, y_max
    if args.xyz_size is None:
        # very rough default
        xyz_info["dxy"] = str(geo.ll_dist(xy[0][0], xy[0][1], xy[1][0], xy[1][1]) * 0.7) + "k"
    else:
        xyz_info["dxy"] = args.xyz_size
    # these corners are without projection, will hold all points, should be cropped
    corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    #gmt.grd_mask(
    #    "%s/sim.modelpath" % (gmt_temp),
    #    "%s/mask.grd" % (gmt_temp),
    #    dx=xyz_info["dxy"],
    #    dy=xyz_info["dxy"],
    #    region=xyz_info["region"],
    #)
    return xyz_info


def load_xyz_col(args_info_i):
    args, xyz_info, i = args_info_i
    swd = os.path.join(gmt_temp, "_%d_" % (i))
    copy("%s/gmt.conf" % (gmt_temp), swd)
    copy("%s/gmt.history" % (gmt_temp), swd)
    

    xyz_val = np.loadtxt(args.xyz, usecols=(0, 1, i + 2), dtype="f")
    # prepare cpt
    cpt_max = np.percentile(xyz_val[:, 2], 99.5)
    if cpt_max > 115:
        # 2 significant figures
        cpt_max = round(
            cpt_max, 1 - int(np.floor(np.log10(abs(cpt_max))))
        )
    else:
        # 1 significant figures
        cpt_max = round(
            cpt_max, -int(np.floor(np.log10(abs(cpt_max))))
        )
    if xyz_val[:, 2].min() < 0:
        cpt_min = -cpt_max
    else:
        cpt_min = 0
    cpt_inc = (cpt_max / 10.0)
    cpt_tick = (cpt_inc * 2.0)
    col_cpt = "%s/stations_%d.cpt" % (swd, i)
    # overlay colour scale
    gmt.makecpt(
        args.xyz_cpt,
        col_cpt,
        cpt_min,
        cpt_max,
        inc=cpt_inc,
        invert=args.xyz_cpt_invert,
        wd=swd,
    )
    

def find_srfs(args, gmt_temp):
    # find srf
    srf_files = []
    if args.srf_only_outline is not None:
        for ex in args.srf_only_outline:
            srf_files.extend(glob(ex))
    # to determine if srf_file is only outline or full
    srf_0 = -len(srf_files)
    if args.srf_files is not None:
        for ex in args.srf_files:
            srf_files.extend(glob(ex))

    # slip cpt
    if abs(srf_0) == len(srf_files):
        # will be plotting slip
        slip_cpt = "%s/slip.cpt" % (gmt_temp)
        gmt.makecpt(gmt.CPTS["slip"], slip_cpt, 0, args.slip_max)

    return srf_files, srf_0


def find_xyts(args):
    xyts_files = []
    if args.xyts_corners is not None:
        for ex in args.xyts_corners:
            xyts_files.extend(glob(ex))
    
    return xyts_files


def find_xyz_ncol(xyz_file):
    if xyz_file is None:
        return 0

    with open(xyz_file) as x:
        return len(x.readline().split()) - 2


def load_vm_corners(args):
    # find/load vm corners
    vm_corners = []
    if args.vm_corners is not None:
        for ex in args.vm_corners:
            vm_corners.extend(glob(ex))
    vm_corners = "\n>\n".join(
        [
            "\n".join(
                [
                    " ".join(map(str, v))
                    for v in np.loadtxt(c, skiprows=2, dtype=np.float32).tolist()
                ]
            )
            for c in vm_corners
        ]
    )
    return vm_corners

args = get_args()
gmt_temp = mkdtemp()
srf_files, srf_0 = find_srfs(args, gmt_temp)
xyts_files = find_xyts(args)
xyz_ncol = find_xyz_ncol(args.xyz)

pool = Pool(args.nproc)
xyz_info = pool.apply_async(load_xyz, [args])
i_srf_data = pool.map_async(load_srf, zip(range(srf_0, srf_0 + len(srf_files)), srf_files))
xyts_corners = pool.map_async(load_xyts_corners, xyts_files)
vm_corners = load_vm_corners(args)

xyz_info = xyz_info.get()
xyz = pool.map_async(load_xyz_col, zip([args] * xyz_ncol, [xyz_info] * xyz_ncol, range(0, xyz_ncol)))

xyz = xyz.get()
exit()
i_srf_data = i_srf_data.get()
xyts_corners = xyts_corners.get()


def basemap(args, wd):
    ps_file = "%s/%s.ps" % (wd, args.filename)
    map_width = 9
    p = gmt.GMTPlot(ps_file)
    if args.region is None:
        region = gmt.nz_region
    else:
        region = map(float, args.region.split("/"))
    p.spacial("M", region, sizing="%si" % (map_width), x_shift=2, y_shift=2)
    if args.fast:
        p.basemap(res="f", land="lightgray", topo=None, road=None, highway=None)
    else:
        p.basemap(topo_cpt="grey1", land="lightgray")
    # border tick labels
    p.ticks(major=2, minor=0.2)
    # QuakeCoRE logo
    p.image("L", "T", "%s/quakecore-logo.png" % (script_dir), width="3i", pos="rel")

    return p

p = basemap(args, gmt_temp)

# plot velocity model corners
p.path(vm_corners, is_file=False, close=True, width="0.5p", split="-")

# add SRF slip
finite_faults = False
for i_s in i_srf_data:
    if i_s is None:
        continue
    finite_faults = True
    for plane in xrange(len(i_s[1][1])):
        p.overlay(
            "%s/srf%d_%d_slip.bin" % (gmt_temp, i_s[0], plane),
            slip_cpt,
            dx=i_s[1][0][0],
            dy=i_s[1][0][1],
            climit=2,
            crop_grd="%s/srf%d_%d_mask.grd" % (gmt_temp, i_s[0], plane),
            land_crop=False,
            transparency=35,
            custom_region=i_s[1][1][plane],
        )
# add outlines for SRFs with slip
for c in glob("%s/srf*.cnrs" % (gmt_temp)):
    p.fault(
        c,
        is_srf=False,
        hyp_size=0,
        plane_width=0.2,
        top_width=0.4,
        hyp_width=0.2,
        plane_colour=args.fault_colour,
        top_colour=args.fault_colour,
        hyp_colour=args.fault_colour,
    )
# add outlines for SRFs without slip
for c in glob("%s/srf*.cnrs-X" % (gmt_temp)):
    p.fault(
        c,
        is_srf=False,
        hyp_size=0,
        plane_width=0.2,
        top_width=0.4,
        hyp_width=0.2,
        plane_colour=args.outline_fault_colour,
        top_colour=args.outline_fault_colour,
        hyp_colour=args.outline_fault_colour,
    )
# add beach balls
for bb in glob("%s/beachball*.bb" % (gmt_temp)):
    p.beachballs(bb, is_file=True, fmt="a", scale=args.bb_scale)

# slip scale
if finite_faults:
    p.cpt_scale(
        "C",
        "B",
        slip_cpt,
        pos="rel_out",
        dy="0.5i",
        label="Slip (cm)",
        length=map_width * 0.618,
    )

# output
p.finalise()
p.png(out_name=os.path.abspath(args.filename), dpi=args.dpi, background="white")
rmtree(gmt_temp)
