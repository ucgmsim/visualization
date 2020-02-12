#!/usr/bin/env python

from argparse import ArgumentParser
from tempfile import mkdtemp
import os
from shutil import rmtree
from subprocess import call

from h5py import File as h5open
import numpy as np
from osgeo import gdal

from qcore import gmt


parser = ArgumentParser()
parser.add_argument("geotiff_file", help="GeoTIFF file to plot")
parser.add_argument("--band", help="GeoTIFF band to plot", type=int, default=1)
args = parser.parse_args()

raster = gdal.Open(args.geotiff_file, gdal.GA_ReadOnly)
band = raster.GetRasterBand(1)
array = band.ReadAsArray()
ds = band.GetDataset()
geo_trans = ds.GetGeoTransform()
wkt_proj = ds.GetProjection()
nx = ds.RasterXSize
ny = ds.RasterYSize
minx = geo_trans[0]
miny = geo_trans[3] + nx * geo_trans[4] + ny * geo_trans[5]
maxx = geo_trans[0] + nx * geo_trans[1] + ny * geo_trans[2]
maxy = geo_trans[3]

wd = mkdtemp()
temp_nc = os.path.join(wd, "temp_tiff.nc")
temp_cpt = os.path.join(wd, "temp.cpt")
call(["gmt", "grdconvert", args.geotiff_file, "-G" + temp_nc])
gmt.makecpt("viridis", temp_cpt, 0, 800)

# would be much faster but need to add metadata
#with h5open("temp_tiff.nc") as g:
#    g["x"] = np.linspace(minx, maxx, nx)
#    g["x"].attrs["CLASS"] = b'DIMENSION_SCALE'
#    g["x"].attrs["NAME"] = b'x'
#    g["x"].attrs["long_name"] = b'x'
#    g["x"].attrs["actual_range"] = np.array([minx, maxx])
#
#    g["y"] = np.linspace(miny, maxy, ny)
#    g["y"].attrs["CLASS"] = b'DIMENSION_SCALE'
#    g["y"].attrs["NAME"] = b'y'
#    g["y"].attrs["long_name"] = b'y'
#    g["y"].attrs["actual_range"] = np.array([miny, maxy])

#    g["z"] = array

p = gmt.GMTPlot(os.path.join(wd, os.path.splitext(os.path.basename(args.geotiff_file))[0] + ".ps"))
p.spacial("X", (minx, maxx, miny, maxy), sizing="12", x_shift=1, y_shift=1)
p.overlay(temp_nc, temp_cpt)
p.ticks(major="500000", minor="100000")
p.cpt_scale("C", "B", temp_cpt, pos="rel_out", dy=0.5, major=200, minor=50)

p.finalise()
p.png(background="white", out_dir=".", dpi=1200)
rmtree(wd)
