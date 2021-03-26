#!/usr/bin/env python

from argparse import ArgumentParser
import os

import numpy as np
import yaml

from qcore import geo, gmt

parser = ArgumentParser()
parser.add_argument("vm_dir", help="path containing VM files", type=os.path.abspath)
parser.add_argument("vm_file", help="binary VM file to plot", type=os.path.abspath)
args = parser.parse_args()

with open(os.path.join(args.vm_dir, "vm_params.yaml")) as y:
    vm_conf = yaml.safe_load(y)

coords = (
    np.vstack(np.mgrid[0 : vm_conf["nx"], 0 : vm_conf["ny"]].T) * vm_conf["hh"]
    - (np.array([vm_conf["extent_x"], vm_conf["extent_y"]]) - vm_conf["hh"]) / 2
)
model_mat, model_mat_inv = geo.gen_mat(
    vm_conf["MODEL_ROT"], vm_conf["MODEL_LON"], vm_conf["MODEL_LAT"]
)
xyll = geo.xy2ll(coords, model_mat)

vm3d = np.memmap(
    args.vm_file, dtype="f4", shape=(vm_conf["ny"], vm_conf["nz"], vm_conf["nx"])
)
