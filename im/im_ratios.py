#!/usr/bin/env python
"""
Create IM csv file containing the ration of 2 IM csv files.
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument("imcsv1", help="path to first IM csv")
parser.add_argument("imcsv2", help="path to second IM csv")
parser.add_argument("result", help="path to result IM csv")
args = parser.parse_args()

# load IMs and common properties
im_df_a = pd.read_csv(args.imcsv1, header=0)
im_df_b = pd.read_csv(args.imcsv2, header=0)
ims = np.intersect1d(im_df_a.columns[2:], im_df_b.columns[2:])

# add both IMs into colums IM_x and IM_y
ratio_df = pd.merge(left=im_df_a, right=im_df_b, on="station")
# process all IMs
for im in ims:
    if im.endswith("_sigma"):
        ratio_df[im] = ratio_df[f"{im}_x"] - ratio_df[f"{im}_y"]
    else:
        ratio_df[im] = ratio_df[f"{im}_x"] / ratio_df[f"{im}_y"]

# tidy up
ratio_df["component"] = ratio_df["component_x"]
ratio_columns = ["station", "component"]
ratio_columns.extend(ims)
ratio_df = ratio_df[ratio_columns]

# save
ratio_df.to_csv(args.result, index=False)
