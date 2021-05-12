#!/usr/bin/env python

from argparse import ArgumentParser

import pandas as pd

parser = ArgumentParser()
parser.add_argument("a", help="path to first IM csv")
parser.add_argument("b", help="path to second IM csv")
parser.add_argument("result", help="path to ratio IM csv")
args = parser.parse_args()

# load IMs and common properties
im_df_a = pd.read_csv(args.a, header=0)
im_df_b = pd.read_csv(args.b, header=0)
ims_a = im_df_a.columns[2:]
ims_b = im_df_b.columns[2:]
ims = []
for im in ims_a:
    if im in ims_b:
        ims.append(im)

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
