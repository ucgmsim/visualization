#!/usr/bin/env python
"""
Create IM csv file containing the ratio of 2 IM csv files.
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument("imcsv1", help="path to first IM csv")
parser.add_argument("imcsv2", help="path to second IM csv")
parser.add_argument("output", help="path to output IM csv")
args = parser.parse_args()

# load IMs and common properties
im_df_a = pd.read_csv(args.imcsv1, index_col=[0, 1])
im_df_b = pd.read_csv(args.imcsv2, index_col=[0, 1])
ims = np.intersect1d(im_df_a.columns, im_df_b.columns)

# add both IMs into colums IM_x and IM_y
ratio_df = im_df_a.merge(im_df_b, left_index=True, right_index=True)
# process all IMs
for im in ims:
    if im.endswith("_sigma"):
        ratio_df[im] = ratio_df[f"{im}_x"] - ratio_df[f"{im}_y"]
    else:
        ratio_df[im] = np.log(ratio_df[f"{im}_x"] / ratio_df[f"{im}_y"])

# tidy up
ratio_df = ratio_df[ims]

# save
ratio_df.to_csv(args.output, index=True)
