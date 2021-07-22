#!/usr/bin/env python
"""
Create IM csv file containing the ratio of 2 IM csv files.
"""

from argparse import ArgumentParser

import numpy as np
from pathlib import Path
from qcore.formats import load_im_file_pd


def ratios_to_csv(
    imcsv1_path: Path, imcsv2_path: Path, output_path: Path, comp=None, summary=False
):
    # load IMs and common properties
    im_df_a = load_im_file_pd(imcsv1_path, comp=comp)
    im_df_b = load_im_file_pd(imcsv2_path, comp=comp)
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
    ratio_df.to_csv(output_path, index=True)
    if summary:
        summary_file = output_path.parent / f"{output_path.stem}_summary.csv"
        ratio_df.describe().to_csv(summary_file)
    return ratio_df.columns, (ratio_df.min(), ratio_df.max())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("imcsv1", help="path to first IM csv")
    parser.add_argument("imcsv2", help="path to second IM csv")
    parser.add_argument("output", type=Path, help="path to output IM csv")
    parser.add_argument("--summary", action="store_true", default=False)
    args = parser.parse_args()

    ratios_to_csv(args.imcsv1, args.imcsv2, args.output, summary=args.output)
