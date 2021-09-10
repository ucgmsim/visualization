#!/usr/bin/env python3
"""
Generate non_uniform.xyz and sim/obs.xyz file
"""

import os
import argparse

from qcore import shared, utils, constants, formats


COMPS = list(constants.Components.iterate_str_values())


def validate_filepath(parser, file_path):
    """
    validates input file path
    :param parser:
    :param file_path: user input
    :return: parser error if error
    """
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r") as f:
                return
        except (IOError, OSError):
            parser.error("Can't open {}".format(file_path))
    else:
        parser.error("No such file {}".format(file_path))


def validate_dir(parser, dir_path):
    """
    validates a dir
    :param parser:
    :param dir_path: user input
    :return:
    """
    if not os.path.isdir(dir_path):
        parser.error("No such directory {}".format(dir_path))


def write_xyz(imcsv, stat_file, out_dir):
    utils.setup_dir(out_dir)

    stat_df = formats.load_station_file(stat_file)
    im_df = formats.load_im_file_pd(imcsv, comp=args.component)

    # must have compatible index names to merge
    stat_df.index.rename("station", inplace=True)

    xyz_df = im_df.merge(stat_df, left_index=True, right_index=True, how="inner")
    xyz_real_station_df = xyz_df[
        [
            not shared.is_virtual_station(stat)
            for stat in xyz_df.index.get_level_values(0)
        ]
    ]

    ims = im_df.columns
    columns = ["lon", "lat", *ims]

    non_uniform_filepath = os.path.join(out_dir, "non_uniform_im.xyz")
    real_station_filepath = os.path.join(out_dir, "real_station_im.xyz")

    xyz_df[columns].to_csv(non_uniform_filepath, sep=" ", header=None, index=None)
    xyz_real_station_df[columns].to_csv(
        real_station_filepath, sep=" ", header=None, index=None
    )

    im_col_file = os.path.join(out_dir, "im_order.txt")
    with open(im_col_file, "w") as fp:
        fp.write(" ".join(ims))

    print("xyz files are output to {}".format(out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imcsv_filepath", help="path to input IMcsv file")
    parser.add_argument("station_filepath", help="path to input station_ll file path")
    parser.add_argument(
        "-o",
        "--out_dir",
        default=".",
        help="path to store output xyz files. Defaults to CWD",
    )

    args = parser.parse_args()
    validate_filepath(parser, args.imcsv_filepath)
    validate_filepath(parser, args.station_filepath)
    write_xyz(args.imcsv_filepath, args.station_filepath, args.out_dir)
