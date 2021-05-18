#!/usr/bin/env python3

"""
Assumption: (1) im_values.csv and im_values_[imcalc|empirical].info are in the same location and
            (2) .csv and _[imcalc|empirical].info have the same prefix

Generate non_uniform.xyz and sim/obs.xyz file

Command:
To generate .xyz:
python im_plot.py ~/darfield_sim/darfield_sim.csv /nesi/project/nesi00213/dev/impp_datasets/Darfield/non_uniform_whole_nz_with_real_stations-hh100_17062017.ll -o ~/test_emp_plot
python im_plot.py ~/darfield_sim/darfield_sim.csv ~/rrup.csv -o ~/test_emp_plot

To plot:
python plot_stations.py ~/test_emp_plot/sim_im_plot_map_darfield_sim.xyz --srf_cnrs /nesi/project/nesi00213/dev/impp_datasets/Darfield/bev01_s103246Allsegm_v8_23.srf --model_params /nesi/project/nesi00213/dev/impp_datasets/Darfield/model_params_nz01-h0.100 --out_dir ~/test_emp_plot/sim_im_plot_map
"""

import os
import sys
import argparse
import glob

from qcore import shared, utils, constants, formats


COMPS = list(constants.Components.iterate_str_values())


def get_runtype(meta_filepath):
    """
    get the run type for output xyz filename from the '_[imcalc|empirical].info' metadata file
    :param meta_filepath: user input
    :return: run_type
    """
    with open(meta_filepath, "r") as meta_file:
        meta_file.readline()  # skip header
        run_type = meta_file.readline().strip().split(",")[2]
    return run_type


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imcsv_filepath", help="path to input IMcsv file")
    parser.add_argument("station_filepath", help="path to input station_ll file path")
    parser.add_argument(
        "-o",
        "--output_path",
        default=".",
        help="path to store output xyz files. Defaults to CWD",
    )
    parser.add_argument(
        "-c",
        "--component",
        default="geom",
        choices=COMPS,
        help=f"which component of the intensity measure. Available components are {COMPS}. Default is 'geom'",
    )
    args = parser.parse_args()

    utils.setup_dir(args.output_path)

    validate_filepath(parser, args.imcsv_filepath)
    validate_filepath(parser, args.station_filepath)
    validate_dir(parser, args.output_path)

    run_name = args.imcsv_filepath.split("/")[-1].split(".")[0]

    stat_df = formats.load_station_file(args.station_filepath)
    im_df = formats.load_im_file_pd(args.imcsv_filepath)
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

    non_uniform_filepath = os.path.join(
        args.output_path, "non_uniform_im.xyz"
    )
    real_station_filepath = os.path.join(
        args.output_path, "real_station_im.xyz"
    )

    xyz_df[columns].to_csv(non_uniform_filepath, sep=" ", header=None, index=None)
    xyz_real_station_df[columns].to_csv(
        real_station_filepath, sep=" ", header=None, index=None
    )

    im_col_file = os.path.join(args.output_path, "im_order.txt")
    with open(im_col_file, "w") as fp:
        fp.write(" ".join(ims))

    print("xyz files are output to {}".format(args.output_path))
