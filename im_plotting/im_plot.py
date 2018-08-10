"""
Assumption: (1) im_values.csv and im_values_[imcalc|empirical].info are in the same location and
            (2) .csv and _[imcalc|empirical].info have the same prefix

Generate non_uniform.xyz and sim/obs.xyz file

Command:
To generate .xyz:
python im_plot.py ~/darfield_sim/darfield_sim.csv /nesi/projects/nesi00213/dev/impp_datasets/Darfield/non_uniform_whole_nz_with_real_stations-hh100_17062017.ll -o ~/test_emp_plot
python im_plot.py ~/darfield_sim/darfield_sim.csv ~/rrup.csv -o ~/test_emp_plot

To plot:
python plot_stations.py ~/test_emp_plot/sim_im_plot_map_darfield_sim.xyz --srf_cnrs /nesi/projects/nesi00213/dev/impp_datasets/Darfield/bev01_s103246Allsegm_v8_23.srf --model_params /nesi/projects/nesi00213/dev/impp_datasets/Darfield/model_params_nz01-h0.100 --out_dir ~/test_emp_plot/sim_im_plot_map
"""

import os
import sys
import argparse
import getpass
import glob

from qcore import shared
from qcore import utils

SIM_HOT = "hot-karim:invert 0.2 80/0/0 0/0/80"
NON_UNIFORM_HOT = "hot-karim:invert,t-30,overlays-blue 1k:g-surface,nns-12m,contours"
SIM_TEMPLATE = 'sim_im_plot_map_{}.xyz'
OBS_TEMPLATE = 'obs_im_plot_map_{}.xyz'
EMP_TEMPLATE = 'emp_im_plot_map_{}.xyz'
NON_UNI_EMP_TEMPLATE = 'nonuniform_im_plot_map_{}_empirical.xyz'
NON_UNI_SIM_TEMPLATE = 'nonuniform_im_plot_map_{}_sim.xyz'
NON_UNI_OBS_TEMPLATE = 'nonuniform_im_plot_map_{}_obs.xyz'
TEMPLATE_DICT = {'simulated': (SIM_TEMPLATE, NON_UNI_SIM_TEMPLATE),
                 'observed': (OBS_TEMPLATE, NON_UNI_OBS_TEMPLATE),
                 'empirical': (EMP_TEMPLATE, NON_UNI_EMP_TEMPLATE)}
COMPS = ['geom', '090', '000', 'ver']
DEFAULT_OUTPUT_DIR = '/home/{}/im_plot_map_xyz'.format(getpass.getuser())


def check_get_meta(csv_filepath):
    """
    :param csv_filepath: user input path to summary im/emp .csv file
    :return: runname, meta_info_file path
    """
    csv_filename = csv_filepath.split('/')[-1]
    csv_dir = os.path.abspath(os.path.dirname(csv_filepath))
    runname = csv_filename.split('.')[0]
    meta_filename = glob.glob1(csv_dir, '{}_*'.format(runname))
    if len(meta_filename) == 1:
        return runname, os.path.join(csv_dir, meta_filename[0])
    else:
        sys.exit("Please provide a meta info file for the csv you have provided")


def get_runtype(meta_filepath):
    """
    get the run type for output xyz filename from the '_[imcalc|empirical].info' metadata file
    :param meta_filepath: user input
    :return: run_type
    """
    with open(meta_filepath, 'r') as meta_file:
        meta_file.readline()  # skip header
        run_type = meta_file.readline().strip().split(',')[2]
    return run_type


def get_data(csv_path):
    """
    Assumes that .info and .csv are in the same location
    :param csv_path:
    :return: lines from summary data csv file
    """
    try:
        with open(csv_path, 'r') as csv_file:
            buf = csv_file.readlines()
        return buf
    except IOError:
        sys.exit("check if you have permission to read {}".format(csv_path))
    except OSError:
        sys.exit("{} does not exit".format(csv_path))


def get_measures_header(data_header, is_non_uniform):
    """
    :param data_header:
    :param is_non_uniform:
    :return:
    """
    measures = data_header.strip().split(',')[2:]
    i = 0
    keep_indexes = []
    new_measures = []
    while i < len(measures):
        new_measure = measures[i]
        if 'sigma' in new_measure:
            i += 1
            continue
        
        if 'pSA' in new_measure:
            new_measure = new_measure.replace('_', ' (') + 's)'

        if is_non_uniform:
            if new_measure == 'MMI':
                i += 1
                continue
        new_measures.append(new_measure)
        keep_indexes.append(i)
        i += 1
    new_measures_header = ', '.join(new_measures)
    return new_measures_header, keep_indexes


def get_coords_dict(file_path):
    """
    reads a rrup or .ll file, return a coords dict
    :param file_path: path to rrup.csv or staiton.ll
    :return: dict {station_name: (lon, lat)}
    """
    coords_dict = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        try:
            for line in lines:
                if '.ll' in file_path:
                    lon, lat, station_name = line.strip().split()
                else:
                    station_name, lon, lat, _, _, _ = line.strip().split(',')
                coords_dict[station_name] = (lon, lat)

        except ValueError:
            sys.exit("Check column numbers in {}".format(file_path))

    return coords_dict


def get_coords(station_name, coords_dict):
    """
    get coords from coords dict
    :param station_name:
    :param coords_dict: {station_name: (lon, lat)}
    :return: coords string
    """
    try:
        lon, lat = coords_dict[station_name]
        return '{} {}'.format(lon, lat)
    except KeyError:
        print("{} does not exist in the rrup or station file that you provided".format(station_name))
        return None


def get_im_values(im_values_list, keep_indexes):
    """
    get mmi excluded or included im values
    :param im_values_list:
    :param keep_indexes: indexes of im_values to keep
    :return: im values
    """
    new_values = []
    for index in keep_indexes:
        new_values.append(im_values_list[index])
    new_values = ' '.join(new_values)
    return new_values


def write_lines(output_dir, filename, data, coords_dict, component, is_non_uniform):
    """
    write xyz file content
    :param output_dir: user input
    :param filename: xyz file name
    :param data: summary csv data buffer
    :param coords_dict: {station_name: (lon, lat)}
    :param component: string
    :param is_non_uniform: Boolean
    :return:
    """
    output_path = os.path.join(output_dir, filename)
    print("output path {}".format(output_path))

    if is_non_uniform:
        hot_type = NON_UNIFORM_HOT
    else:
        hot_type = SIM_HOT

    with open(output_path, 'w') as fp:
        fp.write("IM Plot\n")
        fp.write("IM\n")
        fp.write("{}\n".format(hot_type))
        fp.write("\n")

        measures, mmi_index = get_measures_header(data[0], is_non_uniform)

        total_measures = len(measures.split(','))  # number of measures contains either mmi or not

        fp.write('{} black\n'.format(total_measures))
        fp.write('{}\n'.format(measures))

        for line in data[1:]:
            segs = line.strip().split(',')
            comp = segs[1]
            if comp == component:  # only get data with specified comp
                station_name = segs[0]
                im_values = segs[2:]
                values = get_im_values(im_values, mmi_index)
                coords = get_coords(station_name, coords_dict)
                if coords is not None:
                    if not is_non_uniform:  # uniform.xyz only writes non_virtual station
                        if not shared.is_virtual_station(station_name):
                            fp.write('{} {}\n'.format(coords, values))
                    else:  # non_uniform.xyz writes all stations regardless virtual or not
                        fp.write('{} {}\n'.format(coords, values))


def generate_im_plot_map(run_name, run_type, data, coords_dict, output_dir, comp, is_non_uniform):
    """
    writes im_plot .xyz file
    :param run_name: row2_colum1 string from .info metadata file
    :param run_type: row2_colum3 stirng from .info metadata file
    :param data: summary csv buffer
    :param coords_dict: summary csv data buffer
    :param output_dir: user input
    :param comp: user input
    :param is_non_uniform: int repr of Bool: 0(F)/1(T)
    :return:
    """
    filename = TEMPLATE_DICT[run_type][is_non_uniform].format(run_name)

    write_lines(output_dir, filename, data, coords_dict, comp, is_non_uniform)


def validate_filepath(parser, file_path):
    """
    validates input file path
    :param parser:
    :param file_path: user input
    :return: parser error if error
    """
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as f:
                return
        except (IOError, OSError):
            parser.error("Can't open {}".format(file_path))
    else:
        parser.error("No such file {}".format(file_path))  # might be a dir or not exist


def validate_dir(parser, dir_path):
    """
    validates a dir
    :param parser:
    :param dir_path: user input
    :return:
    """
    if not os.path.isdir(dir_path):
        parser.error('No such directory {}'.format(dir_path))


def validate_component(parser, comp):
    """
    validates input component
    :param parser:
    :param comp: user input single vel/acc component
    :return: parser error if error
    """
    if comp.lower() not in COMPS:
        parser.error("Please enter a valid component. Choose from {}".format(COMPS))


def generate_maps():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_filepath', help='path to input metadata file')
    parser.add_argument('rrup_or_station_filepath', help='path to inpurt rrup_csv/station_ll file path')
    parser.add_argument('-o', '--output_path', default=DEFAULT_OUTPUT_DIR, help='path to store output xyz files')
    parser.add_argument('-c', '--component', default='geom', help="which component of the intensity measure. Available compoents are {}. Default is 'geom'".format(COMPS))
    args = parser.parse_args()

    utils.setup_dir(args.output_path)

    validate_filepath(parser, args.csv_filepath)
    validate_filepath(parser, args.rrup_or_station_filepath)
    validate_dir(parser, args.output_path)
    validate_component(parser, args.component)

    run_name, meta_filepath = check_get_meta(args.csv_filepath)
    run_type = get_runtype(meta_filepath)
    data = get_data(args.csv_filepath)
    coords_dict = get_coords_dict(args.rrup_or_station_filepath)

    generate_im_plot_map(run_name, run_type, data, coords_dict, args.output_path, args.component, 0)
    generate_im_plot_map(run_name, run_type, data, coords_dict, args.output_path, args.component, 1)

    print("xyz files are output to {}".format(args.output_path))


if __name__ == '__main__':
    generate_maps()

