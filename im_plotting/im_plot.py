"""
Assumption: (1) im_values.csv and im_values.info are in the same location and
            (2) .csv and .info have the same prefix
            
Generate non_uniform.xyz and sim/obs.xyz file

Command:
To generate .xyz:
python im_plot.py ~/kelly_sim_ims/kelly_sim_ims.info /home/nesi00213/dev/impp_datasets/Darfield/sample_nz_grid.ll -o ~/xyz_test
python im_plot.py ~/kelly_sim_ims/kelly_sim_ims.info /home/nesi00213/dev/impp_datasets/darfield_benchmark/rrups.csv -o ~/xyz_test

To plot:
python plot_stations.py ~/xyz_test/nonuniform_im_plot_map_kelly_sim_ims.xyz --out_dir ~/xyz_test --model_params /home/nesi00213/VelocityModel/v1.64_FVM/model_params_nz01-h0.100
"""

import os
import sys
import argparse
import getpass
import common

SIM_HOT = "hot-karim:invert 0.2 80/0/0 0/0/80"
NON_UNIFORM_HOT = "hot-karim:invert,t-30,overlays-blue 1k:g-surface,nns-12m,contours"
SIM_TEMPLATE = 'sim_im_plot_map_{}.xyz'
SIM_OBS_TEMPLATE = 'obs_im_plot_map_{}.xyz'
NON_UNI_EMP_TEMPLATE = 'nonuniform_im_plot_map_{}_empirical.xyz'
NON_UNI_TEMPLATE = 'nonuniform_im_plot_map_{}.xyz'
COMPS = ['geom', '090', '000', 'ver']


def get_runname(meta_filepath):
    """
    get the run name for output xyz filename from the .info metadata file
    :param meta_filepath: user input
    :return: run_name, run_type
    """
    with open(meta_filepath, 'r') as meta_file:
        meta_file.readline()  # skip header
        info = meta_file.readline().strip().split(',')
        run_name = info[0]
        run_type = info[2]
    return run_name, run_type


def get_data(meta_filepath):
    """
    Assumes that .info and .csv are in the same location
    :param meta_filepath:
    :return: lines from summary data csv file
    """
    csv_path = meta_filepath.replace('.info', '.csv')
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
    get measures for xyz file and mmi index
    :param data_header: row1 of summary csv
    :param is_non_uniform:
    :return: measures_header, mmi_index
    """
    measures = data_header.strip().split(',')[2:]
    i = 0
    mmi_index = None
    while i < len(measures):
        if 'pSA' in measures[i]:
            measures[i] = measures[i].replace('_', ' (') + 's)'
        if is_non_uniform:
            if measures[i] == 'MMI':
                measures.pop(i)
                mmi_index = i
                i -= 1
                continue
        i += 1
    measures_header = ', '.join(measures)
    return measures_header, mmi_index


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
        print("{} does not exits in the rrup or station file that you provided".format(station_name))


def get_hot_header(is_non_uniform):
    """
    check if a non_uniform plot and return corresponding hot header
    :param is_non_uniform: Boolean
    :return: hot header
    """
    return NON_UNIFORM_HOT if is_non_uniform else SIM_HOT


def check_virtual(station_name, is_non_uniform):
    """
    check if a virtual station
    :param station_name:
    :param is_non_uniform:
    :return: Boolean
    """
    return common.is_virtual_station(station_name) if not is_non_uniform else False


def get_im_values(im_values_list, mmi_index):
    """
    get mmi excluded or included im values
    :param im_values_list:
    :param mmi_index:
    :return: im values
    """
    if mmi_index:  # if we removed mmi from the original measures as generating non_uniform plot, we need to exclude the corresponding value in csv
        im_values_list.pop(mmi_index)
    values = ' '.join(im_values_list)
    return values


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
    third_line = get_hot_header(is_non_uniform)

    with open(output_path, 'w') as fp:
        fp.write("IM Plot\n")
        fp.write("IM\n")
        fp.write("{}\n".format(third_line))
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
                if coords:
                    is_virtual = check_virtual(station_name, is_non_uniform)
                    if not is_virtual:
                        fp.write('{} {}\n'.format(coords, values))


def generate_im_plot_map(run_name, run_type, data, coords_dict, output_dir, comp, is_non_uniform=False):
    """
    writes im_plot .xyz file
    :param run_name: row2_colum1 string from .info metadata file
    :param run_type: row2_colum3 stirng from .info metadata file
    :param data: summary csv buffer
    :param coords_dict: summary csv data buffer
    :param output_dir: user input
    :param comp: user input
    :param is_non_uniform: Bool
    :return:
    """
    if run_type == 'simulated':
        filename = SIM_TEMPLATE.format(run_name)
    else:
        filename = SIM_OBS_TEMPLATE.format(run_name)

    write_lines(output_dir, filename, data, coords_dict, comp, is_non_uniform=is_non_uniform)


def generate_non_uniform_plot_map(run_name, run_type, data, coords_dict, output_dir, comp, is_non_uniform=True):
    """
    writes non_uniform .xyz file
    :param run_name:
    :param run_type:
    :param data:
    :param coords_dict:
    :param output_dir:
    :param comp:
    :param is_non_uniform:
    :return:
    """
    if run_type == 'empirical':
        filename = NON_UNI_EMP_TEMPLATE.format(run_name)
    else:
        filename = NON_UNI_TEMPLATE.format(run_name)

    write_lines(output_dir, filename, data, coords_dict, comp, is_non_uniform=is_non_uniform)


def validate_filepath(parser, file_path):
    """
    validates input file path
    :param parser:
    :param file_path:
    :return: parser error if error
    """
    try:
        with open(file_path, 'r') as f:
            return
    except IOError:
        parser.error("{} is a dir".format(file_path))
    except OSError:
        parser.error("{} does not exit".format(file_path))


def validate_dir(parser, dir_path):
    if not os.path.isdir(dir_path):
        parser.error('{} is not a directory'.format(dir_path))


def validate_compoent(parser, comp):
    """
    validates input component
    :param parser:
    :param comp:
    :return: parser error if error
    """
    if comp.lower() not in COMPS:
        parser.error("Please enter a valid component. Choose from {}".format(COMPS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_filepath', help='path to input metadata file')
    parser.add_argument('rrup_or_station_filepath', help='path to inpurt rrup_csv/station_ll file path')
    parser.add_argument('-o', '--output_path', default='/home/{}'.format(getpass.getuser()), help='path to store output xyz files')
    parser.add_argument('-c', '--component', default='geom', help="single component of the vel/acc vector. Available compoents are {}. Default is 'geom'".format(COMPS))
    args = parser.parse_args()

    validate_filepath(parser, args.meta_filepath)
    validate_filepath(parser, args.rrup_or_station_filepath)
    validate_dir(parser, args.output_path)

    run_name, run_type = get_runname(args. meta_filepath)
    data = get_data(args.meta_filepath)
    coords_dict = get_coords_dict(args.rrup_or_station_filepath)

    generate_im_plot_map(run_name, run_type, data, coords_dict, args.output_path, args.component)
    generate_non_uniform_plot_map(run_name, run_type, data, coords_dict, args.output_path, args.component)

    print("xyz files generated to {}".format(args.output_path))


if __name__ == '__main__':
    main()