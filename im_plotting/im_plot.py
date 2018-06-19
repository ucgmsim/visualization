# TODO check what's common.is_virtual_station
import os
import sys
import itertools
import argparse
import getpass
import common


SIM_HOT = "hot:invert 0.2 80/0/0 0/0/80"
NON_UNIFORM_HOT = "hot-karim:invert,t-30,overlays-blue 1k:g-surface,nns-12m,contours"


def get_runname(meta_filepath):
    with open(meta_filepath, 'r') as meta_file:
        meta_file.readline()  # skip header
        info = meta_file.readline().strip().split(',')
        run_name = info[0]
        run_type = info[2]
    return run_name, run_type


def get_data(meta_filepath):
    csv_path = meta_filepath.replace('.info', '.csv')
    try:
        with open(csv_path, 'r') as csv_file:
            buf = csv_file.readlines()
        return buf
    except IOError:
        sys.exit("check if you have permission to read {}".format(csv_path))
    except OSError:
        sys.exit("{} does not exit".format(csv_path))


def get_measures_header(data_header, run_type):
    measures = data_header.strip().split(',')[2:]
    for i in range(len(measures)):
        if 'pSA' in measures[i]:
            measures[i] = measures[i].replace('_', ' (') + 's)'
        if run_type == 'observed':
            if measures[i] == 'MMI':
                measures.pop(i)
    measures_header = ', '.join(measures)
    return measures_header


def get_coords_dict(file_path):
    coords_dict = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '.ll' in file_path:
                    lon, lat, station_name = line.strip().split()
                else:
                    station_name, lon, lat, _, _, _ = line.strip().split(',')
                coords_dict[station_name] = (lon, lat)
        return coords_dict
    except ValueError:
        sys.exit("Check column numbers in {}".format(file_path))


def get_coords_string(station_name, coords_dict):
    try:
        lon, lat = coords_dict[station_name]
        return '{} {}'.format(lon, lat)
    except KeyError:
        print("{} does not exits in the rrup or station file that you provided".format(station_name))


def write_lines(output_dir, filename, data, run_type, coords_dict):
    output_path = os.path.join(output_dir, filename)
    print("output path {}".format(output_path))
    third_line = SIM_HOT if run_type == 'simulated' else NON_UNIFORM_HOT

    with open(output_path, 'w') as fp:
        fp.write("IM Plot\n")
        fp.write("IM\n")
        fp.write("{}\n".format(third_line))
        fp.write("\n")

        measures = get_measures_header(data[0], run_type)

        fp.write('{} black\n'.format(len(measures.split(','))))
        fp.write('{}\n'.format(measures))

        for line in data[1:]:
            segs = line.strip().split(',')
            station_name = segs[0]
            values = ' '.join(segs[2:])
            coords = get_coords_string(station_name, coords_dict)
            print(coords)
            fp.write('{} {}\n'.format(coords, values))


def generate_im_plot_map(run_name, run_type, data, coords_dict, output_dir):
    if run_type == 'simulated':
        filename = 'sim_im_plot_map_{}.xyz'.format(run_name)
    else:
        filename = 'obs_im_plot_map_{}.xyz'.format(run_name)

    write_lines(output_dir, filename, data, run_type, coords_dict)
            # if not common.is_virtual_station(datum1[5]):
            #     fp.write('%s %s %s %s\n' % (str(datum1[0]), str(datum1[1]), str(datum1[3].replace(',', ' ')), str(datum2[3].replace(',', ' '))))
            #


def generate_non_uniform_plot_map(run_name, run_type, data, coords_dict, output_dir):
    if run_type == 'empirical':
        filename = 'nonuniform_im_plot_map_{}_empirical.xyz'.format(run_name)
    else:
        filename = 'nonuniform_im_plot_map_{}.xyz'.format(run_name)

    write_lines(output_dir, filename, data, run_type, coords_dict)


def validate_filepath(parser, file_path):
    try:
        with open(file_path, 'r') as f:
            return
    except IOError:
        parser.error("check if you have permission to read {}".format(file_path))
    except OSError:
        parser.error("{} does not exit".format(file_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_filepath', help='path to input metadata file')
    parser.add_argument('rrup_or_station_filepath', help='path to inpurt rrup_csv/station_ll file path')
    parser.add_argument('-o', '--output_path', default='/home/{}'.format(getpass.getuser()), help='path to store output xyz files')
    args = parser.parse_args()

    validate_filepath(parser, args.meta_filepath)
    validate_filepath(parser, args.rrup_or_station_filepath)

    run_name, run_type = get_runname(args. meta_filepath)
    data = get_data(args.meta_filepath)
    coords_dict = get_coords_dict(args.rrup_or_station_filepath)

    generate_im_plot_map(run_name, run_type, data, coords_dict, args.output_path)
    generate_non_uniform_plot_map(run_name, run_type, data, coords_dict, args.output_path)

    print("xyz files generated to {}".format(args.output_path))


if __name__ == '__main__':
    main()