import os
import errno
import shutil
import getpass
import pytest
import shutil
from qcore import shared, utils

TEST_FOLDER = os.path.abspath(os.path.dirname(__file__))
SCRIPT = os.path.abspath(os.path.join(TEST_FOLDER, '..', '..', 'im_plotting', 'im_plot.py'))

INPUT_DIR = os.path.join(TEST_FOLDER,'sample1','input')
INPUT_OBS = os.path.join(INPUT_DIR, 'input_obs.info')
INPUT_SIM = os.path.join(INPUT_DIR, 'input_sim.info')
INPUT_EMP = os.path.join(INPUT_DIR, 'input_emp.info')
INPUT_LL = os.path.join(INPUT_DIR, 'station.ll')
INPUT_RRUP = os.path.join(INPUT_DIR, 'rrup.csv')

OUTPUT_DIR = os.path.join(TEST_FOLDER, 'sample1', 'output')
OUTPUT_OBS_XYZ = os.path.join(OUTPUT_DIR, 'obs_im_plot_map_kelly.xyz')
OUTPUT_SIM_XYZ = os.path.join(OUTPUT_DIR, 'sim_im_plot_map_kelly.xyz')
OUTPUT_NONUNI_XYZ = os.path.join(OUTPUT_DIR, 'nonuniform_im_plot_map_kelly.xyz')
OUTPUT_NONUNI_EMP_XYZ = os.path.join(OUTPUT_DIR, 'nonuniform_im_plot_map_kelly_empirical.xyz')

BENCHMARK_OBS_XYZ = os.path.join(INPUT_DIR, 'obs_im_plot_map_kelly.xyz')
BENCHMARK_SIM_XYZ = os.path.join(INPUT_DIR, 'sim_im_plot_map_kelly.xyz')
BENCHMARK_NONUNI_XYZ = os.path.join(INPUT_DIR, 'nonuniform_im_plot_map_kelly.xyz')
BENCHMARK_NONUNI_EMP_XYZ = os.path.join(INPUT_DIR, 'nonuniform_im_plot_map_kelly_empirical.xyz')


def setup_module():
    """ create a tmp directory for storing output from test"""
    print "----------setup_module----------"
    utils.setup_dir(OUTPUT_DIR)


@pytest.fixture()
def garbage_collector():
    pytest.garbage = ''


@pytest.mark.parametrize("input_file, output_file1, output_file2, benchmark1, benchmark2", [(INPUT_OBS, OUTPUT_OBS_XYZ, OUTPUT_NONUNI_XYZ, BENCHMARK_OBS_XYZ, BENCHMARK_NONUNI_XYZ), (INPUT_SIM, OUTPUT_SIM_XYZ, OUTPUT_NONUNI_XYZ, BENCHMARK_SIM_XYZ, BENCHMARK_NONUNI_XYZ), (INPUT_EMP, OUTPUT_OBS_XYZ, OUTPUT_NONUNI_EMP_XYZ, BENCHMARK_OBS_XYZ, BENCHMARK_NONUNI_EMP_XYZ), ])
def test_im_plot_ll(garbage_collector, input_file, output_file1, output_file2, benchmark1, benchmark2):
    cmd1 = 'python {} {} {} -o {}'.format(SCRIPT, input_file, INPUT_LL, OUTPUT_DIR)
    out1, err1 = shared.exe(cmd1)
    assert err1 == ''
    cmd2 = "diff {} {}"
    out2, err2 = shared.exe(cmd2.format(output_file1, benchmark1))
    assert out2 == "" and err2 == ""
    out3, err3 = shared.exe(cmd2.format(output_file2, benchmark2))
    assert out3 == "" and err3 == ""
    pytest.garbage += out2 + out3 + err1 + err2 + err3


@pytest.mark.parametrize("input_file, output_file1, output_file2, benchmark1, benchmark2", [(INPUT_OBS, OUTPUT_OBS_XYZ, OUTPUT_NONUNI_XYZ, BENCHMARK_OBS_XYZ, BENCHMARK_NONUNI_XYZ), (INPUT_SIM, OUTPUT_SIM_XYZ, OUTPUT_NONUNI_XYZ, BENCHMARK_SIM_XYZ, BENCHMARK_NONUNI_XYZ), (INPUT_EMP, OUTPUT_OBS_XYZ, OUTPUT_NONUNI_EMP_XYZ, BENCHMARK_OBS_XYZ, BENCHMARK_NONUNI_EMP_XYZ), ])
def test_im_plot_rrup(garbage_collector, input_file, output_file1, output_file2, benchmark1, benchmark2):
    cmd1 = 'python {} {} {} -o {}'.format(SCRIPT, input_file, INPUT_RRUP, OUTPUT_DIR)
    out1, err1 = shared.exe(cmd1)
    assert err1 == ''
    cmd2 = "diff {} {}"
    out2, err2 = shared.exe(cmd2.format(output_file1, benchmark1))
    assert out2 == "" and err2 == ""
    out3, err3 = shared.exe(cmd2.format(output_file2, benchmark2))
    assert out3 == "" and err3 == ""
    pytest.garbage += out2 + out3 + err1 + err2 + err3


def teardown_module(garbage_collector):
    if not pytest.garbage:
        try:
            shutil.rmtree(OUTPUT_DIR)
        except (IOError, OSError):
            raise
