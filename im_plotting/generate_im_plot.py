import os
import glob
import subprocess
from qcore import utils

SLURM = 'im_plot.sl'
SIM_REPO = '/home/yzh231/kelly_srf'
LL_PATH = '/home/yzh231/non_uniform_whole_nz_with_real_stations-hh400_v18p6.ll'
PLOT_REPO = '/home/yzh231/IM_plot_kelly_srf'
MODEL_PARAMS = '/home/yzh231/model_params_rt01-h0.400'
SRF_1264 = '/home/nesi00213/RunFolder/Cybershake/v18p6/verification/Kelly/Kelly_HYP03-29_S1264.srf'


def plot_im(sim_repo=SIM_REPO, plot_repo=PLOT_REPO, model_params=MODEL_PARAMS):
    #with open(SLURM, 'w') as s:
        #s.write("#!/bin/bash\n#script version: slurm\n# Please modify this file as needed, this is just a sample\n#SBATCH --job-name=im_calc\n#SBATCH --account=nesi00213\n#SBATCH --partition=NeSI\n#SBATCH --ntasks=1\n#SBATCH --time={{time}}\n#SBATCH --output im_calc-%j.out\n#SBATCH --error im_calc-%j.err\n###SBATCH --mail-type=all\n###SBATCH --mail-user=test@test.com\n###SBATCH --mem-per-cpu=16G\n###SBATCH -C avx\n#SBATCH --hint=nomultithread\n## END HEADER\ndate")
    for dire in os.listdir(sim_repo):
        dire_path = os.path.join(sim_repo, dire)
        print(sim_repo, dire_path)
        info = glob.glob1(dire_path, '*.meta_info')[0]
        print("info", info)
        info_path = os.path.join(dire_path, info)
        print("info path", info_path)
        dire_plot_repo = os.path.join(plot_repo, dire)
        utils.setup_dir(dire_plot_repo)
        print("dir made", dire_plot_repo)
        cmd = 'python ~/visualization/im_plotting/im_plot.py {} {} -o {}'.format(info_path, LL_PATH, dire_plot_repo)
        print(cmd)
        subprocess.call(cmd, shell=True)
        xyz_files = glob.glob1(dire_plot_repo, '*.xyz')
        for xyz in xyz_files:
            xyz_path = os.path.join(dire_plot_repo, xyz)
            xyz_plot_dir = os.path.join(dire_plot_repo, xyz.split('.')[0])
            utils.setup_dir(xyz_plot_dir)
            cmd = 'python ~/visualization/gmt/plot_stations.py {} --out_dir {} --model_params {} --srf {}'.format(xyz_path, xyz_plot_dir, model_params, SRF_1264)
            subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    plot_im()
