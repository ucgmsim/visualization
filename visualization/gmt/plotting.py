"""Functions for plotting single or multiple
csv files using the plot_items.py script

Supports a global options dictionary, or individual option yaml
files that have the same name as the input csv (except with extension .yaml)
"""

import os
import glob
import shutil
import tempfile
import subprocess
import multiprocessing as mp
from typing import Dict

import yaml
import pandas as pd

from qcore.utils import change_file_ext

PLOT_CMD_TEMPLATE = "{} {} --xyz {} -f {}"


def plot_multiple(
    plot_items_ffp: str,
    file_pattern: str,
    gen_options_dict: Dict,
    recursive: bool = False,
    n_procs: int = 4,
    no_clobber: bool = True,
):
    files = glob.glob(file_pattern, recursive=recursive)

    # Create a temporary dir for the .xyz files
    with tempfile.TemporaryDirectory() as tmp_dir:
        if n_procs == 1:
            for cur_file in files:
                if not no_clobber or not os.path.exists(
                    change_file_ext(cur_file, "png")
                ):
                    result = plot(plot_items_ffp, cur_file, gen_options_dict, tmp_dir)
        else:
            with mp.Pool(n_procs) as p:
                results = p.starmap(
                    plot,
                    [
                        (plot_items_ffp, cur_file, gen_options_dict, tmp_dir)
                        for cur_file in files
                        if not no_clobber
                        or not os.path.exists(change_file_ext(cur_file, "png"))
                    ],
                )


def plot_single(
    plot_items_ffp: str,
    in_ffp: str,
    gen_options_dict: Dict,
    tmp_dir: str,
    no_clobber: bool = True,
):
    if no_clobber and os.path.exists(in_ffp):
        print(f"{os.path.basename(in_ffp)} already exists, skipping.")
        return None

    # Check if there is a individual options dict
    ind_options_ffp, ind_options_dict = change_file_ext(in_ffp, "yaml"), None
    if os.path.exists(ind_options_ffp):
        with open(ind_options_ffp, "r") as f:
            ind_options_dict = yaml.safe_load(f)

    options_dict = (
        {**gen_options_dict, **ind_options_dict}
        if ind_options_dict is not None
        else gen_options_dict
    )

    return plot(plot_items_ffp, in_ffp, options_dict, tmp_dir)


def plot(plot_items_ffp: str, in_ffp: str, options_dict: Dict, tmp_dir: str):

    # Get the plotting flags
    plot_options = []
    for cur_flag in options_dict["flags"]:
        plot_options.append(f"--{cur_flag}")

    # Get the key-value plotting options
    for key, value in options_dict["options"].items():
        plot_options.append(f"--{key} {value}")
    plot_options = " ".join(plot_options)

    # Create temporary .xyz file for plotting
    # Cleaning up of the tmp dir has to be done by the calling function
    df = pd.read_csv(in_ffp)
    tmp_xyz_ffp = os.path.join(
        tmp_dir, change_file_ext(os.path.basename(in_ffp), "xyz")
    )
    df.to_csv(
        tmp_xyz_ffp, sep=" ", columns=["lon", "lat", "value"], header=False, index=False
    )

    out_f = os.path.basename(in_ffp).split(".")[0]
    cmd = PLOT_CMD_TEMPLATE.format(
        plot_items_ffp, plot_options, tmp_xyz_ffp, out_f
    ).split(" ")
    print(f"Plotting {os.path.basename(in_ffp)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Hack, since plot_items doesn't support saving to a full path
    if result.returncode == 0:
        cur_dir = os.getcwd()
        shutil.move(
            os.path.join(cur_dir, f"{out_f}_0.png"),
            os.path.join(os.path.dirname(in_ffp), f"{out_f}.png"),
        )

    print("----------------------------------------------")
    print(f"Cmd:\n{cmd}\n")
    print(f"stdout:\n{result.stdout.decode()}")
    print(f"stderr:\n{result.stderr.decode()}")
    print("----------------------------------------------")

    return result
