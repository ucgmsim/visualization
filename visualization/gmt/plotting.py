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
from typing import Dict, Iterable

import yaml
import pandas as pd

from qcore.utils import change_file_ext

PLOT_CMD_TEMPLATE = "{} {} --xyz {} -f {}"


def plot_multiple(
    plot_items_ffp: str,
    gen_options_dict: Dict,
    file_pattern: str = None,
    in_ffps: Iterable[str] = None,
    recursive: bool = False,
    n_procs: int = 4,
    no_clobber: bool = True,
):
    """
    Parameters
    ----------
    plot_items_ffp: str
        File path to the plot_items.py script
    gen_options_dict: dictionary
        Contains the general plotting options,
        will be overwritten by the individual
        plotting options
    file_pattern: str, optional
        Glob file pattern to use to select the
        csv files to plot.
        Either the file_pattern or the in_ffps
        argument has to be specified.
    in_ffps: str, optional
        File path to the input csv files
        Required columns are lon, lat & value
    recursive: bool, optional
        If specified then the glob search is recursive,
        no effect if in_ffps is specifed.
    n_procs: int, optional
        Number of processes to uses
    no_clobber: bool, optional
        If set then nothing is done if the
        plot file already exists
    """
    if file_pattern is None and in_ffps is None:
        raise ValueError(
            "Either the file_pattern or the " "in_ffps argument has to be given."
        )

    files = (
        in_ffps if in_ffps is not None else glob.glob(file_pattern, recursive=recursive)
    )

    # Create a temporary dir for the .xyz files
    with tempfile.TemporaryDirectory() as tmp_dir:
        if n_procs == 1:
            results = []
            for cur_file in files:
                results.append(
                    plot_single(
                        plot_items_ffp,
                        cur_file,
                        gen_options_dict,
                        tmp_dir,
                        no_clobber=no_clobber,
                    )
                )
        else:
            with mp.Pool(n_procs) as p:
                results = p.starmap(
                    plot_single,
                    [
                        (
                            plot_items_ffp,
                            cur_file,
                            gen_options_dict,
                            tmp_dir,
                            no_clobber,
                        )
                        for cur_file in files
                    ],
                )


def plot_single(
    plot_items_ffp: str,
    in_ffp: str,
    gen_options_dict: Dict,
    tmp_dir: str,
    no_clobber: bool = True,
):
    """Generates a single spatial plot using the plot_items.py script and
    the specified csv data

    Also checks for an plot options config file and uses
    those options if it exists.
    Format: in_ffp_without_extension.yaml,

    Parameters
    ----------
    plot_items_ffp: str
        File path to the plot_items.py script
    in_ffp: str
        File path to the input csv
        Required columns are lon, lat & value
    gen_options_dict: dictionary
        Contains the general plotting options,
        will be overwritten by the individual
        plotting options
    tmp_dir: str
        Temporary directory used by the
        plot_items.py script
    no_clobber: bool
        If set then nothing is done if the
        plot file already exists
    """
    out_ffp = change_file_ext(in_ffp, "png")
    if no_clobber and os.path.exists(out_ffp):
        print(f"{os.path.basename(out_ffp)} already exists, skipping.")
        return None

    # Check if there is a individual options dict
    ind_options_ffp, ind_options_dict = change_file_ext(in_ffp, "yaml"), None
    if os.path.exists(ind_options_ffp):
        with open(ind_options_ffp, "r") as f:
            ind_options_dict = yaml.safe_load(f)

    options_dict = gen_options_dict
    if ind_options_dict is not None:
        options_dict = {
            "flags": gen_options_dict["flags"] + ind_options_dict["flags"],
            "options": {**gen_options_dict["options"], **ind_options_dict["options"]},
        }

    return plot(plot_items_ffp, in_ffp, options_dict, tmp_dir)


def plot(plot_items_ffp: str, in_ffp: str, options_dict: Dict, tmp_dir: str):
    """Runs the plotting for the given csv file and
    options.

    Does not perform any cleaning of the tmp directory,
    has to be done by the calling function.

    Parameters
    ----------
    plot_items_ffp: str
        File path to the plot_items.py script
    in_ffp: str
        File path to the input csv
        Required columns are lon, lat & value
    options_dict: dictionary
        Options dictionary of the format:
        {"flags": ["flag_1", "flag_2"],
        "options": {"option_1": "value_1",
                    "option_2": "value_2"}
    tmp_dir: str
        Temporary directory used by the
        plot_items.py script

    Returns
    -------
    CompletedProcess:
        The completed process instance from
        the subprocess.run call
    """
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
