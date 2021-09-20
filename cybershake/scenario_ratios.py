"""
Script for generating scenario ratio plots as IM and sigma ratios between different models.

The script will compare each model found in the scenario data directory and generate an im and sigma ratio im_csv.
These im_csvs will then be broken down into single IM fault files and then into xyz files.
After the xyz files have been generated these files will be used to plot the scenario epsilon data in the
directory this script was run in.
"""
import subprocess
import argparse
from pathlib import Path

import yaml
import pandas as pd


def main(config_ffp: Path, scenario_data_ffp: Path, output_dir: Path):
    # Sets the visualization full file path to call other scripts
    visualization_ffp = Path(__file__).parent.parent

    # Load the config file
    with open(config_ffp, "r") as f:
        config = yaml.safe_load(f)

    # Creating the ratio im_csvs
    for fault in config["faults"]:
        file_faults = list(scenario_data_ffp.glob(f"**/*{fault}*.csv"))
        for file in file_faults:
            file_faults.remove(file)
            for file_pair in file_faults:
                output_filename = (
                    output_dir
                    / f"{fault}_{file.parent.name}_{file_pair.parent.name}.csv"
                )
                im_ratios_ffp = visualization_ffp / "im" / "im_ratios.py"
                subprocess.Popen(
                    [
                        str(im_ratios_ffp),
                        str(file),
                        str(file_pair),
                        str(output_filename),
                    ]
                )

    # Splitting up the ratio im_csvs to plot
    for fault in config["faults"]:
        file_faults = list(output_dir.glob(f"*{fault}*.csv"))
        for file in file_faults:
            if "summary" not in str(file):
                df = pd.read_csv(file)
                for im in config["ims"]:
                    # Creates the Fault_IM file
                    im_df = df[["station", "component", im]]
                    model_comp = "_".join(str(file.stem).split("_")[1:])
                    fault_im_dir = file.parent / "fault_ims"
                    fault_im_dir.mkdir(exist_ok=True, parents=True)
                    fault_im_filename = fault_im_dir / f"{model_comp}_{fault}_{im}.csv"
                    im_df.to_csv(fault_im_filename, index=False)

                    # Directory prep for xyz
                    xyz_output_dir = file.parent / "xyz" / model_comp / im / fault
                    xyz_output_dir.mkdir(exist_ok=True, parents=True)

                    # Creates the xyz files
                    spatialise_im_ffp = visualization_ffp / "im" / "spatialise_im.py"
                    subprocess.Popen(
                        [
                            str(spatialise_im_ffp),
                            str(fault_im_filename),
                            str(config["station_file"]),
                            "-o",
                            str(xyz_output_dir),
                        ]
                    )

                    # Plotting setup
                    cpt_max = float(config["max_ranges"][im][1])
                    cpt_min = float(config["max_ranges"][im][0])
                    cpt_range = cpt_max + (cpt_min * -1)
                    cpt_inc = round(cpt_range / 11, 2)
                    cpt_tick = round(cpt_range / 5.5, 2)
                    plot_options = [
                        "--xyz-grid",
                        "--xyz-grid-type",
                        "nearneighbor",
                        "--xyz-grid-search",
                        "10k",
                        "--xyz-landmask",
                        "--xyz-cpt",
                        "polar",
                        "--xyz-grid-contours",
                        "--xyz-transparency",
                        "30",
                        "--xyz-cpt-bg",
                        "0/0/80",
                        "--xyz-cpt-fg",
                        "80/0/0",
                        "--xyz-size",
                        "1k",
                        "--xyz-cpt-inc",
                        str(cpt_inc),
                        "--xyz-cpt-tick",
                        str(cpt_tick),
                        "--xyz-cpt-min",
                        str(cpt_min),
                        "--xyz-cpt-max",
                        str(cpt_max),
                    ]
                    non_uniform_im = xyz_output_dir / "non_uniform_im.xyz"
                    plot_output_filename = f"{fault}_{im}_{model_comp}"

                    print(f"Plotting {plot_output_filename}")
                    # Plotting xyz file
                    plot_items_ffp = visualization_ffp / "sources" / "plot_items.py"
                    plot_cmd = [
                        str(plot_items_ffp),
                        "--xyz",
                        str(non_uniform_im),
                        "-f",
                        str(plot_output_filename),
                        "--xyz-cpt-labels",
                        str(plot_output_filename),
                        "-c",
                        str(config["srfs"][fault]),
                        "--outline-fault-colour",
                        "black",
                    ]
                    plot_cmd.extend(plot_options)
                    subprocess.Popen(plot_cmd)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_ffp",
        type=Path,
        help="Full file path to the scenario ratios config yaml",
        required=True,
    )
    parser.add_argument(
        "--scenario_data_ffp",
        type=Path,
        help="Full file path to the scenario data directory",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for the scenario ratio files",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.config_ffp,
        args.scenario_data_ffp,
        args.output_dir,
    )
