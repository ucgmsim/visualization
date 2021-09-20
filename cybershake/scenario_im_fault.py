"""
Script for generating scenario im fault plots as a hazard map plot.

The script will plot each model and im on a hazard map plot and takes the scenario data im_csv data directly as inputs.
These im_csvs will then be broken down into single IM fault files and then into xyz files.
After the xyz files have been generated these files will be used to plot the scenario epsilon data in the
directory this script was run in.
"""
import subprocess
import argparse
from pathlib import Path

import yaml
import pandas as pd


def main(
    config_ffp: Path,
    scenario_data_ffp: Path,
    output_dir: Path,
    model: str,
):
    # Sets the visualization full file path to call other scripts
    visualization_ffp = Path(__file__).parent.parent

    # Load the config file
    with open(config_ffp, "r") as f:
        config = yaml.safe_load(f)

    for fault in config["faults"]:
        file_faults = list(scenario_data_ffp.glob(f"{model}/*{fault}*.csv"))
        for file in file_faults:
            df = pd.read_csv(file)
            for im in config["ims"]:
                # Creates the Fault_IM file
                im_df = df.loc[df['component'] == config["component"], ["station", "component", im]]
                fault_im_dir = output_dir / file.parent.name
                fault_im_filename = fault_im_dir / f"{fault}_{im}.csv"
                fault_im_dir.mkdir(exist_ok=True, parents=True)
                im_df.to_csv(fault_im_filename, index=False)

                # Directory prep for xyz
                xyz_output_dir = output_dir / file.parent.name / "xyz" / fault / im
                xyz_output_dir.mkdir(exist_ok=True, parents=True)

                # Creates the xyz files
                spatialise_im_ffp = visualization_ffp / "im" / "spatialise_im.py"
                subprocess.call(
                    [
                        str(spatialise_im_ffp),
                        str(fault_im_filename),
                        str(config["station_file"]),
                        "-o",
                        str(xyz_output_dir),
                    ]
                )

                # Plotting setup
                cpt_max = float(config["max_ranges"][im])
                cpt_inc = cpt_max / 10
                cpt_tick = cpt_max / 5
                plot_options = [
                    "--xyz-grid",
                    "--xyz-grid-type",
                    "nearneighbor",
                    "--xyz-grid-search",
                    "10k",
                    "--xyz-landmask",
                    "--xyz-cpt",
                    "hot",
                    "--xyz-grid-contours",
                    "--xyz-transparency",
                    "30",
                    "--xyz-size",
                    "1k",
                    "--xyz-cpt-inc",
                    str(cpt_inc),
                    "--xyz-cpt-tick",
                    str(cpt_tick),
                    "--xyz-cpt-min",
                    "0",
                    "--xyz-cpt-max",
                    str(cpt_max),
                    "--xyz-cpt-invert",
                ]
                non_uniform_im = xyz_output_dir / "non_uniform_im.xyz"
                plot_output_filename = f"{fault}_{im}_{file.parent.name}"

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
                subprocess.call(plot_cmd)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_ffp",
        type=Path,
        help="Full file path to the scenario im fault config yaml",
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
        help="Output directory for the scenario fault files",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Only run for this given model, defaults to all models in scenario data directory",
        default="**",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.config_ffp,
        args.scenario_data_ffp,
        args.output_dir,
        args.model,
    )
