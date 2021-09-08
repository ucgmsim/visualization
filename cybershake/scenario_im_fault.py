import os
import argparse
from pathlib import Path
import pandas as pd


faults = ["HopeConwayOS", "Port2GreyL"]
# faults = ["HopeConwayOS", "Wairau", "AlpineK2T", "Port2GreyL"]
srfs = {
    "HopeConwayOS": "/isilon/cybershake/v20p4/Sources/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/jpa198/cybershake_sources/Data/Sources/HopeConwayOS/Srf/HopeConwayOS_REL01.srf",
    "Wairau": "/isilon/cybershake/v20p4/Sources/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/jpa198/cybershake_sources/Data/Sources/Wairau/Srf/Wairau_REL01.srf",
    "AlpineK2T": "/isilon/cybershake/v20p4/Sources/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/jpa198/cybershake_sources/Data/Sources/AlpineK2T/Srf/AlpineK2T_REL01.srf",
    "Port2GreyL": "/isilon/cybershake/v20p4/Sources/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/jpa198/cybershake_sources/Data/Sources/Port2GreyL/Srf/Port2GreyL_REL01.srf",
}
ims = [
    "PGV",
    "PGV_sigma",
    "PGA",
    "PGA_sigma",
    "pSA_0.1",
    "pSA_0.1_sigma",
    "pSA_1.0",
    "pSA_1.0_sigma",
    "pSA_5.0",
    "pSA_5.0_sigma",
]
max_ranges = {
    "PGV": 80,
    "PGV_sigma": 0.8,
    "PGA": 0.8,
    "PGA_sigma": 0.8,
    "pSA_0.1": 1.5,
    "pSA_0.1_sigma": 0.8,
    "pSA_1.0": 0.8,
    "pSA_1.0_sigma": 0.8,
    "pSA_5.0": 0.2,
    "pSA_5.0_sigma": 0.8,
}


def main(visualization_ffp: Path, scenario_data_ffp: Path, output_dir: str, model: str):
    for fault in faults:
        file_faults = list(scenario_data_ffp.glob(f"{model}/*{fault}*.csv"))
        for file in file_faults:
            df = pd.read_csv(file)
            for im in ims:
                # Creates the Fault_IM file
                im_df = df[["station", "component", im]]
                fault_im_dir = Path(output_dir) / file.parent.name
                fault_im_filename = fault_im_dir / f"{fault}_{im}.csv"
                fault_im_dir.mkdir(exist_ok=True, parents=True)
                pd.DataFrame.to_csv(im_df, fault_im_filename, index=False)

                # Directory prep for xyz
                xyz_output_dir = Path(output_dir) / file.parent.name / "xyz" / im
                xyz_output_dir.mkdir(exist_ok=True, parents=True)

                # Creates the xyz files
                spatialise_im_ffp = Path(visualization_ffp) / "im/spatialise_im.py"
                os.system(
                    f"{spatialise_im_ffp} {fault_im_filename} /isilon/seistech/sites/20p3/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll -o {xyz_output_dir}"
                )

                # Plotting setup
                cpt_max = max_ranges[im]
                cpt_inc = cpt_max / 10
                cpt_tick = cpt_max / 5
                plot_options = f"--xyz-grid --xyz-grid-type nearneighbor --xyz-grid-search 10k --xyz-landmask --xyz-cpt hot --xyz-grid-contours --xyz-transparency 30 --xyz-size 1k --xyz-cpt-inc {cpt_inc} --xyz-cpt-tick {cpt_tick} --xyz-cpt-min 0 --xyz-cpt-max {cpt_max} --xyz-cpt-invert"
                non_uniform_im = xyz_output_dir / "non_uniform_im.xyz"
                plot_output_filename = f"{fault}_{im}_{file.parent.name}"

                print(f"Plotting {plot_output_filename}")
                # Plotting xyz file
                plot_items_ffp = Path(visualization_ffp) / "sources/plot_items.py"
                os.system(
                    f"/{plot_items_ffp} {plot_options} --xyz {non_uniform_im} -f {plot_output_filename} --xyz-cpt-labels {plot_output_filename} -c '{srfs[fault]}' --outline-fault-colour black "
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-visualization_ffp",
        type=str,
        help="Full file path to the visualization repo",
        required=True,
    )
    parser.add_argument(
        "-scenario_data_ffp",
        type=str,
        help="Full file path to the scenario data directory",
        required=True,
    )
    parser.add_argument(
        "-output_dir",
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
    main(Path(args.visualization_ffp), Path(args.scenario_data_ffp), args.output_dir, args.model)
