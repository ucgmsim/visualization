import os
import argparse
from pathlib import Path
from typing import Sequence
from pandas import pd


main_path = Path("/home/joel/local/scenario_ratios")
faults = ["HopeConwayOS", "Wairau", "AlpineK2T", "Port2GreyL"]
ims = ["PGV", "PGV_sigma", "PGA", "PGA_sigma", "pSA_0.1", "pSA_0.1_sigma", "pSA_1.0", "pSA_1.0_sigma", "pSA_5.0", "pSA_5.0_sigma"]
max_ranges = {
    "PGV": [-1, 1],
    "PGV_sigma": [-2, 2],
    "PGA": [-2, 2],
    "PGA_sigma": [-2, 2],
    "pSA_0.1": [-1, 1],
    "pSA_0.1_sigma": [-2, 2],
    "pSA_1.0": [-1.5, 1.5],
    "pSA_1.0_sigma": [-2, 2],
    "pSA_5.0": [-3, 3],
    "pSA_5.0_sigma": [-2, 2],
}
srfs = {
    "HopeConwayOS": "/isilon/cybershake/v20p4/Sources/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/jpa198/cybershake_sources/Data/Sources/HopeConwayOS/Srf/HopeConwayOS_REL01.srf",
    "Wairau": "/isilon/cybershake/v20p4/Sources/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/jpa198/cybershake_sources/Data/Sources/Wairau/Srf/Wairau_REL01.srf",
    "AlpineK2T": "/isilon/cybershake/v20p4/Sources/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/jpa198/cybershake_sources/Data/Sources/AlpineK2T/Srf/AlpineK2T_REL01.srf",
    "Port2GreyL": "/isilon/cybershake/v20p4/Sources/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/jpa198/cybershake_sources/Data/Sources/Port2GreyL/Srf/Port2GreyL_REL01.srf",
}


for fault in faults:
    file_faults = list(main_path.glob(f"*{fault}*.csv"))
    for file in file_faults:
        if "summary" not in str(file):
            df = pd.read_csv(file)
            for im in ims:
                # Creates the Fault_IM file
                im_df = df[["station", "component", im]]
                model_comp = "_".join(str(file.stem).split("_")[1:])
                fault_im_dir = file.parent / "fault_ims"
                fault_im_dir.mkdir(exist_ok=True, parents=True)
                fault_im_filename = fault_im_dir / f"{model_comp}_{fault}_{im}.csv"
                pd.DataFrame.to_csv(im_df, fault_im_filename, index=False)

                # Directory prep for xyz
                xyz_output_dir = file.parent / "xyz" / model_comp / im / fault
                xyz_output_dir.mkdir(exist_ok=True, parents=True)

                # Creates the xyz files
                os.system(
                    f"/home/joel/code/visualization/im/spatialise_im.py {fault_im_filename} /home/joel/local/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll -o {xyz_output_dir}"
                )

                # Plotting setup
                cpt_max = max_ranges[im][1]
                cpt_min = max_ranges[im][0]
                cpt_range = cpt_max + (cpt_min * -1)
                cpt_inc = round(cpt_range / 11, 2)
                cpt_tick = round(cpt_range / 5.5, 2)
                plot_options = f"--xyz-grid --xyz-grid-type nearneighbor --xyz-grid-search 10k --xyz-landmask --xyz-cpt polar --xyz-grid-contours --xyz-transparency 30 --xyz-cpt-bg 0/0/80 --xyz-cpt-fg 80/0/0 --xyz-size 1k --xyz-cpt-inc {cpt_inc} --xyz-cpt-tick {cpt_tick} --xyz-cpt-min {cpt_min} --xyz-cpt-max {cpt_max}"
                non_uniform_im = xyz_output_dir / "non_uniform_im.xyz"
                plot_output_filename = f"{fault}_{im}_{model_comp}"

                print(f"Plotting {plot_output_filename}")
                # Plotting xyz file
                os.system(
                    f"/home/joel/code/visualization/sources/plot_items.py {plot_options} --xyz {non_uniform_im} -f {plot_output_filename} --xyz-cpt-labels {plot_output_filename} -c '{srfs[fault]}' --outline-fault-colour black "
                )