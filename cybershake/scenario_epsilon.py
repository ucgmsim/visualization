faults = ["HopeConwayOS", "Wairau", "AlpineK2T", "Port2GreyL"]
main_path = Path("/home/joel/local/scenarios")
output_path = Path("/home/joel/local/scenario_epsilon")

all_files = list(main_path.glob("**/*.csv"))
combo = []

for fault in faults:
    file_faults = list(main_path.glob(f"**/*{fault}*.csv"))
    for file in file_faults:
        file_faults.remove(file)
        for file_pair in file_faults:
            output_filename = output_path / f"{fault}_{file.parent.name}_{file_pair.parent.name}.csv"
            # os.system(f"/home/joel/code/slurm_gm_workflow/verification/calculate_epsilon.py {file} {file_pair} {output_filename}")

            sim_im_data = pd.read_csv(file, index_col=0)
            emp_im_data = pd.read_csv(file_pair, index_col=0)

            matched_ims = set(sim_im_data.columns.values).intersection(emp_im_data.columns.values)
            im_names = list(matched_ims)

            emp_im_data.columns = ["emp_" + IM for IM in emp_im_data.columns]
            merged_data = sim_im_data.merge(emp_im_data, left_index=True, right_index=True)

            print(im_names)
            epsilon = {}

            for im in im_names:
                if im == "component":
                    epsilon[im] = {}
                    for station in sim_im_data.index.values:
                        epsilon[im][station] = "geom"
                else:
                    emp_sigma = "emp_" + im
                    im_epsilon = im + "_epsilon"
                    emp_im = "emp_" + im
                    merged_data[im_epsilon] = (
                                                      np.log(merged_data[im].values) - np.log(merged_data[emp_im])
                                              ) / merged_data[emp_sigma]

            merged_data.sort_index(inplace=True)
            merged_data.to_csv(
                output_filename,
                columns=[
                    "component",
                    "PGA_epsilon",
                    "PGV_epsilon",
                    "pSA_0.1_epsilon",
                    "pSA_1.0_epsilon",
                    "pSA_5.0_epsilon",
                ],
            )