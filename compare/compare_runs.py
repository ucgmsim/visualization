import argparse
import numpy as np
from os import makedirs
from os.path import expandvars
from pathlib import Path

from subprocess import Popen, run

from time import sleep

from visualization import plot_items_wrapper
from im import im_ratios, spatialise_im


from qcore.formats import load_rrup_file


from functools import partial


def load_args():
    parser = argparse.ArgumentParser(
        "Orchestration script to compare two realisations. Assumes you have a common git directory root.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "base_im_csv", type=Path, nargs=2, help="path to IM file and label"
    )
    parser.add_argument(
        "comp_im_csv",
        type=Path,
        nargs=2,
        help="path to IM file and label, will be compared to base_im_csv",
    )

    parser.add_argument("rrup", type=Path)

    parser.add_argument("--git_base", type=Path, default="$gmsim")
    parser.add_argument("--out_dir", type=Path, default="comp_out")
    parser.add_argument(
        "-n",
        "--n_processes",
        help="Only used for --diff_ani and --plot_items",
        type=int,
        default=1,
    )
    parser.add_argument("--comp", "--component", type=str, default=None, dest="comp")

    parser.add_argument("--im_rrup", action="store_true", default=False)
    parser.add_argument("--psa_bias", action="store_true", default=False)
    parser.add_argument("--psa_ratios_rrup", action="store_true", default=False)
    parser.add_argument("--plot_items", action="store_true", default=False)
    parser.add_argument("--diff_ani", action="store_true", default=False)
    parser.add_argument("--all", action="store_true", default=False)

    parser.add_argument(
        "--im_rrup_bars",
        help="For im_rrup to plot mean values with error bars. ignored if --im_rrup is not set",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--im_rrup_srf_info",
        help="For im_rrup to generate generic empiricals. ignored if --im_rrup is not set",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--plot_items_stat_file",
        help="For plot_items to run, provide a station file (.ll). ignored if --plot_items is not set",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--plot_items_srf",
        help="For plot_items to run, provide a srf file (.srf). ignored if --plot_items is not set",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--diff_ani_base_xyts",
        help="For diff_ani to run, provide a base .xyts file. ignored if --diff_ani is not set",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--diff_ani_comp_xyts",
        help="For diff_ani to run, provide a comp .xyts file. ignored if --diff_ani is not set",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    makedirs(args.out_dir, exist_ok=True)

    return args


def im_rrup(
    git_base: Path,
    rrup_path: Path,
    srf_info: Path,
    component: str,
    bars: bool,
    out_dir: Path,
    base_im_csv: Path,
    base_label: str,
    comp_im_csv: Path = None,
    comp_label: str = None,
):

    im_rrup_path = (git_base / "visualization" / "im" / "im_rrup_mean.py").resolve()

    cmd_args = ["python", im_rrup_path, rrup_path, "--imcsv", base_im_csv, base_label]
    if comp_im_csv is not None:
        cmd_args += ["--imcsv", comp_im_csv, comp_label]
        title = f"Comparison of {base_label} with {comp_label}"
    else:
        title = base_label

    cmd_args += [
        "--config",
        (
            git_base / "Empirical_Engine" / "empirical" / "util" / "model_config.yaml"
        ).resolve(),
        "--srf",
        srf_info,
        "--run_name",
        title,
        "--out_dir",
        (out_dir / "im_rrup").resolve(),
    ]
    max_dist = load_rrup_file(rrup_path)["r_rup"].max()
    if max_dist > 100:
        cmd_args += ["--dist_max", str(2 * max_dist)]

    if component is not None:
        cmd_args += ["--comp", component]
    if bars:
        cmd_args += ["--bars"]
    if srf_info is not None:
        cmd_args += ["--srf", srf_info]

    return cmd_args


def psa_ratios_rrup(
    git_base: Path,
    rrup_path: Path,
    component: str,
    out_dir: Path,
    base_im_csv: Path,
    base_label: str,
    comp_im_csv: Path,
    comp_label: str,
):

    psa_ratios_rrup_path = (
        git_base / "visualization" / "im" / "psa_ratios_rrup.py"
    ).resolve()

    cmd_args = [
        "python",
        psa_ratios_rrup_path,
        rrup_path,
        "--imcsv",
        base_im_csv,
        base_label,
        "--imcsv",
        comp_im_csv,
        comp_label,
        "--run_name",
        f"Comparison of {base_label} with {comp_label}",
        "--out_dir",
        (out_dir / "psa_ratios_rrup").resolve(),
    ]

    if component is not None:
        cmd_args += ["--comp", component]
    return cmd_args


def psa_bias(
    git_base: Path,
    component: str,
    out_dir: Path,
    base_im_csv: Path,
    base_label: str,
    comp_im_csv: Path,
    comp_label: str,
):

    psa_bias_path = (git_base / "visualization" / "im" / "psa_bias.py").resolve()
    cmd_args = [
        "python",
        psa_bias_path,
        "--imcsv",
        base_im_csv,
        base_label,
        "--imcsv",
        comp_im_csv,
        comp_label,
        "--run_name",
        f"Comparison of {base_label} with {comp_label}",
        "--out_dir",
        (out_dir / "psa_bias").resolve(),
    ]
    if component is not None:
        cmd_args += ["--comp", component]
    return cmd_args


def plot_items(
    git_base: Path,
    ims: list,
    min_vals: dict,
    max_vals: dict,
    title: str,
    srf_path: Path,
    out_dir: Path,
    n_procs: int = 1,
):
    """
    Parameters:
    ims: a list of IM names
    min_vals, max_vals: a dictionary of {im_name: value....}
    """

    plot_items_path = (
        git_base / "visualization" / "sources" / "plot_items.py"
    ).resolve()

    cmd_args = []
    for i, im in enumerate(ims):
        if "sigma" in im:
            continue
        lower = min_vals[im]
        upper = max_vals[im]
        upper = max(abs(upper), abs(lower))
        ndigits = 10 ** np.floor(np.log10(upper))
        upper = np.ceil(upper / ndigits) * ndigits
        lower = -upper

        options_dict = {
            "flags": ["xyz-grid", "xyz-landmask", "xyz-grid-contours"],
            "options": {
                "nproc": str(n_procs),
                "srf-only-outline": srf_path,
                "title": title,
                "xyz-cpt-labels": im,
                "xyz-grid-search": "12m",
                "xyz-cpt": "polar",
                "xyz-transparency": "30",
                "xyz-size": "1k",
                "xyz-cpt-inc": str(round((upper - lower) / 10, 2)),
                "xyz-cpt-tick": str(round((upper - lower) / 5, 2)),
                "xyz-cpt-min": str(lower),
                "xyz-cpt-max": str(upper),
            },
        }
        plot_items = partial(
            plot_items_wrapper.plot,
            plot_items_path,
            str(out_dir / "non_uniform_im.xyz"),
            options_dict,
            str(out_dir),
            column_idx=[i],
            sep=" ",
            header_exists=False,
            out_f=f"{im}_ratio",
        )

        cmd_args.append(plot_items)
    return cmd_args


def diff_ani(
    git_base: Path, base_xyts: Path, comp_xyts: Path, out_dir: Path, n_procs: int = 1
):
    cmd_args = [
        git_base / "visualization" / "animation" / "plot_ts.py",
        base_xyts,
        "--xyts2",
        comp_xyts,
        "--output",
        (out_dir / "diff_ani").resolve(),
        "--nproc",
        str(n_procs),
    ]
    return cmd_args


def run_all(parallel_jobs, serial_jobs, func_jobs, n_procs=1):
    # Run the currently assembled tasks in parallel
    running_tasks = []
    while len(parallel_jobs) + len(running_tasks) > 0:
        for n, t in running_tasks:
            rc = t.poll()
            if rc is not None:
                running_tasks.remove((n, t))
        while len(running_tasks) < n_procs and len(parallel_jobs) > 0:
            n, t = parallel_jobs.pop()
            t1 = [str(x) for x in t]
            print(t1)
            running_tasks.append((n, Popen(t1)))
        sleep(10)  # check

    for n, task in serial_jobs:
        task1 = [str(x) for x in task]
        print(n, task1)
        run(task)

    for task in func_jobs:
        task()


def main():
    args = load_args()

    parallel_tasks_to_run = []
    serial_tasks_to_run = []
    funcs_to_run = []

    git_base = Path(expandvars(args.git_base))
    out_dir = args.out_dir

    # IM vs rrup
    rrup_path = args.rrup
    base_im_csv, base_label = args.base_im_csv
    comp_im_csv, comp_label = args.comp_im_csv

    ratios_csv = (out_dir / "im_ratios.csv").resolve()
    ims, (min_vals, max_vals) = im_ratios.ratios_to_csv(
        base_im_csv, comp_im_csv, ratios_csv, args.comp, summary=True
    )

    if args.im_rrup or args.all:
        serial_tasks_to_run.append(
            (
                "im rrup plots",
                im_rrup(
                    git_base,
                    rrup_path,
                    args.im_rrup_srf_info,
                    args.comp,
                    args.im_rrup_bars,
                    out_dir,
                    base_im_csv,
                    base_label,
                    comp_im_csv,
                    comp_label,
                ),
            )
        )
        serial_tasks_to_run.append(
            (
                "im ratio rrup plots",
                im_rrup(
                    git_base,
                    rrup_path,
                    args.im_rrup_srf_info,
                    args.comp,
                    args.im_rrup_bars,
                    out_dir,
                    (out_dir / "im_ratios.csv").resolve(),
                    "ratio",
                ),
            )
        )

    if args.psa_bias or args.all:
        serial_tasks_to_run.append(
            (
                "psa bias plots",
                psa_bias(
                    git_base,
                    args.comp,
                    out_dir,
                    base_im_csv,
                    base_label,
                    comp_im_csv,
                    comp_label,
                ),
            )
        )

    if args.psa_ratios_rrup or args.all:
        serial_tasks_to_run.append(
            (
                "psa ratios rrup plots",
                psa_ratios_rrup(
                    git_base,
                    rrup_path,
                    args.comp,
                    out_dir,
                    base_im_csv,
                    base_label,
                    comp_im_csv,
                    comp_label,
                ),
            )
        )

    if args.plot_items or args.all:

        out_dir_plot_items = out_dir / "plot_items"
        stat_file = args.plot_items_stat_file.resolve()
        # Generate spatial ratio XYZ, generate spatial plots from XYZ
        spatialise_im.write_xyz(ratios_csv, stat_file, out_dir_plot_items)

        funcs_to_run += plot_items(
            git_base,
            ims,
            min_vals,
            max_vals,
            f'"{base_label}_vs._{comp_label}"',
            args.plot_items_srf,
            out_dir_plot_items,
            args.n_processes,
        )

    # create difference animation
    # Should run by itself as it can multi process
    # Run command to plot difference animation if possible
    if args.diff_ani or args.all:
        parallel_tasks_to_run.append(
            (
                "diff_ani plots",
                diff_ani(
                    git_base,
                    args.diff_ani_base_xyts,
                    args.diff_ani_comp_xyts,
                    out_dir,
                    args.n_processes,
                ),
            )
        )

    # run plotting now
    run_all(
        parallel_tasks_to_run,
        serial_tasks_to_run,
        funcs_to_run,
        n_procs=args.n_processes,
    )


if __name__ == "__main__":
    main()
