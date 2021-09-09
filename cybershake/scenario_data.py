import os
import argparse
from pathlib import Path
from typing import Sequence

def main(summarise_im_ffp: str, im_csv_dir: str, faults: Sequence[str], ims: Sequence[str], output_dir: Path):
    for fault in faults:
        os.system(f"{summarise_im_ffp} {im_csv_dir} {fault} --im {' '.join(ims)} --output {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-summarise_im_ffp",
        type=str,
        help="Full file path to the summarise_all_rels_im script",
        required=True,
    )
    parser.add_argument(
        "-im_csv_dir",
        type=str,
        help="Full file path to the IM csv directory",
        required=True,
    )
    parser.add_argument(
        "-faults",
        type=str,
        nargs="+",
        help="Faults to extract from the IM CSV's",
        required=True,
    )
    parser.add_argument(
        "-ims",
        type=str,
        nargs="+",
        help="Ims to extract from the IM CSV's",
        required=True,
    )
    parser.add_argument("-output_dir", help="Output directory for the scenario fault files", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.summarise_im_ffp, args.im_csv_dir, args.faults, args.ims, args.output_dir)