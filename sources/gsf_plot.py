"""GSF Plotting Module.

This module provides functionality to parse GSF files and plot the seismic points on a map using PyGMT.

Usage:
  python gsf_plot.py plot_gsf_file GSF_FILE_PATH PLOT_FILE_PATH [--grid-resolution=GRID_RESOLUTION] [--plot-dpi=PLOT_DPI]

Arguments:
  GSF_FILE_PATH          The path to the GSF file to be plotted.
  PLOT_FILE_PATH         The path where the plot image will be saved.

Options:
  --grid-resolution=GRID_RESOLUTION   The resolution of the grid (in metres). This can be different to
                                      to the resolution of the GSF file. Interpolation will occur
                                      between grid points.
  --plot-dpi=PLOT_DPI                 The output plot DPI (higher is better). [default: 1200]
"""

import re
from pathlib import Path
from typing import Annotated, TextIO

import pandas as pd
import pygmt
import typer
from pygmt_helper import plotting

app = typer.Typer()


def parse_gsf(gsf_file_handle: TextIO) -> pd.DataFrame:
    """parse_gsf.

    Parameters
    ----------
    gsf_file_handle : TextIO
        gsf_file_handle

    Returns
    -------
    pd.DataFrame

    """
    while gsf_file_handle.readline()[0] == "#":
        pass
    # NOTE: This skips one line past the last line beginning with #.
    # This is ok as this line is always the number of points in the GSF file, which we do not need.
    points = []
    for line in gsf_file_handle:
        (
            lon,
            lat,
            depth,
            sub_dx,
            sub_dy,
            strike,
            dip,
            rake,
            slip,
            init_time,
            seg_no,
        ) = re.split(r"\s+", line.strip())
        points.append(
            [
                float(lon),
                float(lat),
                -float(depth),
                float(sub_dx),
                float(sub_dy),
                float(strike),
                float(dip),
                float(rake),
                float(slip),
                float(init_time),
                int(seg_no),
            ]
        )
    return pd.DataFrame(
        columns=[
            "lon",
            "lat",
            "depth",
            "sub_dx",
            "sub_dy",
            "strike",
            "dip",
            "rake",
            "slip",
            "init_time",
            "seg_no",
        ],
        data=points,
    )


def plot_gsf_points(points: pd.DataFrame, grid_resolution: int) -> pygmt.Figure:
    """Plot a set of GSF points (lat, lon, depth) in a map using PyGMT.

    Parameters
    ----------
    points : pd.DataFrame
        The GSF points to plot. DataFrame must at least contain the lat, lon and depth columns.
    grid_resolution : int
        The resolution of the grid. If this is smaller than the resolution
        of the grid points in the GSF file, interpolation will occur between
        gridpoints.

    Returns
    -------
    pygmt.Figure
        The figure containing the plot.
    """
    region = [
        points["lon"].min() - 1,
        points["lon"].max() + 1,
        points["lat"].min() - 1,
        points["lat"].max() + 1,
    ]
    fig = plotting.gen_region_fig("GSF File", region=region, map_data=None)
    point_grid = plotting.create_grid(
        points,
        "depth",
        grid_spacing=f"{grid_resolution}e/{grid_resolution}e",
        region=region,
        set_water_to_nan=False,
    )
    plotting.plot_grid(
        fig,
        point_grid,
        "hot",
        (0, points["depth"].max(), 1),
        ("white", "black"),
        plot_contours=False,
    )

    return fig


@app.command(help="Plot a GSF file using GMT.")
def plot_gsf_file(
    gsf_filepath: Annotated[
        Path, typer.Argument(help="GSF file path to read.", exists=True, readable=True)
    ],
    figure_plot_path: Annotated[
        Path, typer.Argument(help="The file path to output the plot to.")
    ],
    grid_resolution: Annotated[
        int,
        typer.Option(
            help="The resolution to plot the grid points at (in metres)."
            " This can be different to the resolution of the GSF file."
            " Interpolation will occur between grid points.",
            min=5,
        ),
    ] = 100,
    plot_dpi: Annotated[
        int, typer.Option(help="The output plot DPI (higher is better).")
    ] = 1200,
):
    """Plot a GSF file using GMT.

    Parameters
    ----------
    gsf_filepath : Path
        The file path of the GSF file.
    figure_plot_path : Path
        The file path to output the plot to
    grid_resolution : int
        The resolution of the grid. This can be different to the resolution
        of the GSF file. Interpolation will occur between grid points.
    plot_dpi : int
        The output plot DPI (higher for better quality plot output).
    """
    with open(gsf_filepath, "r", encoding='utf-8') as gsf_file_handle:
        points = parse_gsf(gsf_file_handle)

    fig = plot_gsf_points(points, grid_resolution)
    fig.savefig(figure_plot_path, anti_alias=True, dpi=plot_dpi)


if __name__ == "__main__":
    app()
