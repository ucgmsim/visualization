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


def parse_gsf(gsf_filepath: str) -> pd.DataFrame:
    """Parse a GSF file into a pandas DataFrame.

    Parameters
    ----------
    gsf_file_handle : TextIO
        The file handle pointing to the GSF file to read.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all the points in the GSF file. The DataFrame's columns are
        - lon (longitude)
        - lat (latitude)
        - depth (Kilometres below ground, i.e. depth = -10 indicates a point 10km underground).
        - sub_dx (The subdivision size in the strike direction)
        - sub_dy (The subdivision size in the dip direction)
        - strike
        - dip
        - rake
        - slip (nearly always -1)
        - init_time (nearly always -1)
        - seg_no (the fault segment this point belongs to)
    """
    with open(gsf_filepath, mode="r", encoding="utf-8") as gsf_file_handle:
        # we could use pd.read_csv with the skiprows argument, but it's not
        # as versatile as simply skipping the first n rows with '#'
        while gsf_file_handle.readline()[0] == "#":
            pass
        # NOTE: This skips one line past the last line beginning with #.
        # This is ok as this line is always the number of points in the GSF
        # file, which we do not need.
        return pd.read_csv(
            gsf_file_handle,
            sep=r"\s+",
            names=[
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
        )


def plot_gsf_points(
    points: pd.DataFrame, grid_resolution: int, title: str
) -> pygmt.Figure:
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
    fig = plotting.gen_region_fig(title, region=region, map_data=None)
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
    plot_title: Annotated[
        str,
        typer.Option(
            help="The output plot tite. If not specified, this is just the name of the GSF file."
        ),
    ] = None,
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
    points = parse_gsf(gsf_filepath)
    # the plotting module functions will expect the depth parameter to be
    # positive, so we multiply by -1 to make it so.
    points["depth"] *= -1
    fig = plot_gsf_points(
        points, grid_resolution, title=plot_title or gsf_filepath.stem
    )
    fig.savefig(figure_plot_path, anti_alias=True, dpi=plot_dpi)


if __name__ == "__main__":
    app()
