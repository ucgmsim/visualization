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

import functools
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
from matplotlib.pyplot import Figure
from pygmt_helper import plotting
from qcore import coordinates, gsf

app = typer.Typer()


def plot_gsf_segment(fig: Figure, segment_points: pd.DataFrame):
    """Plot a segment of a fault in a given figure.

    Parameters
    ----------
    fig : Figure
        The figure to plot in.
    segment_points : pd.DataFrame
        The points of the segment to plot.
    """
    min_depth = segment_points["depth"].min()
    max_depth = segment_points["depth"].max()
    top_edge = segment_points.loc[segment_points["depth"] == min_depth]
    bottom_edge = segment_points.loc[segment_points["depth"] == max_depth]
    top_edge = coordinates.wgs_depth_to_nztm(
        top_edge[["lat", "lon", "depth"]].to_numpy()
    )
    bottom_edge = coordinates.wgs_depth_to_nztm(
        bottom_edge[["lat", "lon", "depth"]].to_numpy()
    )
    # luckily GSF points are in order of strike, so the first and
    # last elements of the top and bottom edges *must* be the corners of the segment.
    top_left = coordinates.nztm_to_wgs_depth(top_edge[0])
    top_right = coordinates.nztm_to_wgs_depth(top_edge[-1])
    bottom_left = coordinates.nztm_to_wgs_depth(bottom_edge[0])
    bottom_right = coordinates.nztm_to_wgs_depth(bottom_edge[-1])
    corners = np.vstack([top_left, top_right, bottom_left, bottom_right])
    ymin, ymax = corners[:, 0].min(), corners[:, 0].max()
    xmin, xmax = corners[:, 1].min(), corners[:, 1].max()

    cur_grid = plotting.create_grid(
        segment_points,
        "depth",
        grid_spacing="50e/50e",
        region=(xmin, xmax, ymin, ymax),
        set_water_to_nan=False,
    )

    plotting.plot_grid(
        fig,
        cur_grid,
        "hot",
        (min_depth, max_depth, (max_depth - min_depth) / 20),
        ("white", "black"),
        cb_label="Depth (km)",
        transparency=0,
        reverse_cmap=True,
    )
    # plot left, right, and bottom edges with dashed lines.
    fig.plot(
        x=[top_left[1], bottom_left[1]],
        y=[top_left[0], bottom_left[0]],
        pen="0.5p,black,-",
    )
    fig.plot(
        x=[top_right[1], bottom_right[1]],
        y=[top_right[0], bottom_right[0]],
        pen="0.5p,black,-",
    )
    fig.plot(
        x=[bottom_left[1], bottom_right[1]],
        y=[bottom_left[0], bottom_right[0]],
        pen="0.5p,black,-",
    )
    # plot the fault trace in bold.
    fig.plot(
        x=[top_left[1], top_right[1]],
        y=[top_left[0], top_right[0]],
        pen="0.5p,black",
    )


@app.command(help="Plot a GSF file using GMT.")
def plot_gsf_file(
    gsf_filepath: Annotated[
        Path, typer.Argument(help="GSF file path to read.", exists=True, readable=True)
    ],
    figure_plot_path: Annotated[
        Path, typer.Argument(help="The file path to output the plot to.", writable=True)
    ],
    plot_title: Annotated[
        Optional[str],
        typer.Option(
            help="The output plot tite. If not specified, this is just the name of the GSF file."
        ),
    ] = None,
    plot_dpi: Annotated[
        int, typer.Option(help="The output plot DPI (higher is better).", min=0)
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
    points = gsf.read_gsf(gsf_filepath)
    region = [
        points["lon"].min() - 0.5,
        points["lon"].max() + 0.5,
        points["lat"].min() - 0.5,
        points["lat"].max() + 0.5,
    ]
    fig = plotting.gen_region_fig(
        plot_title or gsf_filepath.stem, region=region, map_data=None
    )
    points.groupby("seg_no").apply(
        functools.partial(plot_gsf_segment, fig), include_groups=False
    )
    fig.savefig(figure_plot_path, anti_alias=True, dpi=plot_dpi)


if __name__ == "__main__":
    app()
