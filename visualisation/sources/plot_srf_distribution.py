#!/usr/bin/env python3
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
from matplotlib import pyplot as plt

from source_modelling import srf

app = typer.Typer()


def format_description(arr: np.ndarray, dp: float = 0) -> str:
    """Format a statistical description of an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    dp : float, optional
        Decimal places to round to, by default 0.

    Returns
    -------
    str
        Formatted string containing min, mean, max, and standard deviation.
    """
    min = arr.min()
    mean = np.mean(arr)
    max = arr.max()
    std = np.std(arr)
    return f"min = {min:.{dp}f} / μ = {mean:.{dp}f} / σ = {std:.{dp}f} / max = {max:.{dp}f}"


@app.command(help="Plot SRF slip distribution as a histogram.")
def plot_srf_distribution(
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF to plot.", exists=True, dir_okay=False)
    ],
    plot_png: Annotated[
        Path, typer.Argument(help="Path to output png.", dir_okay=False, writable=True)
    ],
    dpi: Annotated[
        int, typer.Option(help="Plot image pixel density (higher = better)", min=300)
    ] = 300,
    height: Annotated[float, typer.Option(help="Plot height (cm)", min=0)] = 10,
    width: Annotated[float, typer.Option(help="Plot width (cm)", min=0)] = 10,
    title: Annotated[Optional[str], typer.Option(help="Title for plot.")] = None,
) -> None:
    """Plot the slip distribution from an SRF file as a histogram.

    Parameters
    ----------
    srf_ffp : Path
        Path to the SRF file to plot.
    plot_png : Path
        Path to save the output PNG file.
    dpi : int, optional
        Pixel density of the output image (higher = better), by default 300.
    height : float, optional
        Height of the plot in cm, by default 10.
    width : float, optional
        Width of the plot in cm, by default 10.
    title : Optional[str], optional
        Title for the plot, by default None.
    """
    srf_data = srf.read_srf(srf_ffp)
    fig, ax = plt.subplots(figsize=(width, height))

    ax.hist(srf_data.points["slip"], density=True)
    ax.set_xlabel("Slip (cm)")
    ax.set_title(
        title
        or f'Slip PDF for {srf_ffp.stem} ({format_description(srf_data.points["slip"])})'
    )

    plt.savefig(plot_png, dpi=dpi)
