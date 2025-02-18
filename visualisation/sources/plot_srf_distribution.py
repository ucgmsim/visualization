#!/usr/bin/env python3
"""Plot SRF distributions."""
from pathlib import Path
from typing import Annotated, Optional

import typer
from matplotlib import pyplot as plt
from source_modelling import srf

from visualisation import utils

app = typer.Typer()


@app.command()
@utils.from_docstring
def plot_srf_distribution(
    srf_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    plot_png: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    dpi: Annotated[int, typer.Option(min=300)] = 300,
    height: Annotated[float, typer.Option(min=0)] = 10,
    width: Annotated[float, typer.Option(min=0)] = 10,
    title: Annotated[Optional[str], typer.Option()] = None,
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
    title : str, optional
        Title for the plot, by default None.
    """
    srf_data = srf.read_srf(srf_ffp)
    fig, ax = plt.subplots(figsize=(width, height))

    ax.hist(srf_data.points["slip"], density=True)
    ax.set_xlabel("Slip (cm)")
    ax.set_title(
        title
        or f'Slip PDF for {srf_ffp.stem} ({utils.format_description(srf_data.points["slip"])})'
    )

    plt.savefig(plot_png, dpi=dpi)
