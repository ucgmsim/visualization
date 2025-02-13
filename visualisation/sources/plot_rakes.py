"""Plot a sample of rake values across a multi-segment rupture."""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from pygmt_helper import plotting
from source_modelling import srf

app = typer.Typer()


@app.command(help="Plot a sample of rake values across a multi-segment rupture.")
def plot_rakes(
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF file to plot.", exists=True)
    ],
    output_ffp: Annotated[
        Path, typer.Argument(help="Output plot image.", dir_okay=False)
    ],
    dpi: Annotated[
        float, typer.Option(help="Plot output DPI (higher is better).")
    ] = 300,
    title: Annotated[Optional[str], typer.Option(help="Plot title to use.")] = None,
    sample_size: Annotated[
        int, typer.Option(help="Number of points to sample for rake.")
    ] = 200,
    vector_length: Annotated[
        float, typer.Option(help="Length of rake vectors (cm).")
    ] = 0.2,
    seed: Annotated[
        Optional[int], typer.Option(help="Random seed to sample rakes with")
    ] = None,
    width: Annotated[float, typer.Option(help="Plot width (cm)", min=0)] = 17,
) -> None:
    """Plot an SRF file and output a PNG file.

    Parameters
    ----------
    srf_ffp : Path
        Path to the SRF file.
    output_ffp : Path
        Path of the output plot image.
    dpi : float, default 300
        Plot output DPI (higher is better).
    title : Optional[str], default None
        Plot title to use.
    sample_size : int, default 200
        Number of points to sample for rake.
    vector_length : float, default 0.2cm
        Length of rake vectors (cm).
    seed : int
        The random seed to sample rakes with.
    width : float
        Width of plot (in cm).
    """
    srf_data = srf.read_srf(srf_ffp)
    region = (
        srf_data.points["lon"].min() - 0.5,
        srf_data.points["lon"].max() + 0.5,
        srf_data.points["lat"].min() - 0.25,
        srf_data.points["lat"].max() + 0.25,
    )

    fig = plotting.gen_region_fig(
        title, projection=f"M{width}c", region=region, map_data=None
    )
    i = 0

    np.random.seed(seed)
    vectors = srf_data.points[["lon", "lat", "rake"]].sample(sample_size)
    vectors["rake"] = (vectors["rake"] + 90) % 360
    vectors["length"] = vector_length

    fig.plot(
        data=vectors.values.tolist(), style="v0.1c+e+a30", pen="0.2p", fill="black"
    )
    for _, segment in srf_data.header.iterrows():
        nstk = int(segment["nstk"])
        ndip = int(segment["ndip"])
        point_count = nstk * ndip
        segment_points = srf_data.points.iloc[i : i + point_count]
        corners = segment_points.iloc[[0, nstk - 1, -1, (ndip - 1) * nstk]]
        fig.plot(
            x=corners["lon"].iloc[list(range(len(corners))) + [0]].to_list(),
            y=corners["lat"].iloc[list(range(len(corners))) + [0]].to_list(),
            pen="0.5p,black,-",
        )

        i += point_count

    fig.savefig(
        output_ffp,
        dpi=dpi,
        anti_alias=True,
    )


if __name__ == "__main__":
    app()
