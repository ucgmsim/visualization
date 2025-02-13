"""Plot multi-segment rupture with rise."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from pygmt_helper import plotting
from source_modelling import srf

app = typer.Typer()


@app.command(help="Plot multi-segment rupture with rise.")
def plot_rise(
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF file to plot.", exists=True)
    ],
    output_ffp: Annotated[
        Path, typer.Argument(help="Output plot image.", dir_okay=False)
    ],
    dpi: Annotated[
        float, typer.Option(help="Plot output DPI (higher is better)")
    ] = 300,
    title: Annotated[Optional[str], typer.Option(help="Plot title to use")] = None,
    width: Annotated[float, typer.Option(help="Plot width (cm)", min=0)] = 17,
) -> None:
    """Plot multi-segment drupture with rise.

    Parameters
    ----------
    srf_ffp : Path
        Path to SRF file to plot.
    output_ffp : Path
        Output plot image.
    dpi : float, default 300
        Plot output DPI (higher is better).
    title : Optional[str], default None
        Plot title to use.
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
    srf_slip = srf_data.slip

    srf_data.points["trise"] = (
        srf_slip.indptr[1:] - srf_slip.indptr[:-1]
    ) * srf_data.points["dt"]
    trise_cb_max = srf_data.points["trise"].max()
    cmap_limits = (0, trise_cb_max, trise_cb_max / 10)

    fig = plotting.gen_region_fig(
        title, projection=f"M{width}c", region=region, map_data=None
    )

    for i, segment_points in enumerate(srf_data.segments):
        cur_grid = plotting.create_grid(
            segment_points,
            "trise",
            grid_spacing="5e/5e",
            region=(
                segment_points["lon"].min(),
                segment_points["lon"].max(),
                segment_points["lat"].min(),
                segment_points["lat"].max(),
            ),
            set_water_to_nan=False,
        )
        plotting.plot_grid(
            fig,
            cur_grid,
            "hot",
            cmap_limits,
            ("white", "black"),
            transparency=0,
            reverse_cmap=True,
            plot_contours=False,
            cb_label="trise",
            continuous_cmap=True,
        )
        time_grid = plotting.create_grid(
            segment_points,
            "tinit",
            grid_spacing="5e/5e",
            region=(
                segment_points["lon"].min(),
                segment_points["lon"].max(),
                segment_points["lat"].min(),
                segment_points["lat"].max(),
            ),
            set_water_to_nan=False,
        )
        fig.grdcontour(
            levels=0.5,
            annotation=1,
            grid=time_grid,
            pen="0.1p",
        )
        nstk = srf_data.header["nstk"].iloc[i]
        ndip = srf_data.header["ndip"].iloc[i]
        corners = segment_points.iloc[[0, nstk - 1, -1, (ndip - 1) * nstk]]
        fig.plot(
            x=corners["lon"].iloc[list(range(len(corners))) + [0]].to_list(),
            y=corners["lat"].iloc[list(range(len(corners))) + [0]].to_list(),
            pen="0.5p,black,-",
        )

    fig.savefig(
        output_ffp,
        dpi=dpi,
        anti_alias=True,
    )


if __name__ == "__main__":
    app()
