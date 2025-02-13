"""Utility script to plot cumulative moment over time for an SRF."""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
from matplotlib import pyplot as plt

from source_modelling import moment, rupture_propagation, srf

app = typer.Typer()


@app.command(help="Plot cumulative moment for an SRF over time.")
def plot_srf_cumulative_moment(
    srf_ffp: Annotated[
        Path,
        typer.Argument(
            help="SRF filepath to plot", exists=True, readable=True, dir_okay=False
        ),
    ],
    output_png_ffp: Annotated[
        Path, typer.Argument(help="Output plot path", writable=True, dir_okay=False)
    ],
    dpi: Annotated[
        int, typer.Option(help="Plot image pixel density (higher = better)", min=300)
    ] = 300,
    realisation_ffp: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to realisation, used to plot individual fault contribution."
        ),
    ] = None,
    min_shade_cutoff: Annotated[
        float, typer.Option(help="Minimum shading cutoff", min=0, max=1)
    ] = 0.05,
    max_shade_cutoff: Annotated[
        float, typer.Option(help="Maximum shading cutoff", min=0, max=1)
    ] = 0.95,
    height: Annotated[float, typer.Option(help="Plot height (cm)", min=0)] = 10,
    width: Annotated[float, typer.Option(help="Plot width (cm)", min=0)] = 10,
):
    """Plot cumulative moment for an SRF over time.

    Parameters
    ----------
    srf_ffp : Annotated[ Path, typer.Argument( help
        SRF filepath to plot.
    output_png_ffp : Annotated[ Path, typer.Argument(help
        Output plot path.
    dpi : Annotated[ int, typer.Option(help
        Plot image pixel density (higher = better).
    realisation_ffp : Annotated[ Optional[Path], typer.Option( help
        Path to realisation, used to plot individual fault
        contribution.
    min_shade_cutoff : Annotated[ float, typer.Option(help
        Minimum shading cutoff.
    max_shade_cutoff : Annotated[ float, typer.Option(help
        Maximum shading cutoff.
    height : float
        Height of plot (in cm).
    width : float
        Width of plot (in cm).
    """
    srf_data = srf.read_srf(srf_ffp)

    overall_moment = moment.moment_over_time_from_moment_rate(
        moment.moment_rate_over_time_from_slip(
            srf_data.points["area"],
            srf_data.slip,
            srf_data.dt,
            srf_data.nt,
        )
    )
    total_moment = overall_moment["moment"].iloc[-1]

    shaded_moments = overall_moment[
        (overall_moment["moment"] >= total_moment * min_shade_cutoff)
        & (overall_moment["moment"] <= total_moment * max_shade_cutoff)
    ]
    fig, ax = plt.subplots()
    cm = 1 / 2.54
    fig.set_size_inches(width * cm, height * cm)
    ax.fill_between(shaded_moments.index.values, shaded_moments["moment"], alpha=0.2)
    ax.plot(
        overall_moment.index.values, overall_moment["moment"], label="Overall Moment"
    )

    if realisation_ffp:  # pragma: no cover
        # NOTE: this import is here because the workflow is, as yet,
        # not ready to be installed along-side source modelling.
        from workflow.realisations import RupturePropagationConfig, SourceConfig

        source_config = SourceConfig.read_from_realisation(realisation_ffp)
        rupture_propogation_config = RupturePropagationConfig.read_from_realisation(
            realisation_ffp
        )
        segment_counter = 0
        point_counter = 0
        for fault_name in rupture_propagation.tree_nodes_in_order(
            rupture_propogation_config.rupture_causality_tree
        ):
            plane_count = len(source_config.source_geometries[fault_name].planes)
            segments = srf_data.header.iloc[
                segment_counter : segment_counter + plane_count
            ]
            num_points = (segments["nstk"] * segments["ndip"]).sum()
            segment_points = srf_data.points.iloc[
                point_counter : point_counter + num_points
            ]

            individual_moment = moment.moment_over_time_from_moment_rate(
                moment.moment_rate_over_time_from_slip(
                    segment_points["area"],
                    srf_data.slip[point_counter : point_counter + num_points],
                    srf_data.dt,
                    srf_data.nt,
                )
            )

            ax.plot(
                individual_moment.index.values,
                individual_moment["moment"],
                label=fault_name,
            )
            total_moment = individual_moment["moment"].iloc[-1]
            shaded_moments = individual_moment[
                (individual_moment["moment"] >= total_moment * min_shade_cutoff)
                & (individual_moment["moment"] <= total_moment * max_shade_cutoff)
            ]
            ax.fill_between(
                shaded_moments.index.values, shaded_moments["moment"], alpha=0.2
            )
            segment_counter += plane_count
            point_counter += num_points

    ax.set_ylabel("Cumulative Moment (Nm)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    min_shade_percent = int(np.round(min_shade_cutoff * 100))
    max_shade_percent = int(np.round(max_shade_cutoff * 100))
    ax.set_title(
        f"Cumulative Moment over Time (Shaded Area: {min_shade_percent}% - {max_shade_percent}%)"
    )

    fig.savefig(output_png_ffp, dpi=dpi)


if __name__ == "__main__":
    app()
