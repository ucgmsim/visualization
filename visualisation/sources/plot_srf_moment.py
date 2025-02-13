"""Utility script to plot moment over time for an SRF."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from matplotlib import pyplot as plt

from source_modelling import moment, rupture_propagation, srf

app = typer.Typer()


@app.command(help="Plot released moment for an SRF over time.")
def plot_srf_moment(
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
    height: Annotated[float, typer.Option(help="Plot height (cm)", min=0)] = 10,
    width: Annotated[float, typer.Option(help="Plot width (cm)", min=0)] = 10,
) -> None:
    """Plot released moment for an SRF over time.

    Parameters
    ----------
    srf_ffp : Path
        SRF filepath to plot.
    output_png_ffp : Path
        Output plot path.
    dpi : float, default 300
        Plot image pixel density (higher = better).
    realisation_ffp : Optional[Path], default None
        Path to realisation, used to plot individual fault contribution.
    height : float
        Height of plot (in cm).
    width : float
        Width of plot (in cm).
    """
    srf_data = srf.read_srf(srf_ffp)

    magnitude = moment.moment_to_magnitude(
        moment.MU * (srf_data.points["area"] * srf_data.points["slip"] / (100**3)).sum()
    )

    dt = srf_data.points["dt"].iloc[0]

    overall_moment_rate = moment.moment_rate_over_time_from_slip(
        srf_data.points["area"], srf_data.slip, dt, srf_data.nt
    )
    fig, ax = plt.subplots()
    cm = 1 / 2.54
    fig.set_size_inches(width * cm, height * cm)
    ax.plot(
        overall_moment_rate.index.values,
        overall_moment_rate["moment_rate"],
        label="Overall Moment Rate",
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
            individual_moment_rate = moment.moment_rate_over_time_from_slip(
                srf_data.points["area"]
                .iloc[point_counter : point_counter + num_points]
                .to_numpy(),
                srf_data.slip[point_counter : point_counter + num_points],
                dt,
                srf_data.nt,
            )
            ax.plot(
                individual_moment_rate.index.values,
                individual_moment_rate["moment_rate"],
                label=fault_name,
            )
            segment_counter += plane_count
            point_counter += num_points

    ax.set_ylabel("Moment Rate (Nm/s)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_title(f"Moment over Time (Total Mw: {magnitude:.2f})")

    fig.savefig(output_png_ffp, dpi=dpi)


if __name__ == "__main__":
    app()
