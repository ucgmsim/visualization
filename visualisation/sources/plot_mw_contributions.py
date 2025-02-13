"""Plot magnitude contributions of each segment in a rupture against the Leonard scaling relation."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from matplotlib import pyplot as plt

from source_modelling import moment, rupture_propagation, srf
from workflow.realisations import (
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
)

app = typer.Typer()


@app.command(help="Plot segment magnitudes against the Leonard scaling relation.")
def plot_mw_contributions(
    srf_ffp: Annotated[
        Path, typer.Argument(help="Path to SRF file", exists=True, dir_okay=False)
    ],
    realisation_ffp: Annotated[
        Path,
        typer.Argument(help="Realisation filepath", dir_okay=False, exists=True),
    ],
    output_ffp: Annotated[
        Path, typer.Argument(help="Output plot path.", writable=True, dir_okay=False)
    ],
    dpi: Annotated[
        float, typer.Option(help="Output plot DPI (higher is better).")
    ] = 300,
    height: Annotated[float, typer.Option(help="Plot height (cm)", min=0)] = 10,
    width: Annotated[float, typer.Option(help="Plot width (cm)", min=0)] = 10,
) -> None:
    """Plot segment magnitudes against the Leonard scaling relation.

    Parameters
    ----------
    srf_ffp : Path
        Path to SRF file.
    realisation_ffp : Path
        Realisation filepath.
    output_ffp : Path
        Output plot path.
    dpi : float, default 300
        Output plot DPI (higher is better).
    height : float
        Height of plot (in cm).
    width : float
        Width of plot (in cm).
    """
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    rupture_propogation_config = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    realisation_metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    total_area = sum(fault.area() for fault in source_config.source_geometries.values())
    smallest_area = min(
        fault.area() for fault in source_config.source_geometries.values()
    )
    area = np.linspace(smallest_area, total_area)
    fig, ax = plt.subplots()
    cm = 1 / 2.54
    fig.set_size_inches(width * cm, height * cm)
    # Mw = log(area) + 3.995 is the Leonard2014 magnitude scaling relation
    # for average rake.
    ax.plot(
        area, np.log10(area) + 3.995, label="Leonard 2014 Interplate (Average Rake)"
    )

    srf_data = srf.read_srf(srf_ffp)
    total_magnitude = moment.moment_to_magnitude(
        moment.MU * (srf_data.points["area"] * srf_data.points["slip"] / (100**3)).sum()
    )
    ax.scatter(total_area, total_magnitude, label="Total Magnitude")

    segment_counter = 0
    point_counter = 0
    for fault_name in rupture_propagation.tree_nodes_in_order(
        rupture_propogation_config.rupture_causality_tree
    ):
        plane_count = len(source_config.source_geometries[fault_name].planes)
        segments = srf_data.header.iloc[segment_counter : segment_counter + plane_count]

        num_points = (segments["nstk"] * segments["ndip"]).sum()
        individual_area = source_config.source_geometries[fault_name].area()

        # get all points associated with all segments in the current fault
        segment_points = srf_data.points.iloc[
            point_counter : point_counter + num_points
        ]
        individual_magnitude = moment.moment_to_magnitude(
            (
                moment.MU * segment_points["area"] * segment_points["slip"] / (100**3)
            ).sum()
        )
        ax.scatter(individual_area, individual_magnitude, label=fault_name)

        # advance segment counter and point counter to skip all points from the current point
        segment_counter += plane_count
        point_counter += num_points

    ax.set_xlabel("Area (m^2)")
    ax.set_ylabel("Mw")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title(f"Log Area vs Magnitude ({realisation_metadata.name})")
    fig.savefig(output_ffp, dpi=dpi)


if __name__ == "__main__":
    app()
