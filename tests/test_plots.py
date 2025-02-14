from collections.abc import Callable
from pathlib import Path

import diffimg
import pytest

from visualisation.sources import (
    plot_rakes,
    plot_rise,
    plot_slip_rise_rake,
    plot_srf,
    plot_srf_cumulative_moment,
    plot_srf_distribution,
    plot_srf_moment,
)

PLOT_IMAGE_DIRECTORY = Path("wiki/images")
SRF_FFP = Path(__file__).parent / "srfs" / "rupture_1.srf"
MULTI_SUMMARY_SRF_FFP = Path(__file__).parent / "srfs" / "nevis.srf"
REALISATION_FFP = Path(__file__).parent / "srfs" / "realisation.json"


@pytest.mark.parametrize(
    "plot_function, expected_image_name",
    [
        (plot_srf.plot_srf, "srf_plot_example.png"),
        (plot_srf_moment.plot_srf_moment, "srf_moment_rate_example.png"),
        (
            plot_srf_cumulative_moment.plot_srf_cumulative_moment,
            "srf_cumulative_moment_rate_example.png",
        ),
        (plot_rise.plot_rise, "rise_example.png"),
        (plot_rakes.plot_rakes, "rakes_example.png"),
        (plot_srf_distribution.plot_srf_distribution, "srf_distribution_example.png"),
    ],
)
def test_plot_functions(
    tmp_path: Path, plot_function: Callable, expected_image_name: str
):
    """Check that the plotting scripts produce the wiki images within the expected tolerance."""
    output_image_path = tmp_path / "output.png"

    # plot-rakes expects a seed parameter that controls the distribution of rake vectors.
    # We set this seed to 1 to match the seed in the output image.
    if plot_function == plot_rakes.plot_rakes:
        plot_function(SRF_FFP, output_image_path, seed=1)
    else:
        plot_function(SRF_FFP, output_image_path)

    original = PLOT_IMAGE_DIRECTORY / expected_image_name
    generated = output_image_path

    diff = diffimg.diff(original, generated)
    assert diff <= 0.05


@pytest.mark.parametrize("plot_type", list(plot_slip_rise_rake.PlotType))
def test_plot_slip_rise_rake(tmp_path: Path, plot_type: plot_slip_rise_rake.PlotType):
    """Check that the slip-rise-rake plots work."""
    output_image_path = tmp_path / "output.png"
    original = PLOT_IMAGE_DIRECTORY / f"summary_{plot_type}.png"
    plot_slip_rise_rake.plot_slip_rise_rake(
        REALISATION_FFP,
        MULTI_SUMMARY_SRF_FFP,
        output_image_path,
        plot_type=plot_type,
        width=30,
        height=15,
    )

    diff = diffimg.diff(original, output_image_path)
    assert diff <= 0.05


def test_plot_slip_rise_rake_segment(tmp_path: Path):
    """Check that the slip-rise-rake plots work."""
    output_image_path = tmp_path / "output.png"
    original = PLOT_IMAGE_DIRECTORY / "summary_segment_1.png"
    plot_slip_rise_rake.plot_slip_rise_rake(
        REALISATION_FFP,
        MULTI_SUMMARY_SRF_FFP,
        output_image_path,
        segment=1,
        width=15,
        height=30,
    )

    diff = diffimg.diff(original, output_image_path)
    assert diff <= 0.05
