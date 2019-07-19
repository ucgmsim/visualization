"""
Install using pip, e.g. pip install ./visualization
use --no-deps to prevent re-installation of dependencies
use -I to force re-install
"""
from setuptools import setup


setup(
    name="visualization",
    version="1.0.0",
    packages=["visualization"],
    url="https://github.com/ucgmsim/visualization",
    description="visualization code",
    install_requires=["numpy>=1.14.3"],
    package_data={"visualization": ["gmt/quakecore-logo.png"]},
    scripts=[
        "visualization/comparisons/waveforms_sim_obs.py",
        "visualization/comparisons/psa_bias.py",
        "visualization/comparisons/psa_ratios.py",
        "visualization/comparisons/im_rrup.py",
        "visualization/comparisons/psa_ratios_rrup.py",
        "visualization/comparisons/psa_sim_obs.py",
        "visualization/test/test_im_plotting/test_im_plot.py",
        "visualization/im_plotting/im_plot.py",
        "visualization/gmt/plot_items.py",
        "visualization/gmt/plot_faults.py",
        "visualization/gmt/plot_srf_square.py",
        "visualization/gmt/plot_stations.py",
        "visualization/gmt/plot_vs30.py",
        "visualization/gmt/plot_seismo_single.py",
        "visualization/gmt/plot_gen_seismo.py",
        "visualization/gmt/plot_srf_map.py",
        "visualization/gmt/plot_obs.py",
        "visualization/gmt/plot_ts.py",
        "visualization/gmt/plot_ts_sum.py",
        "visualization/gmt/plot_transfer.py",
        "visualization/gmt/plot_faults_srf.py",
        "visualization/gmt/plot_srf_animation.py",
        "visualization/gmt/plot_deagg.py",
        "visualization/gmt/plot_srf_perspective.py",
        "visualization/gmt/plot_srf_map_global.py",
    ],
)
