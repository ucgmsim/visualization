"""
Install using pip, e.g. pip install ./visualization
use --no-deps to prevent re-installation of dependencies
use -I to force re-install
"""

from glob import glob
from setuptools import setup

scripts = glob("animation/*.py")
scripts.extend(glob("im/*.py"))
scripts.extend(glob("prototype/*.py"))
scripts.extend(glob("sources/*.py"))
scripts.extend(glob("waveform/*.py"))

setup(
    name="visualization",
    version="1.0.0",
    packages=["visualization"],
    url="https://github.com/ucgmsim/visualization",
    description="visualization code",
    install_requires=["numpy>=1.14.3"],
    package_data={"visualization": glob("visualization/data/*")},
    include_package_data=True,
    scripts=scripts,
)
