"""
Install using pip, e.g. pip install ./visualization
use --no-deps to prevent re-installation of dependencies
use -I to force re-install
"""
from setuptools import setup


setup(
    name="visualization",
    version="1.0.0",
    packages=['visualization'],
    url="https://github.com/ucgmsim/visualization",
    description="visualization code",
    install_requires=["numpy>=1.14.3"],
)
