[![Build Status](http://13.238.107.244:8080/job/visualization/badge/icon?build=last:${params.ghprbActualCommit=master)](http://13.238.107.244:8080/job/visualization)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Visualization

Visualization scripts, grouped into categories.

Install requires: [qcore](https://github.com/ucgmsim/qcore)\
Install command: ```pip install --user .```\
One time requirement: ```export PATH=$PATH:$HOME/.local/bin```


## /animation
Scripts with video / animated outputs.

## /im
Scripts relating to the visualization of Intensity Measures.

## /prototype
Prototypes that are being worked on.

## /sources
Datafile visualization. SRFs, VMs, etc.

**plot_srf_map**\
```plot_srf_map.py fault.srf```\
<img src="samples/plot_srf_map.jpg">

**plot_srf_slip_rise_rake.py**\
```plot_srf_slip_rise_rake.py fault.srf```\
<img src="samples/plot_srf_slip_rise_rake.jpg">

## /waveform
Scripts relating to plotting waveforms.

**waveforms.py**\
```waveforms.py accBB/ Benchmark BinaryAcc/BB.bin Comparison --n-stations 1```\
<img src="samples/REHS.png">
