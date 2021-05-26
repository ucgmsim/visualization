[![Build Status](http://13.238.107.244:8080/job/visualization/badge/icon?build=last:${params.ghprbActualCommit=master)](http://13.238.107.244:8080/job/visualization)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Visualization

Visualization scripts, grouped into categories.

Install requires: [qcore](https://github.com/ucgmsim/qcore)\
Install command: ```pip install --user .```\
One time requirement: ```export PATH=$PATH:$HOME/.local/bin```\

## Waveforms

Scripts relating to plotting waveforms.

**waveforms.py**\
```waveforms.py accBB/ Benchmark BinaryAcc/BB.bin Comparison --n-stations 1```\
<img src="samples/REHS.png">
