[![Build Status](http://13.238.107.244:8080/job/visualization/badge/icon?build=last:${params.ghprbActualCommit=master)](http://13.238.107.244:8080/job/visualization)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Visualization

Visualization scripts, grouped into categories.

Install requires: [qcore](https://github.com/ucgmsim/qcore)\
Install command: ```pip install --user .```\
One time requirement: ```export PATH=$PATH:$HOME/.local/bin```


## /animation
Scripts with video / animated outputs.

**plot_srf_animation.py**\
```plot_srf_animation.py fault.srf```

https://user-images.githubusercontent.com/7464462/119743896-f194a580-bede-11eb-971f-9b32c9522de5.mov

**plot_ts.py**\
```plot_ts.py fault.srf```

https://user-images.githubusercontent.com/7464462/119743518-2fdd9500-bede-11eb-8883-104b55a2eb71.mov



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
