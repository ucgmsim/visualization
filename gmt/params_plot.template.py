"""
this file should not be added to, parameters should be handled with argparse instead.
"""
import os

fault_model = 'Automatically generated source model'
# <HH> is converted automatically
vel_model = 'NZVM v1.65 h=<HH>km'
# pick a predefined region or model_params sim domain used ('')
# 'CANTERBURY', 'WIDERCANT', 'MIDNZ', 'SOUTHISLAND'
region = None

# PGV plotting
class PGV:
    dpi = 300
    title = event_title + ' PGV'
    # cpt for PGV should be kept static for the most part
    cpt = 'hot'
    cpt_min = 0
    # don't exclude values close to 0
    lowcut = None
    cpt_legend = 'peak ground velocity (cm/s)'
    # crop overlay to land
    land_crop = False

# MMI plotting
class MMI:
    dpi = 300
    title = event_title + ' MMI'
    cpt_legend = 'modified mercalli intensity'
    # MMI deals with +- 1 values, needs smaller convergence limit
    convergence_limit = 0.1
    # crop MMI to land
    land_crop = False

# observed / simulated seismogram plotting
class SEISMO:
    title = 'Observed Ground Motions'
    wd = os.path.abspath('GMT_SEIS_WD')
    # output filename excluding file extension
    name = 'ObservedMap'
    dpi = 300
    width = '7i'
    # override velocity model region default = None
    # eg: region = (x_min, x_max, y_min, y_max)
    region = None
    # list of stations that should always be plotted
    wanted_stats = []
    # minimum distance stations should be appart (km)
    min_dist = None
    # GMT seismo files
    # set obs_src or sim_src to None to only plot one
    obs_ts = 'obsVelBB/%s.090'
    obs_src = 'gmt-seismo_obs.xy'
    obs_colour = 'black'
    sim_ts = 'simVelBB/%s.090'
    sim_src = 'gmt-seismo_sim.xy'
    sim_colour = 'red'
    seis_width = 0.3
    # timestep cutoff: with dt = 0.005, 200 per second
    max_ts = 20000
    # timeseries x-azimuth, x length and max y length in degrees
    ts_xaz = 90
    ts_xlen = None
    ts_ymax = None

class SRF:
    # default is the filename without extention
    title = None
    # PNG output, 'srfdir' for same location as SRF
    out_dir = 'srfdir'
    # use average rake (True) or midpoint rake (False)
    rake_average = False
    # length of the longest rake arrow (based on slip)
    rake_length = 0.4
    # place rake arrows every X subfaults, less than 1 for automatic
    rake_decimation = 0

