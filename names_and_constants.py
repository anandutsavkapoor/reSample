import numpy as np
import os

# names
MAIN_DIR = '/home/akapoor/pynb/models/SB99/template/'
CLOUDY_DATA_DIR = '/home/akapoor/cloudy/data/'
SED_DIR = os.path.join(MAIN_DIR, 'SED_unified_GASS10_prodRun/')
HDF5_FILENAME = 'recollapse_data.hdf5'

# constants
AGE_LIMIT        = 30
N_TEMPORAL_BINS  = 30
M_STAR_MEAN      = 10000
MASS_BIN_CENTERS = np.array([10**np.around(i, 2) for i in np.arange(5.0, 6.8, .25)])
SAMPLE_SIZE      = 10**6
EXPONENT         = -1.8

# main file constants
LENGTH_UNIT = 'kpc'
AGE_UNIT = 'Myr'
SPREAD_RADIUS = 0.25  # kpc
SMOOTH_AGE_SCALE = 0.1  # Myr
    