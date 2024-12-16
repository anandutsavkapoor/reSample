import numpy as np
import os

# Stellar population parameters
STELLAR_TEMPLATE = "SB99"     # Alternative: "BPASS"
IMF_TYPE = "kroupa100"        # Alternative: "chab100" 
STAR_TYPE = "sin"            # Alternative: "bin"

# Create consistent model identifier
MODEL_PREFIX = f"{STELLAR_TEMPLATE}_{IMF_TYPE}_{STAR_TYPE}"
print(f'recollapse_data_{MODEL_PREFIX}.hdf5')
# Directory paths
MAIN_DIR = '/home/akapoor/pynb/models/SB99/template/'
CLOUDY_DATA_DIR = '/home/akapoor/cloudy/data/'
SED_DIR = os.path.join(MAIN_DIR, 'SED_unified_GASS10_prodRun/')
HDF5_FILENAME = f'recollapse_data_{MODEL_PREFIX}.hdf5'

# Constants
AGE_LIMIT = 30               # Myr
AGE_START = 0.1              # Myr
N_TEMPORAL_BINS = 15
M_STAR_MEAN = 10000         # Solar masses
SAMPLE_SIZE = 10**6        # For cloud mass distribution
EXPONENT = -1.8            # Cloud mass function slope
SAMPLE_SIZE_TIME = 1000    # For synthetic sfr data

# Simulation constants
LENGTH_UNIT = 'kpc'
AGE_UNIT = 'Myr'
SPREAD_RADIUS = 0.0       # kpc
SMOOTH_AGE_SCALE = 0.0    # Myr

# Parameter space
if STELLAR_TEMPLATE == 'SB99': # TODDLERS v0
    METALLICITIES = np.array([.001, .004, .008, .02, .04])
    STAR_FORMATION_EFFICIENCIES = np.array([.01, .025, .05, .075, .1, .125, .15])
    CLOUD_DENSITIES = np.around(10**np.arange(1, 3.5, .30102), 0)  # ~[10-2560] cm^-3
    MASS_BIN_CENTERS = np.array([10**np.around(i, 2) for i in np.arange(5.0, 6.8, .25)])
elif STELLAR_TEMPLATE == 'BPASS': # Placeholder, TBD
    METALLICITIES = np.array([.001, .004, .008, .02, .04])
    STAR_FORMATION_EFFICIENCIES = np.array([.01, .025, .05, .075, .1, .125, .15])
    CLOUD_DENSITIES = np.around(10**np.arange(1, 3.5, .30102), 0)  # ~[10-2560] cm^-3
    MASS_BIN_CENTERS = np.array([10**np.around(i, 2) for i in np.arange(5.0, 6.8, .25)])

################################## SFR normalized templates
# Parameter validation ranges
VALID_RANGES = {
    'Z': (0.0001, 0.05),
    'eta': (0.001, 0.2),
    'n_cl': (1.0, 5000.0)
}

# Wavelength grid settings
WAVELENGTH_GRID_FILE = "/home/akapoor/pynb/STROMGREN-S/template_generation/wavelength_grid/consolidated_wavelength_grid.txt"
#"/home/akapoor/pynb/STROMGREN-S/template_generation/wavelength_grid/continuum_wavelength_grid.txt"

# Interpolator settings  
SED_INTERPOLATOR_FILE = "/home/akapoor/pynb/STROMGREN-S/template_generation/TODDLERS_RGI_prodRun/TODDLERS_tot_hr_singleModels_lines_emergent=False_RGI.obj"
#"/home/akapoor/pynb/STROMGREN-S/template_generation/TODDLERS_RGI_prodRun/TODDLERS_totSED_lr_singleModels_RGI.obj"

# Input/Output directories
SED_OUTPUT_DIR = "sed_output_inci_hr"
RECOLLAPSE_SIM_DIR = "recollapse_sims"