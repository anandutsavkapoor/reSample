# reSample
A script for converting simulation particles for use with TODDLERS SED Family (Cloud mode) by decomposing particles into cloud mass components and accounting for stellar mass from recollapse events.

## Purpose
This tool allows galaxy simulation particles to be properly processed for use with TODDLERS SED Family (Cloud mode). It:

- Decomposes each star particle into multiple cloud mass components based on a power-law cloud mass distribution
- Calculates appropriate weight adjustments to account for recollapse events in star-forming regions
- Produces an output file that properly weights SEDs to maintain the galaxy's star formation rate

## Usage
The `main.py` script provides an easy way to convert simulation input files to TODDLERS SED Family (Cloud mode) format. Here's an example:

```python
from main import convert_to_new_format

# Define your column mapping (required)
column_info = {
    'x': {'index': 0, 'unit': 'kpc'},
    'y': {'index': 1, 'unit': 'kpc'},
    'z': {'index': 2, 'unit': 'kpc'},
    'h': {'index': 3, 'unit': 'kpc'},
    'v_x': {'index': 4, 'unit': 'km/s'},  # Velocity columns are preserved
    'v_y': {'index': 5, 'unit': 'km/s'},  
    'v_z': {'index': 6, 'unit': 'km/s'},
    'age': {'index': 7, 'unit': 'Gyr'},   # Will be converted to Myr internally
    'metallicity': {'index': 8},
    'sfe': {'index': 9},
    'density': {'index': 10},
    'mass': {'index': 11, 'unit': 'Msun'}
}

# Process the file
convert_to_new_format(
    input_filename="galaxy_sim.dat",
    output_filename="galaxy_sim_processed.dat",
    column_info=column_info,
    sfe_override=0.05,      # Optional: override SFE value in input file
    density_override=160.0  # Optional: override density value in input file
)
```

For simpler files without velocity components:

```python
column_info = {
    'x': {'index': 0, 'unit': 'kpc'},
    'y': {'index': 1, 'unit': 'kpc'},
    'z': {'index': 2, 'unit': 'kpc'},
    'h': {'index': 3, 'unit': 'kpc'},
    'age': {'index': 4, 'unit': 'Myr'},
    'metallicity': {'index': 5},
    'sfe': {'index': 6},
    'density': {'index': 7},
    'mass': {'index': 8, 'unit': 'Msun'}
}
```

## Important Notes
- **Units for length (`x`, `y`, `z`, `h`)** - will be converted to kpc internally
- **Time (`age`)** - will be converted to Myr internally
- **Mass should be in solar masses (Msun)**
- **Velocity components (if present)** are inherited by split components from the parent particle

## Dependencies
- NumPy  
- Matplotlib  
- h5py  
- Astropy  

## License
MIT License

## Contact
For questions and support, please reach out to **Anand Utsav Kapoor**.

