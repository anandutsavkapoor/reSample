import numpy as np
from astropy import units as u
import os
from names_and_constants import *
from resample_simulation import process_simulation

def generate_temp_filename(original_filename, suffix):
    """Generate unique temporary filename based on original filename"""
    dirname = os.path.dirname(original_filename) or '.'
    basename = os.path.basename(original_filename)
    name_without_ext = os.path.splitext(basename)[0]
    return os.path.join(dirname, f"{name_without_ext}_{suffix}_temp.dat")

def cleanup_temp_files(temp_files):
    """Remove temporary files"""
    for file in temp_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {file}: {e}")

def preprocess_input_file(input_filename, output_filename, column_info):
    """
    Create an intermediate file with the required column structure for the package.
    
    Parameters
    ----------
    input_filename : str
        Path to the original input file
    output_filename : str
        Path where the intermediate file will be saved
    column_info : dict
        Dictionary specifying the column indices and units for required fields
        Example:
        {
            'x': {'index': 0, 'unit': 'kpc'},
            'y': {'index': 1, 'unit': 'kpc'},
            'z': {'index': 2, 'unit': 'kpc'},
            ...
        }
    """
    # Verify all required columns are present
    required_columns = {'x', 'y', 'z', 'h', 'age', 'metallicity', 'sfe', 'density', 'mass'}
    missing_columns = required_columns - set(column_info.keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Read input file
    data = []
    with open(input_filename, 'r') as f:
        while True:
            line = f.readline()
            if not line.startswith('#'):
                first_data = line.split()
                data.append(list(map(float, first_data)))
                break
        for line in f:
            data.append(list(map(float, line.split())))
    
    data = np.array(data)
    
    # Create mapping of unused columns
    required_indices = {info['index'] for name, info in column_info.items() 
                       if name in required_columns}
    all_indices = {info['index'] for info in column_info.values()}
    unused_columns = {i: data[:, i] for i in all_indices if i not in required_indices}
    
    # Prepare output data
    output_data = np.zeros((len(data), 9))
    col_mapping = {'x': 0, 'y': 1, 'z': 2, 'h': 3, 'age': 4, 
                  'metallicity': 5, 'sfe': 6, 'density': 7, 'mass': 8}
    
    for col_name, info in column_info.items():
        if col_name in required_columns:
            idx = col_mapping[col_name]
            values = data[:, info['index']]
            
            # Apply unit conversions using astropy if unit is specified
            if 'unit' in info:
                if col_name in ['x', 'y', 'z', 'h']:
                    try:
                        quantity = values * u.Unit(info['unit'])
                        values = quantity.to(LENGTH_UNIT).value
                    except Exception as e:
                        raise ValueError(f"Length unit conversion failed for {col_name}: {e}")
                        
                elif col_name == 'age':
                    try:
                        quantity = values * u.Unit(info['unit'])
                        values = quantity.to(AGE_UNIT).value
                    except Exception as e:
                        raise ValueError(f"Time unit conversion failed for age: {e}")
                        
                elif col_name == 'mass':
                    try:
                        quantity = values * u.Unit(info['unit'])
                        values = quantity.to('Msun').value
                    except Exception as e:
                        raise ValueError(f"Mass unit conversion failed: {e}")
            
            output_data[:, idx] = values
    
    # Write intermediate file
    with open(output_filename, 'w') as f:
        f.write("# Intermediate file for TODDLERS processing\n")
        f.write("# Column 1: x-coordinate (kpc)\n")
        f.write("# Column 2: y-coordinate (kpc)\n")
        f.write("# Column 3: z-coordinate (kpc)\n")
        f.write("# Column 4: smoothing length h (kpc)\n")
        f.write("# Column 5: age (Myr)\n")
        f.write("# Column 6: metallicity (1)\n")
        f.write("# Column 7: SFE (1)\n")
        f.write("# Column 8: density (cm^-3)\n")
        f.write("# Column 9: mass (Msun)\n")
        
        for row in output_data:
            line = "  ".join(f"{val:.6e}" for val in row)
            f.write(line + "\n")
    
    return unused_columns

def postprocess_output_file(input_filename, output_filename, unused_columns, column_info, num_original_particles):
    """
    Take intermediate2 file and merge with unused columns, maintaining original column positions
    while keeping cloud mass and weight at the end. Display internal units for processed quantities.
    
    Parameters
    ----------
    input_filename : str
        Path to the intermediate2 file containing processed data with cloud mass and weight
    output_filename : str
        Path where the final output file will be saved
    unused_columns : dict
        Dictionary of column indices and their corresponding data that wasn't used in
        processing (velocity columns, for example)
    column_info : dict
        Original column information dictionary containing indices and units
    num_original_particles : int
        Number of particles in the original input file
    """
    # Read the intermediate2 file
    data = []
    with open(input_filename, 'r') as f:
        while True:
            line = f.readline()
            if not line.startswith('#'):
                data.append(list(map(float, line.split())))
                break
        for line in f:
            data.append(list(map(float, line.split())))
    
    data = np.array(data)
    particles_per_original = len(data) / num_original_particles

    # Extract cloud mass and weight (last two columns)
    cloud_mass = data[:, -2]
    weight = data[:, -1]
    processed_data = data[:, :-2]  # All columns except cloud_mass and weight

    # Create mapping of processed columns to their original positions
    processed_cols_mapping = {
        'x': 0,
        'y': 1,
        'z': 2,
        'h': 3,
        'age': 4,
        'metallicity': 5,
        'sfe': 6,
        'density': 7
    }

    # Get original positions for processed columns
    original_positions = {column_info[col]['index']: idx for col, idx in processed_cols_mapping.items()}

    # Expand unused columns
    expanded_unused = {idx: np.repeat(values, particles_per_original) 
                      for idx, values in unused_columns.items()}

    # Find the highest original column index
    max_original_idx = max(
        max(info['index'] for name, info in column_info.items() if name != 'mass'), # mass is unused
        max(unused_columns.keys() if unused_columns else [0])
    )

    # Create final array with space for all columns plus cloud mass and weight
    final_data = np.zeros((len(data), max_original_idx + 3))

    # Fill in the processed data at their original positions
    for orig_pos, proc_idx in original_positions.items():
        if proc_idx < processed_data.shape[1]:
            final_data[:, orig_pos] = processed_data[:, proc_idx]

    # Fill in the unused columns at their original positions
    for col_idx, values in expanded_unused.items():
        final_data[:, col_idx] = values

    # Add cloud mass and weight at the end
    final_data[:, -2] = cloud_mass
    final_data[:, -1] = weight

    # Write output file
    with open(output_filename, 'w') as f:
        f.write("# SKIRT 9 import format for SFR with TODDLERS model\n")

        # Define internal units for processed quantities
        internal_units = {
            'x': 'kpc',
            'y': 'kpc',
            'z': 'kpc',
            'h': 'kpc',
            'age': 'Myr',
            'metallicity': '1',
            'sfe': '1',
            'density': '1/cm3'
        }

        # Write headers for all columns
        for i in range(final_data.shape[1] - 2):  # Exclude cloud mass and weight
            # Find if this column is in the original column_info
            col_info = next((
                (name, info) for name, info in column_info.items() 
                if info['index'] == i and name != 'mass'  # Exclude mass column
            ), None)

            if col_info:
                name, info = col_info
                # Use internal units for processed quantities, original units for unused columns
                if name in internal_units:
                    unit = internal_units[name]
                else:
                    unit = info.get('unit', '1')
                
                # Use standard names for processed columns
                if name == 'x':
                    f.write(f"# Column {i+1}: x-coordinate ({unit})\n")
                elif name == 'y':
                    f.write(f"# Column {i+1}: y-coordinate ({unit})\n")
                elif name == 'z':
                    f.write(f"# Column {i+1}: z-coordinate ({unit})\n")
                elif name == 'h':
                    f.write(f"# Column {i+1}: size h ({unit})\n")
                elif name == 'age':
                    f.write(f"# Column {i+1}: Age ({unit})\n")
                elif name == 'metallicity':
                    f.write(f"# Column {i+1}: metallicity ({unit})\n")
                elif name == 'sfe':
                    f.write(f"# Column {i+1}: SFE ({unit})\n")
                elif name == 'density':
                    f.write(f"# Column {i+1}: Cloud Particle Density ({unit})\n")
                else:
                    f.write(f"# Column {i+1}: {name} ({unit})\n")

        # Add cloud mass and weight headers
        f.write(f"# Column {final_data.shape[1]-1}: Associated Cloud mass (Msun)\n")
        f.write(f"# Column {final_data.shape[1]}: Weight (1)\n")

        # Write data
        for row in final_data:
            line = "  ".join(f"{val:.6e}" for val in row)
            f.write(line + "\n")

def convert_to_new_format(input_filename, output_filename, column_info, 
                          sfe_override=None, density_override=None):
    """
    Convert TODDLERS SED Family input file to TODDLERS Cloud SED Family input.
    
    Parameters
    ----------
    input_filename : str
        Path to input file
    output_filename : str
        Path for final output file
    column_info : dict
        Dictionary specifying the column indices and units for the fields in the input file

        It is advisable to use this template:
        Without velocity columns:
        column_info = {
            'x': {'index': 0, 'unit': 'kpc'},  # Unit will be converted to kpc if different
            'y': {'index': 1, 'unit': 'pc'},   # Unit will be converted to kpc if different
            'z': {'index': 2, 'unit': 'Mpc'},  # Unit will be converted to kpc if different
            'h': {'index': 3, 'unit': 'pc'},   # Unit will be converted to kpc if different
            'age': {'index': 4, 'unit': 'Gyr'}, # Unit will be converted to Myr if different
            'metallicity': {'index': 5},
            'sfe': {'index': 6},
            'density': {'index': 7},
            'mass': {'index': 8, 'unit': 'Msun'}
        }

        With velocity columns:
        column_info = {
            'x': {'index': 0, 'unit': 'kpc'},
            'y': {'index': 1, 'unit': 'kpc'},
            'z': {'index': 2, 'unit': 'kpc'},
            'h': {'index': 3, 'unit': 'kpc'},
            'v_x': {'index': 4, 'unit': 'km/s'},
            'v_y': {'index': 5, 'unit': 'km/s'},
            'v_z': {'index': 6, 'unit': 'km/s'},
            'v_disp': {'index': 7, 'unit': 'km/s'},
            'age': {'index': 8, 'unit': 'Gyr'},
            'metallicity': {'index': 9},
            'sfe': {'index': 10},
            'density': {'index': 11},
            'mass': {'index': 12, 'unit': 'Msun'}
        }    

    sfe_override : float, optional
        Override star formation efficiency value to use another value
    density_override : float, optional
        Override density value to use another value

    Note: The split components of each particle inherit the same velocity values as the parent particle.
    Thus, there will be repeated entries in these columns in the output file.
    """

    # Generate unique temporary filenames
    temp_file1 = generate_temp_filename(input_filename, "intmdt1")
    temp_file2 = generate_temp_filename(input_filename, "intmdt2")
    temp_files = [temp_file1, temp_file2]
    
    try:
        # Count original particles
        with open(input_filename, 'r') as f:
            num_original_particles = sum(1 for line in f if not line.startswith('#'))
        
        # Step 1: Preprocess
        unused_columns = preprocess_input_file(input_filename, temp_file1, column_info)
        
        # Step 2: Process with package - hardcoded units and parameters to avoid confusion
        process_simulation(
            input_filename=temp_file1,
            input_length_unit=LENGTH_UNIT,
            input_age_unit=AGE_UNIT,
            output_filename=temp_file2,
            sfe_override=sfe_override,
            density_override=density_override,
            r=SPREAD_RADIUS,
            smooth_age_scale=SMOOTH_AGE_SCALE
        )
        
        # Finaly files
        postprocess_output_file(temp_file2, output_filename, unused_columns, 
                              column_info, num_original_particles)
    
    finally:
        cleanup_temp_files(temp_files)

if __name__ == "__main__":
    # Examples (read the comments please)
    #1 column information in the input file which including velocity components, use this template
    file_column_info = {
        'x': {'index': 0, 'unit': 'kpc'},
        'y': {'index': 1, 'unit': 'kpc'},
        'z': {'index': 2, 'unit': 'kpc'},
        'h': {'index': 3, 'unit': 'kpc'},
        'v_x': {'index': 4, 'unit': 'km/s'},
        'v_y': {'index': 5, 'unit': 'km/s'},
        'v_z': {'index': 6, 'unit': 'km/s'},
        'v_disp': {'index': 7, 'unit': 'km/s'},
        'age': {'index': 8, 'unit': 'Gyr'},
        'metallicity': {'index': 9},
        'sfe': {'index': 10},
        'density': {'index': 11},
        'mass': {'index': 12, 'unit': 'Msun'}
    }
    
    # Process the file with automatic temporary file handling
    convert_to_new_format(
        input_filename="test_velo.dat",
        output_filename="test_velo_out.dat",
        column_info=file_column_info,
        sfe_override=0.025,
        density_override=320.,
    )

    #2 column information for a file without velocity components, use this template
    file_column_info = {
        'x': {'index': 0, 'unit': 'kpc'},
        'y': {'index': 1, 'unit': 'kpc'},
        'z': {'index': 2, 'unit': 'kpc'},
        'h': {'index': 3, 'unit': 'kpc'},
        'age': {'index': 4, 'unit': 'Myr'}, # in this file the time is in Myr
        'metallicity': {'index': 5},
        'sfe': {'index': 6},
        'density': {'index': 7},
        'mass': {'index': 8, 'unit': 'Msun'}
    }
    
    # Process the file with automatic temporary file handling
    convert_to_new_format(
        input_filename="test_no_velo.dat",
        output_filename="test_no_velo_out.dat",
        column_info=file_column_info,
        sfe_override=0.025,
        density_override=320.,
    )