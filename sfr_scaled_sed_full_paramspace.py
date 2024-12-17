# sed_generator.py
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
from tqdm import tqdm
import psutil
import pickle
from astropy import units as u
from sfr_scaled_sed import SEDmanipulator, SimulationParameters
from names_and_constants import *

class SEDGenerator:
    def __init__(self, output_dir=SED_OUTPUT_DIR, max_workers=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Determine resolution from output directory
        is_hr = 'hr' in str(output_dir)
        
        # Memory-based worker calculation
        # Use 40GB for high resolution, 19GB for low resolution
        base_memory = 35e9 if is_hr else 19e9
        memory_per_process = base_memory * (SAMPLE_SIZE_TIME / 1000)  # Estimate per process
        available_memory = psutil.virtual_memory().available
        suggested_workers = max(1, int(available_memory / memory_per_process))
        self.max_workers = min(suggested_workers, os.cpu_count()) if max_workers is None else max_workers
        
        self.setup_logging()
        self.log_system_info()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'sed_generation_{MODEL_PREFIX}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_system_info(self):
        """Log system information and configuration parameters."""
        self.logger.info(
            f"Initializing SED Generator\n"
            f"System Configuration:\n"
            f"- Available Memory: {psutil.virtual_memory().available / 1e9:.2f} GB\n"
            f"- CPU Cores: {os.cpu_count()}\n"
            f"- Workers: {self.max_workers}\n"
            f"Input Parameters:\n"
            f"- Wavelength Grid File: {WAVELENGTH_GRID_FILE}\n"
            f"- SED Interpolator File: {SED_INTERPOLATOR_FILE}\n"
            f"- Output Directory: {self.output_dir}\n"
            f"- Sample Size: {SAMPLE_SIZE_TIME}\n"
            f"Parameter Ranges:\n"
            f"- Metallicities: {METALLICITIES}\n"
            f"- Star Formation Efficiencies: {STAR_FORMATION_EFFICIENCIES}\n"
            f"- Cloud Densities: {CLOUD_DENSITIES}"
        )

    def process_parameter_combination(self, params):
        """Process a single parameter combination."""
        Z, eta, n_cl = params
        try:
            # Generate output filename
            output_file = self.output_dir / f"sed_sfr_scaled_{MODEL_PREFIX}_Z_{Z:.3f}_eta_{eta:.3f}_n_{n_cl:.1f}.txt"
            
            if output_file.exists():
                self.logger.info(f"Skipping existing file: {output_file}")
                return True

            # Load interpolator in each process to avoid memory sharing issues
            with open(SED_INTERPOLATOR_FILE, 'rb') as f:
                sed_interpolator = pickle.load(f)

            # Create simulation parameters
            params = SimulationParameters(
                Z=Z * u.dimensionless_unscaled,
                eta=eta,
                n=n_cl * u.cm**-3
            )

            # Initialize SEDmanipulator
            manip = SEDmanipulator(
                wavelength_grid_file=WAVELENGTH_GRID_FILE,
                sed_interpolator=sed_interpolator,
                recollapse_sim_dir=RECOLLAPSE_SIM_DIR
            )

            # Compute SED
            _, _, sed_per_sfr = manip.compute_sfr_scaled_sed(params, load_sim=True)

            # Save wavelength grid and SED values
            save_data = np.column_stack((manip.wavelength_grid.value, sed_per_sfr.value))
            np.savetxt(output_file, save_data, fmt='%.18e', 
                      header='Wavelength[m] SED_per_SFR[W/m/Msun/yr]')

            self.logger.info(f"Successfully generated SED for Z={Z}, eta={eta}, n_cl={n_cl}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing Z={Z}, eta={eta}, n_cl={n_cl}: {str(e)}")
            return False

    def generate_all_seds(self):
        """Generate SEDs for all parameter combinations."""
        parameter_combinations = [
            (Z, eta, n_cl) 
            for Z in METALLICITIES
            for eta in STAR_FORMATION_EFFICIENCIES
            for n_cl in CLOUD_DENSITIES
        ]
        
        total_combinations = len(parameter_combinations)
        self.logger.info(f"Processing {total_combinations} parameter combinations using {self.max_workers} workers")
        
        failures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_parameter_combination, parameter_combinations),
                total=total_combinations,
                desc="Generating SEDs"
            ))
            
            # Track failures
            for params, success in zip(parameter_combinations, results):
                if not success:
                    failures.append(params)

        if failures:
            self.logger.error(f"Failed parameter combinations: {failures}")
        else:
            self.logger.info("All SEDs generated successfully")

if __name__ == "__main__":
    generator = SEDGenerator()
    generator.generate_all_seds()