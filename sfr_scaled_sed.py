import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Union, List
import logging
import warnings
import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import TODDLERS_recollapse_handling as TORCH
from names_and_constants import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SimulationParameters:
    """Store and validate simulation parameters."""
    Z: u.Quantity  # metallicity (dimensionless)
    eta: float     # star formation efficiency (dimensionless)
    n: u.Quantity  # cluster number density (cm^-3)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        # Convert inputs to Quantities if they aren't already
        if not isinstance(self.Z, u.Quantity):
            self.Z = self.Z * u.dimensionless_unscaled
        if not isinstance(self.n, u.Quantity):
            self.n = self.n * u.cm**-3
            
        # Validate ranges
        if not (VALID_RANGES['Z'][0] <= self.Z.value <= VALID_RANGES['Z'][1]):
            raise ValueError(f"Z={self.Z.value} outside valid range {VALID_RANGES['Z']}")
        if not (VALID_RANGES['eta'][0] <= self.eta <= VALID_RANGES['eta'][1]):
            raise ValueError(f"eta={self.eta} outside valid range {VALID_RANGES['eta']}")
        if not (VALID_RANGES['n_cl'][0] <= self.n.value <= VALID_RANGES['n_cl'][1]):
            raise ValueError(f"n_cl={self.n.value} outside valid range {VALID_RANGES['n_cl']}")

class SEDmanipulator:
    """Handle SED SFR scaled computations."""
    
    def __init__(
        self,
        wavelength_grid_file: Union[str, Path],
        sed_interpolator: RegularGridInterpolator,
        recollapse_sim_dir: Union[str, Path]
    ):
        """Initialize the SED analyzer."""
        self.recollapse_sim_dir = Path(recollapse_sim_dir)
        self.recollapse_sim_dir.mkdir(parents=True, exist_ok=True)
        self.sed_interpolator = sed_interpolator
        
        # Load wavelength grid from file
        self.wavelength_grid = self._load_wavelength_grid(wavelength_grid_file)
        self.age_limit = AGE_LIMIT * u.Myr
        self.age_start = AGE_START * u.Myr
            
    def _load_wavelength_grid(self, grid_file: Union[str, Path]) -> u.Quantity:
        """Load wavelength grid from file and filter to desired range."""
        try:
            # Load the wavelength grid and convert to microns
            grid = np.loadtxt(grid_file) * u.m

            logger.info(f"Wavelength grid: {len(grid)} points between "
                    f"{grid[0].to(u.micron):.4f} and {grid[-1].to(u.micron):.4f}")
            
            return grid
            
        except Exception as e:
            raise RuntimeError(f"Error loading wavelength grid: {str(e)}")
            
    def compute_sfr_scaled_sed(
        self,
        params: SimulationParameters,
        load_sim: bool = True,
    ) -> Tuple[float, float, u.Quantity]:
        """Compute time-averaged SED for given parameters."""
        # Create filename using physical parameters
        filename = f"recollapse_sim_Z={params.Z.value:.3f}_epsilon={params.eta:.3f}_nCl={params.n.to(u.cm**-3).value:.1f}.sim"
        sim_file = self.recollapse_sim_dir / filename
        
        # Load or create simulation
        sim = self._load_or_create_simulation(sim_file, params, load_sim)
        
        # Process simulation and compute SED
        return self._process_simulation(sim, params)
        
    def _load_or_create_simulation(
        self,
        sim_file: Path,
        params: SimulationParameters,
        load_sim: bool
    ) -> 'TORCH.StarFormationSimulation':
        """Load existing simulation or create new one."""
        try:
            if load_sim:
                sim_dir = sim_file.parent
                if sim_dir.exists():
                    z_pattern = f"{params.Z.value:.6f}".rstrip('0').rstrip('.')
                    eta_pattern = f"{params.eta:.6f}".rstrip('0').rstrip('.')
                    n_pattern = f"{params.n.to(u.cm**-3).value:.6f}".rstrip('0').rstrip('.')
                    
                    for existing_file in sim_dir.glob("recollapse_sim_*.sim"):
                        name_parts = existing_file.stem.split('_')
                        try:
                            z_val = name_parts[2].split('=')[1]
                            eps_val = name_parts[3].split('=')[1]
                            n_val = name_parts[4].split('=')[1]
                            
                            z_val = f"{float(z_val):.6f}".rstrip('0').rstrip('.')
                            eps_val = f"{float(eps_val):.6f}".rstrip('0').rstrip('.')
                            n_val = f"{float(n_val):.6f}".rstrip('0').rstrip('.')
                            
                            if (z_val == z_pattern and 
                                eps_val == eta_pattern and 
                                n_val == n_pattern):
                                logger.info(f"Loading existing simulation from {existing_file}")
                                with open(existing_file, 'rb') as f:
                                    return pickle.load(f)
                        except (IndexError, ValueError):
                            continue
            
            logger.info("Creating new simulation")
            sim = TORCH.StarFormationSimulation(
                Z=params.Z.value,
                epsilon=params.eta,
                n_cl=params.n.to(u.cm**-3).value,
                simulation_data=[],
                num_particles=SAMPLE_SIZE_TIME
            )
            
            with open(sim_file, 'wb') as f:
                pickle.dump(sim, f)
            
            self._plot_mass_distributions(sim, params)
            return sim
            
        except Exception as e:
            raise RuntimeError(f"Error handling simulation: {str(e)}")

    def _plot_mass_distributions(
        self,
        sim: 'TORCH.StarFormationSimulation',
        params: SimulationParameters
    ) -> None:
        """Plot mass distributions and their contributions."""
        M_star_particles_per_bin = [
            np.sum(sim.M_star_particles[sim.temporal_bins == bin_idx]) 
            for bin_idx in range(N_TEMPORAL_BINS)
        ]
        
        adjusted_M_star_particles_per_bin = [
            np.sum(
                sim.M_cloud_particles[sim.temporal_bins == bin_idx] * 
                sim.w_ki[sim.temporal_bins == bin_idx] * 
                sim.epsilon[sim.temporal_bins == bin_idx]
            ) 
            for bin_idx in range(N_TEMPORAL_BINS)
        ]
        
        M_tot_bin = (np.array(adjusted_M_star_particles_per_bin) + 
                    sim.recollapse_contributions_per_time_bin)
        
        figures_dir = Path("./figures")
        figures_dir.mkdir(exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            plt.figure(figsize=(5, 5))
            plt.subplot(1, 1, 1)
            
            plt.plot(M_tot_bin/M_star_particles_per_bin, label='total')
            plt.plot(
                np.array(adjusted_M_star_particles_per_bin) / M_star_particles_per_bin, 
                label='pristine'
            )
            plt.plot(
                sim.recollapse_contributions_per_time_bin / M_star_particles_per_bin, 
                label='recollapse'
            )
            
            plt.xlabel("temporal bins")
            plt.ylabel("contributions")
            plt.legend()
            plt.tight_layout()
            
            save_figname = figures_dir / f"adjusted_distribs_{MODEL_PREFIX}_Z_{params.Z.value:.3f}_sfe_{params.eta:.3f}_density_{params.n.value:.1f}.pdf"
            plt.savefig(save_figname)
            plt.close()
            
            logger.info(f"Distribution plot saved to {save_figname}")

    def _process_simulation(
        self,
        sim: 'TORCH.StarFormationSimulation',
        params: SimulationParameters
    ) -> Tuple[float, float, u.Quantity]:
        """Process simulation data and compute SED."""
        # Calculate ages with units
        t_arr = sim.ages * u.Myr
        mask = t_arr <= self.age_limit # safety
        
        # Prepare arrays for interpolation 
        t_arr = np.clip(t_arr[mask].value, self.age_start.value, self.age_limit.value)
        
        # Create 2D arrays
        wl_grid_2d = np.repeat(self.wavelength_grid.value[np.newaxis, :], len(t_arr), axis=0)
        t_2d = np.repeat(t_arr[:, np.newaxis], len(self.wavelength_grid), axis=1)
        
        # Stack arrays for interpolation
        # Order must match RGI creation order:
        # log_wl_grid_cont, t, logZ, etaSF, logN_cl, logM_cl
        x_arr_sed = np.dstack((
            np.log10(wl_grid_2d),                                            # log wavelength
            t_2d,                                                            # time
            np.log10(params.Z.value) * np.ones_like(t_2d),                  # log Z
            params.eta * np.ones_like(t_2d),                                # etaSF (not logged)
            np.log10(params.n.to(u.cm**-3).value) * np.ones_like(t_2d),    # log n
            np.log10(sim.M_cloud_particles[mask])[:, np.newaxis] * np.ones_like(t_2d)  # log M
        ))
        
        # Compute SED using the provided interpolator
        log_sed = self.sed_interpolator(x_arr_sed.reshape(-1, 6))
        log_sed = self.check_and_fix_nans_in_sed(log_sed, x_arr_sed, replacement_value=-300.0)

        sed = (10**log_sed).reshape(len(t_arr), len(self.wavelength_grid)) * u.W/u.m
        
        # Calculate SFR with units
        mass_based_sfr = (np.sum(sim.M_star_particles[mask]) * u.Msun / self.age_limit).to(u.Msun/u.yr)
        
        # Calculate final SED per SFR with units
        sed_per_sfr = np.sum(sim.w_ki[mask][:, np.newaxis] * sed, axis=0) / mass_based_sfr
        
        return params.eta, params.n.to(u.cm**-3).value, sed_per_sfr

    def check_and_fix_nans_in_sed(self, log_sed: np.ndarray, x_arr_sed: np.ndarray, 
                                replacement_value: float = -300.0) -> np.ndarray:
        """
        Check for NaN values in SED calculation results, report summary, and replace with a small number.
        
        Args:
            log_sed: Array of log SED values from interpolator
            x_arr_sed: Array of input coordinates used for interpolation
            replacement_value: Value to use for replacing NaNs (in log space)
        
        Returns:
            np.ndarray: log_sed array with NaNs replaced by replacement_value
        """
        # Check for NaN values
        nan_mask = np.isnan(log_sed)
        if np.any(nan_mask):
            total_nans = np.sum(nan_mask)
            
            # Get ranges where NaNs occur
            nan_coords = x_arr_sed.reshape(-1, 6)[nan_mask]
            print(f"\nFound {total_nans} NaN values")
            print(f"Wavelength range: {10**np.min(nan_coords[:,0])*1e6:.3f} to {10**np.max(nan_coords[:,0])*1e6:.3f} μm")
            
            # Replace NaN values
            log_sed[nan_mask] = replacement_value
        
        return log_sed

    def plot_sfr_scaled_sed(
        self,
        seds_dict: dict,
        params: SimulationParameters,
        output_file: Optional[Union[str, Path]] = None
    ) -> None:
        """Plot multiple SEDs for different densities."""
        plt.figure(figsize=(12, 8))
        
        colors = ['b', 'r', 'g']  # Colors for different densities
        for (n, sed_per_sfr), color in zip(seds_dict.items(), colors):
            plt.loglog(
                self.wavelength_grid.to(u.micron),
                sed_per_sfr * self.wavelength_grid,
                f'{color}-',
                linewidth=2,
                label=f'n = {n:.1f} cm$^{{-3}}$'
            )
        
        plt.xlabel('Wavelength [μm]')
        plt.ylabel('SED per SFR [W/M$_{\odot}$ yr$^{-1}$]')
        title_str = f'SFR-Scaled SED Comparison\nZ={params.Z.value:.3f}, η={params.eta:.3f}'
        plt.title(title_str)
        
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        save_str = f'SFR-Scaled_SED_comparison_{MODEL_PREFIX}.pdf'
        output_path = Path(output_file or save_str)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")

if __name__ == "__main__":
    try:
        # Base parameters
        base_params = SimulationParameters(
            Z=0.02 * u.dimensionless_unscaled,
            eta=0.025,
            n=320. * u.cm**-3  # This will be overridden
        )
        
        # Set up interpolator
        with open(SED_INTERPOLATOR_FILE, 'rb') as file_sed_interp:
            sed_interpolator = pickle.load(file_sed_interp)
        
        # Initialize analyzer
        manip = SEDmanipulator(
            wavelength_grid_file=WAVELENGTH_GRID_FILE,
            sed_interpolator=sed_interpolator,
            recollapse_sim_dir=RECOLLAPSE_SIM_DIR
        )
        
        # Densities to analyze
        densities = [80., 320., 1280.]
        seds_dict = {}
        
        # Compute SEDs for each density
        for n in densities:
            params = SimulationParameters(
                Z=base_params.Z,
                eta=base_params.eta,
                n=n * u.cm**-3
            )
            _, _, sed_per_sfr = manip.compute_sfr_scaled_sed(params)
            seds_dict[n] = sed_per_sfr
        
        # Plot all SEDs
        manip.plot_sfr_scaled_sed(seds_dict, base_params)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)