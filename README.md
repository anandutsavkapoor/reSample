# reSample
A script for converting TODDLERS SED Family input files to TODDLERS Cloud SED Family by decomposing particles into cloud components.

## Usage
The main.py converts TODDLERS SED Family input files to TODDLERS Cloud SED Family input format, 
examples included in that file.

## Important Notes
- Units for length (x, y, z, h) - will be converted to kpc internally
- Time (age) - will be converted to Myr internally
- Mass should be in solar masses (Msun)
- Velocity components (if present) are inherited by split components from parent particle

## Dependencies:
- NumPy
- Matplotlib
- h5py
- Astropy

## License
MIT License

## Contact
For questions and support, please reach out to Anand Utsav Kapoor.