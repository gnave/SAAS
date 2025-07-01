"""
HDF5 data manager for spectroscopy data import wizard.
Handles structured storage of spectra, calculations, levels, and identifications.
"""

import h5py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HDF5Error(Exception):
    """Custom exception for HDF5 operations."""
    pass


class HDF5Manager:
    """Manager for HDF5 spectroscopy data files."""
    
    # Standard group structure for spectroscopy data
    STANDARD_GROUPS = {
        'Calculations': 'Calculated transition probabilities and related data',
        'Levels': 'Energy levels for different ions and states',
        'Standard_lamp_calibrations': 'Calibration certificates and data',
        'Previous_identifications': 'Known spectral line identifications',
        'Spectra': 'Spectrum data organized by measurement'
    }
    
    def __init__(self, filepath: str, mode: str = 'a'):
        """
        Initialize HDF5 manager.
        
        Args:
            filepath: Path to HDF5 file
            mode: File access mode ('r', 'w', 'a')
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self._file = None
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def open(self):
        """Open HDF5 file."""
        try:
            self._file = h5py.File(self.filepath, self.mode)
            logger.info(f"Opened HDF5 file: {self.filepath}")
        except Exception as e:
            raise HDF5Error(f"Failed to open HDF5 file {self.filepath}: {str(e)}")
    
    def close(self):
        """Close HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info(f"Closed HDF5 file: {self.filepath}")
    
    @property
    def file(self):
        """Get the HDF5 file handle."""
        return self._file
    
    def create_group_structure(self, force: bool = False):
        """
        Create standard group structure for spectroscopy data.
        
        Args:
            force: If True, recreate existing groups
        """
        if self._file is None:
            raise HDF5Error("HDF5 file not open")
        
        for group_name, description in self.STANDARD_GROUPS.items():
            if group_name in self._file:
                if force:
                    del self._file[group_name]
                    logger.info(f"Recreated group: {group_name}")
                else:
                    logger.info(f"Group already exists: {group_name}")
                    continue
            
            group = self._file.create_group(group_name)
            group.attrs['description'] = description
            group.attrs['created'] = datetime.now().isoformat()
            logger.info(f"Created group: {group_name}")
    
    def add_spectrum(self, spectrum_data: np.ndarray, wavenumbers: Optional[np.ndarray] = None,
                    metadata: Optional[Dict[str, Any]] = None, spectrum_name: str = None,
                    group_path: str = "Spectra") -> str:
        """
        Add spectrum data to HDF5 file.
        
        Args:
            spectrum_data: Spectrum intensity array
            wavenumbers: Optional wavenumber array
            metadata: Spectrum metadata from header file
            spectrum_name: Name for the spectrum dataset
            group_path: Group path to store spectrum
            
        Returns:
            Full path to created dataset
        """
        if self._file is None:
            raise HDF5Error("HDF5 file not open")
        
        if spectrum_name is None:
            spectrum_name = f"spectrum_{len(self._file[group_path]) + 1:03d}"
        
        # Create spectrum group if it doesn't exist
        if group_path not in self._file:
            self._file.create_group(group_path)
        
        spectrum_group = self._file[group_path]
        
        # Create spectrum dataset
        dataset_path = f"{group_path}/{spectrum_name}"
        
        # Handle existing datasets
        if spectrum_name in spectrum_group:
            choice = self._handle_existing_dataset(dataset_path, "spectrum")
            if choice == "skip":
                return dataset_path
            elif choice == "replace":
                del spectrum_group[spectrum_name]
        
        # Create spectrum dataset
        spectrum_ds = spectrum_group.create_dataset(spectrum_name, data=spectrum_data)
        
        # Add wavenumbers if provided (as separate dataset, not attribute)
        if wavenumbers is not None:
            wavenumber_ds = spectrum_group.create_dataset(f"{spectrum_name}_wavenumbers", 
                                                        data=wavenumbers, compression='gzip')
            spectrum_ds.attrs['has_wavenumbers'] = True
            spectrum_ds.attrs['wavenumber_range'] = [float(wavenumbers[0]), float(wavenumbers[-1])]
        
        # Add metadata as attributes (limit to essential ones to avoid size issues)
        if metadata:
            essential_keys = ['spect', 'npo', 'wstart', 'wstop', 'delw', 'resolutn', 
                            'nscan', 'source', 'day', 'spectype']
            for key in essential_keys:
                if key in metadata:
                    try:
                        value = metadata[key]
                        if isinstance(value, str):
                            spectrum_ds.attrs[key] = value
                        else:
                            spectrum_ds.attrs[key] = value
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Skipping metadata key '{key}': {e}")
        
        # Add processing timestamp
        spectrum_ds.attrs['imported'] = datetime.now().isoformat()
        spectrum_ds.attrs['npoints'] = len(spectrum_data)
        
        logger.info(f"Added spectrum: {dataset_path}")
        return dataset_path
    
    def add_calculations(self, calc_data: pd.DataFrame, dataset_name: str = None,
                        group_path: str = "Calculations") -> str:
        """
        Add calculated transition probabilities.
        
        Args:
            calc_data: DataFrame with columns [wavenumber, lower_key, upper_key, probability]
            dataset_name: Name for the calculations dataset
            group_path: Group path to store calculations
            
        Returns:
            Full path to created dataset
        """
        required_columns = ['wavenumber', 'lower_key', 'upper_key']
        if not all(col in calc_data.columns for col in required_columns):
            raise HDF5Error(f"Calculations data missing required columns: {required_columns}")
        
        if dataset_name is None:
            dataset_name = f"calculations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self._add_tabular_data(calc_data, dataset_name, group_path)
    
    def add_levels(self, levels_data: pd.DataFrame, dataset_name: str = None,
                  group_path: str = "Levels") -> str:
        """
        Add energy levels data.
        
        Args:
            levels_data: DataFrame with columns [energy, key, j_value, parity, lifetime]
            dataset_name: Name for the levels dataset
            group_path: Group path to store levels
            
        Returns:
            Full path to created dataset
        """
        required_columns = ['key']
        if not all(col in levels_data.columns for col in required_columns):
            raise HDF5Error(f"Levels data missing required columns: {required_columns}")
        
        if dataset_name is None:
            dataset_name = f"levels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self._add_tabular_data(levels_data, dataset_name, group_path)
    
    def add_identifications(self, ident_data: pd.DataFrame, dataset_name: str = None,
                           group_path: str = "Previous_identifications") -> str:
        """
        Add previous spectral line identifications.
        
        Args:
            ident_data: DataFrame with columns [wavenumber, lower_key, upper_key, intensity]
            dataset_name: Name for the identifications dataset
            group_path: Group path to store identifications
            
        Returns:
            Full path to created dataset
        """
        required_columns = ['wavenumber', 'lower_key', 'upper_key']
        if not all(col in ident_data.columns for col in required_columns):
            raise HDF5Error(f"Identifications data missing required columns: {required_columns}")
        
        if dataset_name is None:
            dataset_name = f"identifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self._add_tabular_data(ident_data, dataset_name, group_path)
    
    def add_calibration(self, calib_data: pd.DataFrame, dataset_name: str = None,
                       group_path: str = "Standard_lamp_calibrations") -> str:
        """
        Add calibration certificate data.
        
        Args:
            calib_data: DataFrame with wavelength and spectral radiance columns
            dataset_name: Name for the calibration dataset
            group_path: Group path to store calibration
            
        Returns:
            Full path to created dataset
        """
        if dataset_name is None:
            dataset_name = f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self._add_tabular_data(calib_data, dataset_name, group_path)
    
    def _add_tabular_data(self, data: pd.DataFrame, dataset_name: str, group_path: str) -> str:
        """
        Add tabular data to HDF5 file.
        
        Args:
            data: DataFrame to store
            dataset_name: Name for the dataset
            group_path: Group path to store data
            
        Returns:
            Full path to created dataset
        """
        if self._file is None:
            raise HDF5Error("HDF5 file not open")
        
        # Create group if it doesn't exist
        if group_path not in self._file:
            self._file.create_group(group_path)
        
        group = self._file[group_path]
        dataset_path = f"{group_path}/{dataset_name}"
        
        # Handle existing datasets
        if dataset_name in group:
            choice = self._handle_existing_dataset(dataset_path, "tabular data")
            if choice == "skip":
                return dataset_path
            elif choice == "replace":
                del group[dataset_name]
        
        # Convert DataFrame to structured array for HDF5
        # Handle string columns by converting to fixed-length strings
        data_copy = data.copy()
        
        # Create dtype mapping for structured array
        dtype_list = []
        for col in data_copy.columns:
            if data_copy[col].dtype == 'object':
                # Convert to string and determine max length
                max_len = min(max(len(str(val)) for val in data_copy[col]), 100)
                dtype_list.append((col, f'S{max_len}'))  # Use bytes instead of unicode
                # Force conversion to string then bytes
                data_copy[col] = data_copy[col].astype(str).str.encode('utf-8')
            else:
                dtype_list.append((col, data_copy[col].dtype))
        
        # Convert to structured array with explicit dtype
        records = np.array([tuple(row) for row in data_copy.values], dtype=dtype_list)
        dataset = group.create_dataset(dataset_name, data=records, compression='gzip')
        
        # Add metadata
        dataset.attrs['columns'] = [col.encode('utf-8') for col in data.columns]
        dataset.attrs['nrows'] = len(data)
        dataset.attrs['ncols'] = len(data.columns)
        dataset.attrs['imported'] = datetime.now().isoformat()
        
        logger.info(f"Added tabular data: {dataset_path}")
        return dataset_path
    
    def validate_keys(self, calculations_path: str, levels_path: str) -> List[str]:
        """
        Validate referential integrity between calculations and levels datasets.
        
        Args:
            calculations_path: Path to calculations dataset
            levels_path: Path to levels dataset
            
        Returns:
            List of validation errors
        """
        if self._file is None:
            raise HDF5Error("HDF5 file not open")
        
        errors = []
        
        try:
            # Load datasets
            calc_data = pd.DataFrame(self._file[calculations_path][:])
            levels_data = pd.DataFrame(self._file[levels_path][:])
            
            # Get keys
            level_keys = set(levels_data['key'].astype(str))
            calc_lower_keys = set(calc_data['lower_key'].astype(str))
            calc_upper_keys = set(calc_data['upper_key'].astype(str))
            
            # Find missing keys
            missing_lower = calc_lower_keys - level_keys
            missing_upper = calc_upper_keys - level_keys
            
            if missing_lower:
                errors.append(f"Missing lower level keys: {missing_lower}")
            if missing_upper:
                errors.append(f"Missing upper level keys: {missing_upper}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def _handle_existing_dataset(self, dataset_path: str, data_type: str) -> str:
        """
        Handle existing dataset conflicts.
        
        Args:
            dataset_path: Path to existing dataset  
            data_type: Type of data for user prompt
            
        Returns:
            Action to take: 'replace', 'skip', or 'append'
        """
        # For now, default to replace - in GUI version this would prompt user
        logger.warning(f"Dataset already exists: {dataset_path}. Replacing.")
        return "replace"
    
    def list_datasets(self, group_path: str = None) -> Dict[str, List[str]]:
        """
        List all datasets in the file or specific group.
        
        Args:
            group_path: Optional group path to list
            
        Returns:
            Dictionary mapping group names to dataset lists
        """
        if self._file is None:
            raise HDF5Error("HDF5 file not open")
        
        datasets = {}
        
        if group_path:
            if group_path in self._file:
                group = self._file[group_path]
                datasets[group_path] = list(group.keys())
            else:
                datasets[group_path] = []
        else:
            for group_name in self.STANDARD_GROUPS.keys():
                if group_name in self._file:
                    group = self._file[group_name]
                    datasets[group_name] = list(group.keys())
                else:
                    datasets[group_name] = []
        
        return datasets
    
    def get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Dictionary with dataset information
        """
        if self._file is None:
            raise HDF5Error("HDF5 file not open")
        
        if dataset_path not in self._file:
            raise HDF5Error(f"Dataset not found: {dataset_path}")
        
        dataset = self._file[dataset_path]
        
        info = {
            'path': dataset_path,
            'shape': dataset.shape,
            'dtype': str(dataset.dtype),
            'size': dataset.size,
            'attributes': dict(dataset.attrs)
        }
        
        return info
    
    def export_dataset(self, dataset_path: str, output_path: str, format: str = 'csv'):
        """
        Export dataset to external format.
        
        Args:
            dataset_path: Path to dataset in HDF5 file
            output_path: Output file path
            format: Export format ('csv', 'json')
        """
        if self._file is None:
            raise HDF5Error("HDF5 file not open")
        
        if dataset_path not in self._file:
            raise HDF5Error(f"Dataset not found: {dataset_path}")
        
        dataset = self._file[dataset_path]
        
        if format.lower() == 'csv':
            df = pd.DataFrame(dataset[:])
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df = pd.DataFrame(dataset[:])
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise HDF5Error(f"Unsupported export format: {format}")
        
        logger.info(f"Exported dataset {dataset_path} to {output_path}")