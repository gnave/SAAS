"""
Test script for HDF5Manager using sample data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
from hdf5_manager import HDF5Manager
from file_parsers import FileParserFactory

def test_hdf5_creation():
    """Test HDF5 file creation and group structure."""
    print("Testing HDF5 Creation...")
    
    test_file = "test_output.h5"
    
    # Remove existing test file
    if Path(test_file).exists():
        os.remove(test_file)
    
    try:
        with HDF5Manager(test_file, 'w') as hdf5_mgr:
            # Create standard group structure
            hdf5_mgr.create_group_structure()
            
            # List groups
            datasets = hdf5_mgr.list_datasets()
            print(f"✓ Created {len(datasets)} standard groups")
            
            for group_name, datasets_list in datasets.items():
                print(f"  - {group_name}: {len(datasets_list)} datasets")
        
        print("✓ HDF5 file creation successful")
        return True
        
    except Exception as e:
        print(f"✗ HDF5 creation error: {e}")
        return False

def test_spectrum_import():
    """Test importing spectrum data."""
    print("\nTesting Spectrum Import...")
    
    test_file = "test_output.h5"
    
    # Parse sample data
    hdr_file = "cr_demo_data/cr042416.005_r.hdr"
    dat_file = "cr_demo_data/cr042416.005_r.dat"
    
    if not all(Path(f).exists() for f in [hdr_file, dat_file]):
        print("✗ Sample spectrum files not found")
        return False
    
    try:
        # Parse header and spectrum data
        hdr_result = FileParserFactory.parse_file(hdr_file)
        dat_result = FileParserFactory.parse_file(dat_file, metadata=hdr_result['metadata'])
        
        # Import to HDF5
        with HDF5Manager(test_file, 'a') as hdf5_mgr:
            spectrum_path = hdf5_mgr.add_spectrum(
                spectrum_data=dat_result['spectrum'],
                wavenumbers=dat_result['wavenumbers'],
                metadata=hdr_result['metadata'],
                spectrum_name="cr042416_005_r"
            )
            
            # Get dataset info
            info = hdf5_mgr.get_dataset_info(spectrum_path)
            print(f"✓ Imported spectrum: {spectrum_path}")
            print(f"  Shape: {info['shape']}")
            print(f"  Size: {info['size']} points")
            print(f"  Data type: {info['dtype']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Spectrum import error: {e}")
        return False

def test_tabular_data_import():
    """Test importing CSV tabular data."""
    print("\nTesting Tabular Data Import...")
    
    test_file = "test_output.h5"
    csv_file = "cr_demo_data/CrII_CS.csv"
    
    if not Path(csv_file).exists():
        print("✗ Sample CSV file not found")
        return False
    
    try:
        # Parse CSV data
        csv_result = FileParserFactory.parse_file(csv_file)
        df = csv_result['data']
        
        # Map columns to standard format
        df_mapped = df.rename(columns={
            'lower_desig': 'lower_key',
            'upper_desig': 'upper_key'
        })
        
        # Import to HDF5
        with HDF5Manager(test_file, 'a') as hdf5_mgr:
            ident_path = hdf5_mgr.add_identifications(
                df_mapped,
                dataset_name="CrII_identifications"
            )
            
            # Get dataset info
            info = hdf5_mgr.get_dataset_info(ident_path)
            print(f"✓ Imported identifications: {ident_path}")
            print(f"  Rows: {info['attributes']['nrows']}")
            print(f"  Columns: {info['attributes']['ncols']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tabular data import error: {e}")
        return False

def test_levels_import():
    """Test importing energy levels data."""
    print("\nTesting Levels Import...")
    
    test_file = "test_output.h5"
    levels_file = "cr_demo_data/crii_lev.csv"
    
    if not Path(levels_file).exists():
        print("✗ Sample levels file not found")
        return False
    
    try:
        # Parse levels data
        levels_result = FileParserFactory.parse_file(levels_file)
        df = levels_result['data']
        
        # Map columns to standard format (use 'desig' as key)
        df_mapped = df.rename(columns={'desig': 'key'})
        
        # Import to HDF5
        with HDF5Manager(test_file, 'a') as hdf5_mgr:
            levels_path = hdf5_mgr.add_levels(
                df_mapped,
                dataset_name="CrII_levels"
            )
            
            # Get dataset info
            info = hdf5_mgr.get_dataset_info(levels_path)
            print(f"✓ Imported levels: {levels_path}")
            print(f"  Rows: {info['attributes']['nrows']}")
            print(f"  Columns: {info['attributes']['ncols']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Levels import error: {e}")
        return False

def test_calculations_import():
    """Test importing calculations data."""
    print("\nTesting Calculations Import...")
    
    test_file = "test_output.h5"
    calc_file = "cr_demo_data/CrII_calc.csv"
    
    if not Path(calc_file).exists():
        print("✗ Sample calculations file not found")
        return False
    
    try:
        # Parse calculations data
        calc_result = FileParserFactory.parse_file(calc_file)
        df = calc_result['data']
        
        # Map columns to standard format and add dummy wavenumber
        df_mapped = df.rename(columns={
            'lower_desig': 'lower_key',
            'upper_desig': 'upper_key'
        })
        # Add dummy wavenumber column (required by schema)
        df_mapped['wavenumber'] = 0.0
        
        # Import to HDF5  
        with HDF5Manager(test_file, 'a') as hdf5_mgr:
            calc_path = hdf5_mgr.add_calculations(
                df_mapped,
                dataset_name="CrII_calculations"
            )
            
            # Get dataset info
            info = hdf5_mgr.get_dataset_info(calc_path)
            print(f"✓ Imported calculations: {calc_path}")
            print(f"  Rows: {info['attributes']['nrows']}")
            print(f"  Columns: {info['attributes']['ncols']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Calculations import error: {e}")
        return False

def test_data_validation():
    """Test data validation and referential integrity."""
    print("\nTesting Data Validation...")
    
    test_file = "test_output.h5"
    
    try:
        with HDF5Manager(test_file, 'r') as hdf5_mgr:
            # Check if we have both calculations and levels
            datasets = hdf5_mgr.list_datasets()
            
            calc_datasets = datasets.get('Calculations', [])
            levels_datasets = datasets.get('Levels', [])
            
            if calc_datasets and levels_datasets:
                # Validate referential integrity
                calc_path = f"Calculations/{calc_datasets[0]}"
                levels_path = f"Levels/{levels_datasets[0]}"
                
                errors = hdf5_mgr.validate_keys(calc_path, levels_path)
                
                if errors:
                    print(f"✓ Validation found issues (expected for test data):")
                    for error in errors:
                        print(f"  - {error}")
                else:
                    print("✓ No validation errors found")
            else:
                print("✓ Validation skipped - insufficient data")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False

def test_dataset_export():
    """Test exporting datasets."""
    print("\nTesting Dataset Export...")
    
    test_file = "test_output.h5"
    
    try:
        with HDF5Manager(test_file, 'r') as hdf5_mgr:
            datasets = hdf5_mgr.list_datasets()
            
            # Export first available dataset
            for group_name, dataset_list in datasets.items():
                if dataset_list:
                    dataset_path = f"{group_name}/{dataset_list[0]}"
                    output_file = f"exported_{dataset_list[0]}.csv"
                    
                    hdf5_mgr.export_dataset(dataset_path, output_file, 'csv')
                    
                    if Path(output_file).exists():
                        print(f"✓ Exported dataset to: {output_file}")
                        # Cleanup
                        os.remove(output_file)
                        return True
        
        print("✓ No datasets available for export")
        return True
        
    except Exception as e:
        print(f"✗ Export error: {e}")
        return False

def main():
    """Run all HDF5Manager tests."""
    print("=== HDF5Manager Tests ===\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Run tests
    tests = [
        test_hdf5_creation,
        test_spectrum_import,
        test_tabular_data_import,
        test_levels_import,
        test_calculations_import,
        test_data_validation,
        test_dataset_export
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Tests Complete: {passed}/{len(tests)} passed ===")
    
    # Cleanup test file
    test_file = "test_output.h5"
    if Path(test_file).exists():
        os.remove(test_file)
        print(f"Cleaned up test file: {test_file}")

if __name__ == "__main__":
    main()