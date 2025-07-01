"""
Test script for file parsers using sample data.
"""

import sys
from pathlib import Path
from file_parsers import FileParserFactory, HdrFileParser, DatFileParser, CsvFileParser

def test_hdr_parser():
    """Test header file parser with sample data."""
    print("Testing HDR Parser...")
    
    hdr_file = "cr_demo_data/cr042416.005_r.hdr"
    if not Path(hdr_file).exists():
        print(f"Sample HDR file not found: {hdr_file}")
        return False
    
    try:
        parser = HdrFileParser()
        result = parser.parse(hdr_file)
        
        print(f"✓ Parsed {len(result['metadata'])} metadata entries")
        print(f"✓ File type: {result['file_type']}")
        
        # Check some key metadata
        metadata = result['metadata']
        print(f"✓ Instrument: {metadata.get('spect', 'N/A')}")
        print(f"✓ Number of points: {metadata.get('npo', 'N/A')}")
        print(f"✓ Wavenumber start: {metadata.get('wstart', 'N/A')}")
        print(f"✓ Dispersion: {metadata.get('delw', 'N/A')}")
        
        # Validate
        if parser.validate(result):
            print("✓ HDR validation passed")
            return True, result
        else:
            print("✗ HDR validation failed")
            return False, None
            
    except Exception as e:
        print(f"✗ HDR parser error: {e}")
        return False, None

def test_dat_parser():
    """Test binary data file parser."""
    print("\nTesting DAT Parser...")
    
    dat_file = "cr_demo_data/cr042416.005_r.dat"
    if not Path(dat_file).exists():
        print(f"Sample DAT file not found: {dat_file}")
        return False
    
    # First get metadata from HDR file
    hdr_success, hdr_result = test_hdr_parser()
    if not hdr_success:
        print("✗ Cannot test DAT parser without HDR metadata")
        return False
    
    try:
        parser = DatFileParser()
        result = parser.parse(dat_file, hdr_result['metadata'])
        
        print(f"✓ Parsed spectrum with {result['npoints']} points")
        print(f"✓ Spectrum range: {result['spectrum'].min():.2e} to {result['spectrum'].max():.2e}")
        
        if result['wavenumbers'] is not None:
            print(f"✓ Wavenumber range: {result['wavenumbers'][0]:.2f} to {result['wavenumbers'][-1]:.2f} cm⁻¹")
        
        # Validate
        if parser.validate(result):
            print("✓ DAT validation passed")
            return True, result
        else:
            print("✗ DAT validation failed")
            return False, None
            
    except Exception as e:
        print(f"✗ DAT parser error: {e}")
        return False, None

def test_csv_parser():
    """Test CSV file parser with sample data."""
    print("\nTesting CSV Parser...")
    
    csv_file = "cr_demo_data/CrII_CS.csv"
    if not Path(csv_file).exists():
        print(f"Sample CSV file not found: {csv_file}")
        return False
    
    try:
        parser = CsvFileParser()
        result = parser.parse(csv_file)
        
        print(f"✓ Parsed CSV with {result['nrows']} rows and {result['ncols']} columns")
        print(f"✓ Columns: {result['columns']}")
        print(f"✓ Delimiter: '{result['delimiter']}'")
        
        # Show sample data
        df = result['data']
        print(f"✓ Sample data:")
        print(df.head(3).to_string(index=False))
        
        # Validate
        if parser.validate(result):
            print("✓ CSV validation passed")
            return True, result
        else:
            print("✗ CSV validation failed")
            return False, None
            
    except Exception as e:
        print(f"✗ CSV parser error: {e}")
        return False, None

def test_factory():
    """Test the parser factory."""
    print("\nTesting Parser Factory...")
    
    test_files = [
        ("cr_demo_data/cr042416.005_r.hdr", "HDR"),
        ("cr_demo_data/CrII_CS.csv", "CSV")
    ]
    
    for filepath, file_type in test_files:
        if not Path(filepath).exists():
            print(f"✗ Test file not found: {filepath}")
            continue
        
        try:
            result = FileParserFactory.parse_file(filepath)
            print(f"✓ Factory successfully parsed {file_type} file")
            print(f"  File type detected: {result['file_type']}")
        except Exception as e:
            print(f"✗ Factory failed for {file_type}: {e}")

def main():
    """Run all parser tests."""
    print("=== File Parser Tests ===\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    # Run tests
    test_hdr_parser()
    test_dat_parser() 
    test_csv_parser()
    test_factory()
    
    print("\n=== Tests Complete ===")

if __name__ == "__main__":
    main()