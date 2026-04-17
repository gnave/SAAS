# importers.py (FULLY DOCUMENTED)

import os
import h5py
import numpy as np
import pandas as pd
from struct import unpack
import re
import h5_manager

def parse_hdr_file(header_filepath: str):
    """
    Parses a proprietary .hdr metadata file into a dictionary.

    The .hdr file format is a key-value store with some specific rules:
    - Lines can be commented out with '/'.
    - Values are separated from keys by '='.
    - String values are enclosed in single quotes.
    - Comments can appear on the same line as a value, separated by '/'.
    - A key named 'continue' signifies that its value should be appended to the
      value of the previously defined key.

    The function attempts to cast values to integer, then float, and finally
    defaults to a string.

    Args:
        header_filepath (str): The path to the .hdr file.

    Returns:
        dict: A dictionary containing the parsed metadata.
    """
    metadata = {}
    last_key = None
    with open(header_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore empty lines or lines that are full-line comments or the end marker.
            if not line or line.startswith('/') or line.startswith('END'): continue
            if '=' not in line: continue
            
            key_part, value_part = line.split('=', 1)
            key = key_part.strip()
            
            # Strip trailing comments from the value part.
            if '/' in value_part:
                value_str = value_part.split('/', 1)[0].strip()
            else:
                value_str = value_part.strip()
            
            # Strip quotes from string values.
            if value_str.startswith("'") and value_str.endswith("'"):
                value_str = value_str[1:-1].strip()
            
            # Handle the special 'continue' key for multi-line values.
            if key == 'continue' and last_key is not None:
                metadata[last_key] = f"{metadata[last_key]} {value_str}"
                continue
                
            # Attempt to cast the value to the most appropriate numeric type.
            try: value = int(value_str)
            except ValueError:
                try: value = float(value_str)
                except ValueError: value = value_str
            
            metadata[key] = value
            last_key = key
            
    return metadata

def import_spectrum_pair(h5_filepath, raw_data_file, header_file, is_calibration_spectrum=False, target_spectrum_group=None):
    """
    Imports a spectrum and handles 'Complex' vs 'Real' data formats.
    If 'data_is' is 'Complex', extracts only every other point (the real parts).
    """
    base_name = os.path.splitext(os.path.basename(raw_data_file))[0]
    sanitized_name = base_name.replace('.', '_').replace('-', '_')

    # 1. Parse metadata FIRST to check data format
    metadata_dict = parse_hdr_file(header_file)
    
    # 2. Read the binary data
    spec_data = np.fromfile(raw_data_file, dtype=np.float32)

    # 3. Handle Complex vs Real slicing
    # 'data_is' might be 'Complex', 'Real', or missing (default to Real)
    data_format = str(metadata_dict.get('data_is', 'Real')).strip()
    
    if 'Complex' in data_format:
        # Extract every other point starting from 0 (Real parts)
        # Slicing syntax [start:stop:step]
        spec_data = spec_data[::2]
        print(f"Detected Complex data: Discarded imaginary components. Points remaining: {len(spec_data)}")
    else:
        print(f"Detected Real data: Reading all points. Total: {len(spec_data)}")

    # 4. Setup HDF5 paths
    if is_calibration_spectrum:
        if not target_spectrum_group:
            raise ValueError("A target spectrum group must be provided for calibration spectra.")
        group_path = f"{target_spectrum_group}/Calibration_Spectra"
        dataset_name = sanitized_name
        with h5py.File(h5_filepath, 'a') as f:
            if group_path not in f:
                f.create_group(group_path)
    else:
        spectrum_name = sanitized_name
        base_spectrum_group = f"/Spectra/{spectrum_name}"
        raw_data_group = f"{base_spectrum_group}/Raw_Data"

        with h5py.File(h5_filepath, 'a') as f:
            if base_spectrum_group in f:
                raise FileExistsError(f"A spectrum group named '{spectrum_name}' already exists.")
            f.create_group(raw_data_group)
            f.create_group(f"{base_spectrum_group}/Raw_Linelists")
            f.create_group(f"{base_spectrum_group}/Calibrated_Linelists")
            f.create_group(f"{base_spectrum_group}/Calibration_Spectra")
            f.create_group(f"{base_spectrum_group}/Identified_Lines")
            
        group_path = raw_data_group
        dataset_name = 'spectrum'
    
    # 5. Save and finalize
    metadata_dict['original_data_filename'] = os.path.basename(raw_data_file)
    metadata_dict['original_header_filename'] = os.path.basename(header_file)
    
    h5_manager.add_dataset_to_file(h5_filepath, group_path, dataset_name, spec_data, metadata=metadata_dict)
    print(f"--- Successfully imported {data_format} spectrum '{sanitized_name}' ---")
    
def read_linelist(lin_filepath: str):
    """
    Parses a binary .lin linelist file.

    This function reads the proprietary binary format of a .lin file, which consists
    of a short header followed by a series of fixed-width records, one for each
    spectral line. It uses `struct.unpack` to decode the binary data into a list
    of dictionaries.

    Args:
        lin_filepath (str): The path to the .lin file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a line
              and contains its parsed parameters.
    """
    linel = []
    if not os.path.exists(lin_filepath):
        raise FileNotFoundError(f"Linelist file not found: {lin_filepath}")
    
    with open(lin_filepath, "rb") as flin:
        try:
            # Read the header to get the number of lines in the file.
            nlin = unpack("i", flin.read(4))[0]
            # Skip the rest of the file header.
            flin.read(4)
            flin.read(312)
            
            # Loop through each line record.
            for _ in range(nlin):
                sp = {}
                # Unpack the binary data according to the fixed format string.
                # d=double, f=float, h=short
                sp['sig'], sp['xint'], sp['width'], sp['dmping'], sp['itn'], sp['ihold'] = unpack("dfffhh", flin.read(24))
                sp['tags'] = flin.read(4)
                sp['epstot'], sp['epsevn'], sp['epsodd'], sp['epsran'], sp['spare'] = unpack("fffff", flin.read(20))
                # Decode the identification string, ignoring errors.
                sp['ident'] = flin.read(32).decode('utf-8', errors='ignore').strip('\x00')
                linel.append(sp)
        except Exception as e:
            print(f"Error reading linelist file {lin_filepath}: {e}")
            raise IOError(f"Failed to parse binary .lin file: {e}")
            
    return linel

def import_binary_linelist(h5_filepath, lin_filepath, target_spectrum_group):
    """
    Reads a binary .lin file, converts it to a DataFrame, and saves it to the
    HDF5 file under the specified spectrum group.

    Also applies a known transformation to the 'dmping' parameter to convert it
    to a more conventional 0-1 scale.

    Args:
        h5_filepath (str): Path to the HDF5 project file.
        lin_filepath (str): Path to the .lin file to import.
        target_spectrum_group (str): The HDF5 path to the parent spectrum group
                                     (e.g., '/Spectra/MySpectrum').
    """
    print(f"--- Importing binary linelist into spectrum: {target_spectrum_group} ---")
    linelist_data = read_linelist(lin_filepath)
    if not linelist_data:
        print("Warning: Linelist file was empty or failed to parse.")
        return
        
    df = pd.DataFrame(linelist_data)
    # Apply the required transformation to the damping parameter.
    df['dmping'] = (df['dmping'] - 1.0) / 25.0
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(lambda x: x.decode('utf-8', errors='ignore').strip('\x00'))
    
    base_name = os.path.splitext(os.path.basename(lin_filepath))[0]
    table_name = base_name.replace('.', '_').replace('-', '_')
    group_path = f"{target_spectrum_group}/Raw_Linelists"
    metadata = {
        'conversion_applied_dmping': '(original_value - 1.0) / 25.0',
        'original_filename': os.path.basename(lin_filepath)
    }
    h5_manager.add_pandas_table(h5_filepath, group_path, table_name, df, metadata_dict=metadata)

def import_calibrated_linelist(h5_filepath, txt_filepath, target_spectrum_group):
    """
    Parses and imports a calibrated text linelist (.txt).

    This format has a few header lines containing metadata followed by a table
    of space-separated values for each identified line.

    Args:
        h5_filepath (str): Path to the HDF5 project file.
        txt_filepath (str): Path to the .txt linelist file.
        target_spectrum_group (str): HDF5 path to the parent spectrum group.
    """
    print(f"--- Importing calibrated text linelist into: {target_spectrum_group} ---")
    header_metadata = {}
    with open(txt_filepath, 'r') as f:
        all_lines = f.readlines()
        
    line_iterator = iter(all_lines)
    # Parse metadata from the first few header lines.
    line1 = next(line_iterator, ''); line2 = next(line_iterator, ''); line3 = next(line_iterator, '')
    if "wavcorr" in line1:
        match = re.search(r'wavcorr\s*=\s*([0-9.E+-]+)', line1)
        if match: header_metadata['wavcorr'] = float(match.group(1))
    header_metadata['air_correction'] = "applied" if "applied" in line2 else "not applied"
    header_metadata['intensity_calibration'] = "applied" if "APPLIED" in line3 else "not applied"

    # Parse the main data table.
    parsed_rows = []
    for line in line_iterator:
        parts = line.strip().split()
        if not parts or not parts[0].isdigit(): continue # Skip non-data lines
        if len(parts) < 14:
            print(f"Warning: Skipping malformed data line: {line.strip()}"); continue
        try:
            # Handle potential '*' characters indicating bad fits by replacing them with 'nan'.
            cleaned_parts = [p if '*' not in p else 'nan' for p in parts]
            row = {
                'line': int(cleaned_parts[0]), 'wavenumber': float(cleaned_parts[1]),
                'peak': float(cleaned_parts[2]), 'width': float(cleaned_parts[3]),
                'dmp': float(cleaned_parts[4]), 'eq_width': float(cleaned_parts[5]),
                'itn': int(cleaned_parts[6]), 'H': int(cleaned_parts[7]),
                'tags': cleaned_parts[8], 'epstot': float(cleaned_parts[9]),
                'epsevn': float(cleaned_parts[10]), 'epsodd': float(cleaned_parts[11]),
                'epsran': float(cleaned_parts[12]), 'identification': " ".join(cleaned_parts[13:-1]),
                'wavelength_air': float(cleaned_parts[-1])
            }
            parsed_rows.append(row)
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse data line: {line.strip()} | Error: {e}"); continue
            
    if not parsed_rows: raise ValueError("No valid data lines could be parsed from the file.")
    df = pd.DataFrame(parsed_rows)
    numeric_cols = ['wavenumber', 'peak', 'width', 'dmp', 'eq_width', 'epstot', 'epsevn', 'epsodd', 'epsran', 'wavelength_air']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    table_name = os.path.splitext(os.path.basename(txt_filepath))[0].replace('.','_').replace('-', '_')
    group_path = f"{target_spectrum_group}/Calibrated_Linelists"
    header_metadata['original_filename'] = os.path.basename(txt_filepath)
    h5_manager.add_pandas_table(h5_filepath, group_path, table_name, df, metadata_dict=header_metadata)

def import_lamp_calibration(h5_filepath, cal_filepath, user_metadata):
    """
    Parses and imports a standard lamp calibration file.

    This format consists of comment lines starting with '#' and data lines
    with two space-separated columns (wavelength, spectral radiance).

    Args:
        h5_filepath (str): Path to the HDF5 project file.
        cal_filepath (str): Path to the lamp calibration .txt file.
        user_metadata (dict): A dictionary of metadata provided by the user
                              (e.g., author, notes).
    """
    print(f"--- Importing lamp calibration file: {cal_filepath} ---")
    comments, data_lines = [], []
    with open(cal_filepath, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line: continue
            if stripped_line.startswith('#'):
                comments.append(stripped_line.lstrip('#').strip())
            else:
                data_lines.append(stripped_line)
                
    if not data_lines: raise ValueError("No valid data lines found in the calibration file.")
    
    # Use StringIO to treat the list of data lines as a virtual file for pandas.
    from io import StringIO
    data_io = StringIO("\n".join(data_lines))
    df = pd.read_csv(data_io, sep=r'\s+', header=None, names=['wavelength_nm', 'spectral_radiance'])
    
    full_metadata = user_metadata.copy()
    full_metadata['header_comments'] = "\n".join(comments) # Store comments from file header
    full_metadata['original_filename'] = os.path.basename(cal_filepath)
    table_name = os.path.splitext(os.path.basename(cal_filepath))[0].replace('.', '_').replace('-', '_')
    group_path = "/Standard_Lamp_Calibrations"
    h5_manager.add_pandas_table(h5_filepath, group_path, table_name, df, metadata_dict=full_metadata)

def parse_generic_text_file(filepath, file_type='delimited', delimiter=',', has_header=False, col_widths=None):
    """
    A flexible, general-purpose parser for tabular text files.

    This function acts as a wrapper around pandas `read_csv` (for delimited files)
    and `read_fwf` (for fixed-width files), providing a unified interface for the
    Import Wizard.

    Args:
        filepath (str): Path to the text file.
        file_type (str): Either 'delimited' or 'fixed'.
        delimiter (str): The delimiter character (e.g., ',', 'space', 'tab').
        has_header (bool): True if the first row of the file is a header.
        col_widths (list, optional): A list of integers specifying the width of
                                     each column for fixed-width files.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed data. Returns an empty
                      DataFrame on failure.
    """
    header_row = 0 if has_header else None
    try:
        if file_type == 'delimited':
            delimiter_regex = r'\s+' if delimiter == 'space' else delimiter
            df = pd.read_csv(filepath, sep=delimiter_regex, header=header_row, engine='python')
        elif file_type == 'fixed':
            df = pd.read_fwf(filepath, widths=col_widths, header=header_row)
        else:
            print(f"Error: Unknown file type '{file_type}'"); return pd.DataFrame()
        
        # If no header, assign generic column names.
        if not has_header:
            df.columns = [f'Column {i+1}' for i in range(len(df.columns))]
        return df
    except Exception as e:
        print(f"Error parsing file: {e}"); return pd.DataFrame()