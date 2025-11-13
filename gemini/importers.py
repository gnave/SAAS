# importers.py (Definitive Final Version)

import os
import h5py
import numpy as np
import pandas as pd
from struct import unpack
import re
import h5_manager

def parse_hdr_file(header_filepath: str):
    metadata = {}
    last_key = None
    with open(header_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('/') or line.startswith('END'): continue
            if '=' not in line: continue
            key_part, value_part = line.split('=', 1)
            key = key_part.strip()
            if '/' in value_part:
                value_str = value_part.split('/', 1)[0].strip()
            else:
                value_str = value_part.strip()
            if value_str.startswith("'") and value_str.endswith("'"):
                value_str = value_str[1:-1].strip()
            if key == 'continue' and last_key is not None:
                metadata[last_key] = f"{metadata[last_key]} {value_str}"
                continue
            try: value = int(value_str)
            except ValueError:
                try: value = float(value_str)
                except ValueError: value = value_str
            metadata[key] = value
            last_key = key
    return metadata

def import_spectrum_pair(h5_filepath, raw_data_file, header_file, is_calibration_spectrum=False, target_spectrum_group=None):
    base_name = os.path.splitext(os.path.basename(raw_data_file))[0]
    sanitized_name = base_name.replace('.', '_').replace('-', '_')
    if is_calibration_spectrum:
        if not target_spectrum_group:
            raise ValueError("A target spectrum group must be provided for calibration spectra.")
        print(f"--- Importing CALIBRATION spectrum into: {target_spectrum_group} ---")
        group_path = f"{target_spectrum_group}/Calibration_Spectra"
        dataset_name = sanitized_name
        with h5py.File(h5_filepath, 'a') as f:
            if not group_path in f: f.create_group(group_path)
    else:
        print(f"--- Starting import for MAIN spectrum: {raw_data_file} ---")
        spectrum_name = sanitized_name
        base_spectrum_group = f"/Spectra/{spectrum_name}"
        raw_data_group = f"{base_spectrum_group}/Raw_Data"
        linelists_group = f"{base_spectrum_group}/Raw_Linelists"
        calibrated_linelists_group = f"{base_spectrum_group}/Calibrated_Linelists"
        calibration_spectra_group = f"{base_spectrum_group}/Calibration_Spectra"
        identified_lines_group = f"{base_spectrum_group}/Identified_Lines" # NEW

        with h5py.File(h5_filepath, 'a') as f:
            if base_spectrum_group in f:
                raise FileExistsError(f"A spectrum group named '{spectrum_name}' already exists.")
            f.create_group(raw_data_group)
            f.create_group(linelists_group)
            f.create_group(calibrated_linelists_group)
            f.create_group(calibration_spectra_group)
            f.create_group(identified_lines_group) # NEW
            
        group_path = raw_data_group
        dataset_name = 'spectrum'
    spec_data = np.fromfile(raw_data_file, dtype=np.float32)
    metadata_dict = parse_hdr_file(header_file)
    metadata_dict['original_data_filename'] = os.path.basename(raw_data_file)
    metadata_dict['original_header_filename'] = os.path.basename(header_file)
    h5_manager.add_dataset_to_file(h5_filepath, group_path, dataset_name, spec_data, metadata=metadata_dict)
    print(f"--- Successfully imported spectrum '{sanitized_name}' into '{group_path}' ---")

def read_linelist(lin_filepath: str):
    linel = []
    if not os.path.exists(lin_filepath):
        raise FileNotFoundError(f"Linelist file not found: {lin_filepath}")
    with open(lin_filepath, "rb") as flin:
        try:
            nlin = unpack("i", flin.read(4))[0]
            flin.read(4)
            flin.read(312)
            for _ in range(nlin):
                sp = {}
                sp['sig'], sp['xint'], sp['width'], sp['dmping'], sp['itn'], sp['ihold'] = unpack("dfffhh", flin.read(24))
                sp['tags'] = flin.read(4)
                sp['epstot'], sp['epsevn'], sp['epsodd'], sp['epsran'], sp['spare'] = unpack("fffff", flin.read(20))
                sp['ident'] = flin.read(32).decode('utf-8', errors='ignore').strip('\x00')
                linel.append(sp)
        except Exception as e:
            print(f"Error reading linelist file {lin_filepath}: {e}")
            raise IOError(f"Failed to parse binary .lin file: {e}")
    return linel

def import_binary_linelist(h5_filepath, lin_filepath, target_spectrum_group):
    print(f"--- Importing binary linelist into spectrum: {target_spectrum_group} ---")
    linelist_data = read_linelist(lin_filepath)
    if not linelist_data:
        print("Warning: Linelist file was empty or failed to parse.")
        return
    df = pd.DataFrame(linelist_data)
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
    print(f"--- Importing calibrated text linelist into: {target_spectrum_group} ---")
    header_metadata = {}
    with open(txt_filepath, 'r') as f:
        all_lines = f.readlines()
    line_iterator = iter(all_lines)
    line1 = next(line_iterator, '')
    if "wavcorr" in line1:
        match = re.search(r'wavcorr\s*=\s*([0-9.E+-]+)', line1)
        if match: header_metadata['wavcorr'] = float(match.group(1))
    line2 = next(line_iterator, '')
    header_metadata['air_correction'] = "applied" if "applied" in line2 else "not applied"
    line3 = next(line_iterator, '')
    header_metadata['intensity_calibration'] = "applied" if "APPLIED" in line3 else "not applied"
    parsed_rows = []
    for line in line_iterator:
        print(line)
        parts = line.strip().split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) < 14:
            print(f"Warning: Skipping malformed data line: {line.strip()}")
            continue
        try:
            cleaned_parts = [p if '*' not in p else 'nan' for p in parts]
            row = {
                'line': int(cleaned_parts[0]),
                'wavenumber': float(cleaned_parts[1]),
                'peak': float(cleaned_parts[2]),
                'width': float(cleaned_parts[3]),
                'dmp': float(cleaned_parts[4]),
                'eq_width': float(cleaned_parts[5]),
                'itn': int(cleaned_parts[6]),
                'H': int(cleaned_parts[7]),
                'tags': cleaned_parts[8],
                'epstot': float(cleaned_parts[9]),
                'epsevn': float(cleaned_parts[10]),
                'epsodd': float(cleaned_parts[11]),
                'epsran': float(cleaned_parts[12]),
                'identification': " ".join(cleaned_parts[13:-1]),
                'wavelength_air': float(cleaned_parts[-1])
            }

            parsed_rows.append(row)
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse data line: {line.strip()} | Error: {e}")
            continue
    if not parsed_rows:
        raise ValueError("No valid data lines could be parsed from the file.")
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
    print(f"--- Importing lamp calibration file: {cal_filepath} ---")
    comments = []
    data_lines = []
    with open(cal_filepath, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line: continue
            if stripped_line.startswith('#'):
                comments.append(stripped_line.lstrip('#').strip())
            else:
                data_lines.append(stripped_line)
    if not data_lines:
        raise ValueError("No valid data lines found in the calibration file.")
    from io import StringIO
    data_io = StringIO("\n".join(data_lines))
    df = pd.read_csv(data_io, sep=r'\s+', header=None, names=['wavelength_nm', 'spectral_radiance'])
    full_metadata = user_metadata.copy()
    full_metadata['header_comments'] = "\n".join(comments)
    full_metadata['original_filename'] = os.path.basename(cal_filepath)
    table_name = os.path.splitext(os.path.basename(cal_filepath))[0].replace('.', '_').replace('-', '_')
    group_path = "/Standard_Lamp_Calibrations"
    h5_manager.add_pandas_table(h5_filepath, group_path, table_name, df, metadata_dict=full_metadata)

def parse_generic_text_file(filepath, file_type='delimited', delimiter=',', has_header=False, col_widths=None):
    header_row = 0 if has_header else None
    try:
        if file_type == 'delimited':
            if delimiter == 'space':
                delimiter_regex = r'\s+'
                df = pd.read_csv(filepath, sep=delimiter_regex, header=header_row, engine='python')
            else:
                df = pd.read_csv(filepath, sep=delimiter, header=header_row, engine='python')
        elif file_type == 'fixed':
            df = pd.read_fwf(filepath, widths=col_widths, header=header_row)
        else:
            print(f"Error: Unknown file type '{file_type}'")
            return pd.DataFrame()
        if not has_header:
            df.columns = [f'Column {i+1}' for i in range(len(df.columns))]
        return df
    except Exception as e:
        print(f"Error parsing file: {e}")
        return pd.DataFrame()
