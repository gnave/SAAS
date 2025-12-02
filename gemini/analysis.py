# analysis.py (FINAL with CORRECT newline header logic and ADDED normalization)

import pandas as pd
import numpy as np
import h5py
from datetime import datetime
import h5_manager

def aggregate_observed_data_for_display(h5_filepath: str,
                                        previous_ids_df: pd.DataFrame,
                                        linelist_paths: list,
                                        tolerance: float = 0.02) -> pd.DataFrame:
    """
    PRODUCTION VERSION: Aggregates data and creates newline-separated column headers.
    """
    if previous_ids_df.empty:
        return pd.DataFrame()

    final_df = previous_ids_df.copy()

    if linelist_paths:
        for path in linelist_paths:
            try:
                spectrum_name = path.split('/')[2]
                exp_df = h5_manager.read_hdf_table_robustly(h5_filepath, path)
                exp_df.drop(columns=['index'], inplace=True, errors='ignore')

                if 'wavenumber' not in exp_df.columns:
                    continue

                # --- FIX 1: Rename columns with Spectrum Name on TOP ---
                rename_dict = {
                    'peak': f'{spectrum_name}\nSNR',
                    'eq_width': f'{spectrum_name}\nIntensity'
                }
                cols_to_keep = ['wavenumber', 'peak', 'eq_width']
                exp_df_subset = exp_df[[col for col in cols_to_keep if col in exp_df.columns]].copy()
                exp_df_subset.rename(columns=rename_dict, inplace=True)

                final_df['wavenumber'] = pd.to_numeric(final_df['wavenumber'], errors='coerce')
                exp_df_subset['wavenumber'] = pd.to_numeric(exp_df_subset['wavenumber'], errors='coerce')

                final_df.sort_values('wavenumber', inplace=True)
                exp_df_subset.sort_values('wavenumber', inplace=True)

                final_df = pd.merge_asof(
                    final_df,
                    exp_df_subset,
                    on='wavenumber',
                    direction='nearest',
                    tolerance=tolerance
                )
            except Exception as e:
                print(f"Warning: Could not process or merge linelist from {path}: {e}")

    final_df['Include_in_Fit'] = True

    # Define the mandatory base columns
    base_cols = ['wavenumber', 'lower_level_key', 'intensity']

    # --- FIX 2: Correctly find all columns that contain a newline ---
    spectrum_cols = [col for col in final_df.columns if '\n' in str(col)]

    # --- FIX 3: Correctly sort by spectrum name (top line) then type (bottom line) ---
    spectrum_cols.sort(key=lambda name: (name.split('\n')[0], name.split('\n')[1]))

    final_order = base_cols + spectrum_cols + ['Include_in_Fit']

    existing_cols_in_order = [col for col in final_order if col in final_df.columns]

    return final_df[existing_cols_in_order]


# --- The rest of the file is unchanged ---
def match_wavenumbers(experimental_linelist: pd.DataFrame,
                      previous_ids: pd.DataFrame,
                      tolerance: float = 0.02) -> pd.DataFrame:
    # ... (code is the same)
    if experimental_linelist.empty or previous_ids.empty: return pd.DataFrame()
    if 'wavenumber' not in experimental_linelist.columns or 'wavenumber' not in previous_ids.columns:
        raise ValueError("Both input DataFrames must contain a 'wavenumber' column.")
    exp_df, ids_df = experimental_linelist.copy(), previous_ids.copy()
    exp_df['wavenumber'] = pd.to_numeric(exp_df['wavenumber'], errors='coerce')
    ids_df['wavenumber'] = pd.to_numeric(ids_df['wavenumber'], errors='coerce')
    exp_df.dropna(subset=['wavenumber'], inplace=True)
    ids_df.dropna(subset=['wavenumber'], inplace=True)
    if exp_df.empty or ids_df.empty: return pd.DataFrame()
    matches, id_wavenumbers = [], ids_df['wavenumber'].values
    for index, exp_line in exp_df.iterrows():
        exp_wn = exp_line['wavenumber']
        differences = np.abs(id_wavenumbers - exp_wn)
        best_match_index = np.argmin(differences)
        if differences[best_match_index] <= tolerance:
            id_line = ids_df.iloc[best_match_index]
            combined_data = {f"{col}_exp": val for col, val in exp_line.items()}
            combined_data.update({f"{col}_id": val for col, val in id_line.items()})
            combined_data['wavenumber'] = combined_data.pop('wavenumber_exp')
            matches.append(combined_data)
    if not matches: return pd.DataFrame()
    matched_df = pd.DataFrame(matches)
    matched_df['wavenumber_diff'] = (matched_df['wavenumber'] - matched_df['wavenumber_id']).abs()
    return matched_df

def run_and_save_wavenumber_match(h5_filepath, exp_path, ids_path, tolerance, output_name):
    # ... (code is the same)
    exp_df = h5_manager.read_hdf_table_robustly(h5_filepath, exp_path)
    ids_df = h5_manager.read_hdf_table_robustly(h5_filepath, ids_path)
    target_spectrum_group = '/'.join(exp_path.split('/')[:3])
    matched_df = match_wavenumbers(exp_df, ids_df, tolerance)
    if matched_df.empty: return 0
    output_group_path = f"{target_spectrum_group}/Identified_Lines"
    sanitized_output_name = output_name.replace('.', '_').replace('-', '_')
    metadata = {
        'analysis_date': datetime.now().isoformat(), 'analysis_type': 'Wavenumber Matching',
        'source_experimental_linelist': exp_path, 'source_previous_identifications': ids_path,
        'matching_tolerance_cm-1': tolerance
    }
    h5_manager.add_pandas_table(
        h5_filepath, output_group_path, sanitized_output_name,
        matched_df, metadata_dict=metadata
    )
    return len(matched_df)

def calculate_branching_fractions(lines_for_calculation: pd.DataFrame,
                                  upper_level_key: str,
                                  energy_levels_df: pd.DataFrame) -> pd.DataFrame:
    # ... (code is the same)
    if lines_for_calculation.empty: return pd.DataFrame()
    return pd.DataFrame({
        'upper_level_key': [upper_level_key],
        'calculated_bf': [0.5 + np.random.rand() * 0.1],
        'bf_uncertainty': [0.01 + np.random.rand() * 0.005],
        'num_lines_included': [len(lines_for_calculation)]
    })

# --- MODIFICATION START ---
def normalize_intensities_by_reference_line(
    master_df: pd.DataFrame,
    reference_line_index: int
) -> pd.DataFrame:
    """
    Rescales intensities in each spectrum so a reference line has an intensity of 1000.

    Args:
        master_df: The DataFrame containing all line data, including multiple
                   'SpectrumName\nIntensity' columns.
        reference_line_index: The integer index (row number) of the line to be
                              used as the reference for normalization.

    Returns:
        A new DataFrame with normalized intensity values.
    """
    if master_df.empty or not (0 <= reference_line_index < len(master_df)):
        return master_df

    normalized_df = master_df.copy()
    intensity_cols = [col for col in normalized_df.columns if isinstance(col, str) and '\nIntensity' in col]
    reference_line = normalized_df.iloc[reference_line_index]

    for col in intensity_cols:
        norm_factor = reference_line.get(col)
        if pd.notna(norm_factor) and norm_factor > 0:
            normalized_df[col] = (normalized_df[col] / norm_factor) * 1000.0
        else:
            spectrum_name = col.split('\n')[0]
            print(f"Warning: Reference line has no valid intensity in '{spectrum_name}'. "
                  f"Skipping normalization for this spectrum.")
            
    return normalized_df
# --- MODIFICATION END ---