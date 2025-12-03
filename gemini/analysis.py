# analysis.py (DEFINITIVE FIX for uncertainty calculation)

import pandas as pd
import numpy as np
import h5py
from datetime import datetime
import h5_manager
import math

def aggregate_observed_data_for_display(h5_filepath: str,
                                        previous_ids_df: pd.DataFrame,
                                        linelist_paths: list,
                                        tolerance: float = 0.02) -> pd.DataFrame:
    """
    DEFINITIVE FIX: Aggregates data robustly, preventing duplicate column errors,
    and preserves original formatting for base 'wavenumber' and 'intensity'.
    """
    if previous_ids_df.empty:
        return pd.DataFrame()

    final_df = previous_ids_df.copy()
    
    if 'wavenumber' in final_df.columns:
        final_df['wavenumber'] = final_df['wavenumber'].astype(str)
        final_df['wavenumber_numeric_merge_key'] = pd.to_numeric(final_df['wavenumber'], errors='coerce')
    else:
        return final_df
        
    if 'intensity' in final_df.columns:
        final_df['intensity'] = final_df['intensity'].astype(str)

    if linelist_paths:
        for path in linelist_paths:
            try:
                spectrum_name = path.split('/')[2]
                exp_df = h5_manager.read_hdf_table_robustly(h5_filepath, path)
                exp_df.drop(columns=['index'], inplace=True, errors='ignore')

                if 'wavenumber' not in exp_df.columns:
                    continue
                
                rename_dict = {
                    'peak': f'{spectrum_name}\nSNR',
                    'eq_width': f'{spectrum_name}\nIntensity'
                }
                cols_to_keep = ['wavenumber', 'peak', 'eq_width']
                exp_df_subset = exp_df[[col for col in cols_to_keep if col in exp_df.columns]].copy()
                exp_df_subset.rename(columns=rename_dict, inplace=True)
                exp_df_subset['wavenumber'] = pd.to_numeric(exp_df_subset['wavenumber'], errors='coerce')
                
                final_df.sort_values('wavenumber_numeric_merge_key', inplace=True)
                exp_df_subset.rename(columns={'wavenumber': 'wavenumber_numeric_merge_key'}, inplace=True)
                exp_df_subset.sort_values('wavenumber_numeric_merge_key', inplace=True)
                
                final_df = pd.merge_asof(
                    final_df,
                    exp_df_subset,
                    on='wavenumber_numeric_merge_key',
                    direction='nearest',
                    tolerance=tolerance
                )

            except Exception as e:
                print(f"Warning: Could not process or merge linelist from {path}: {e}")
    
    final_df.drop(columns=['wavenumber_numeric_merge_key'], inplace=True, errors='ignore')

    base_cols = ['wavenumber', 'lower_level_key', 'intensity']
    spectrum_cols = [col for col in final_df.columns if '\n' in str(col)]
    spectrum_cols.sort(key=lambda name: (name.split('\n')[0], name.split('\n')[1]))
    final_order = base_cols + spectrum_cols
    existing_cols_in_order = [col for col in final_order if col in final_df.columns]
    
    return final_df[existing_cols_in_order]


# --- MODIFICATION START: Corrected uncertainty logic ---
def add_weighted_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the weighted average intensity and its FRACTIONAL uncertainty,
    applying a cap on the maximum statistical weight based on SNR.
    """
    if df.empty:
        return df

    df_out = df.copy()
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df.columns if '\n' in col])))
    
    sum_of_weights = pd.Series(0.0, index=df.index)
    sum_of_val_x_weight = pd.Series(0.0, index=df.index)
    max_weight = 555

    for name in spectrum_names:
        intensity_col = f'{name}\nIntensity'
        snr_col = f'{name}\nSNR'
        
        if intensity_col not in df_out.columns or snr_col not in df_out.columns:
            continue
            
        intensities = pd.to_numeric(df_out[intensity_col], errors='coerce')
        snrs = pd.to_numeric(df_out[snr_col], errors='coerce')
        
        # Weight is based on SNR^2 (inverse of fractional uncertainty squared)
        weights = snrs ** 2
        weights = weights.fillna(0)
        
        # Apply the maximum weight cap
        weights = weights.clip(upper=max_weight)
        
        sum_of_weights += weights
        sum_of_val_x_weight += intensities.multiply(weights).fillna(0)
        
    mean_intensity = sum_of_val_x_weight.divide(sum_of_weights).replace([np.inf, -np.inf], np.nan)
    
    # This is now the FRACTIONAL uncertainty of the mean
    mean_fractional_uncertainty = (1.0 / np.sqrt(sum_of_weights)).replace([np.inf, -np.inf], np.nan)
    
    df_out['Mean Intensity'] = mean_intensity
    df_out['Mean Uncertainty'] = mean_fractional_uncertainty
    
    return df_out
# --- MODIFICATION END ---


def match_wavenumbers(experimental_linelist: pd.DataFrame,
                      previous_ids: pd.DataFrame,
                      tolerance: float = 0.02) -> pd.DataFrame:
    if experimental_linelist.empty or previous_ids.empty: return pd.DataFrame()
    if 'wavenumber' not in experimental_linelist.columns or 'wavenumber' not in previous_ids.columns:
        raise ValueError("Both input DataFrames must contain a 'wavenumber' column.")
    exp_df, ids_df = experimental_linelist.copy(), previous_ids.copy()
    exp_df['wavenumber'] = pd.to_numeric(exp_df['wavenumber'], errors='coerce')
    ids_df['wavenumber'] = pd.to_numeric(ids_df['wavenumber'], errors='coerce')
    exp_df.dropna(subset=['wavenumber'], inplace=True); ids_df.dropna(subset=['wavenumber'], inplace=True)
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
    exp_df = h5_manager.read_hdf_table_robustly(h5_filepath, exp_path)
    ids_df = h5_manager.read_hdf_table_robustly(h5_filepath, ids_path)
    target_spectrum_group = '/'.join(exp_path.split('/')[:3])
    matched_df = match_wavenumbers(exp_df, ids_df, tolerance)
    if matched_df.empty: return 0
    output_group_path = f"{target_spectrum_group}/Identified_Lines"
    sanitized_output_name = output_name.replace('.', '_').replace('-', '_')
    metadata = {'analysis_date': datetime.now().isoformat(), 'analysis_type': 'Wavenumber Matching', 'source_experimental_linelist': exp_path, 'source_previous_identifications': ids_path, 'matching_tolerance_cm-1': tolerance}
    h5_manager.add_pandas_table(h5_filepath, output_group_path, sanitized_output_name, matched_df, metadata_dict=metadata)
    return len(matched_df)

def calculate_branching_fractions(lines_for_calculation: pd.DataFrame, 
                                  upper_level_key: str,
                                  energy_levels_df: pd.DataFrame) -> pd.DataFrame:
    if lines_for_calculation.empty: return pd.DataFrame()
    df = lines_for_calculation.copy()
    if 'Mean Intensity' not in df.columns: df = add_weighted_averages(df)
    df.dropna(subset=['Mean Intensity'], inplace=True)
    if df.empty: return pd.DataFrame()
    total_intensity = df['Mean Intensity'].sum()
    if total_intensity > 0: df['Branching Fraction'] = df['Mean Intensity'] / total_intensity
    else: df['Branching Fraction'] = 0.0
    result_cols = ['wavenumber', 'lower_level_key', 'Mean Intensity', 'Mean Uncertainty', 'Branching Fraction']
    results = df[[col for col in result_cols if col in df.columns]].copy()
    return results

def normalize_intensities_by_reference_line(
    master_df: pd.DataFrame,
    reference_line_index: int
) -> pd.DataFrame:
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
            print(f"Warning: Reference line has no valid intensity in '{spectrum_name}'. Skipping normalization for this spectrum.")
    return normalized_df