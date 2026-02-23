# analysis.py (FULLY DOCUMENTED)

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
    Aggregates experimental data from multiple linelists and merges it with a
    base DataFrame of previously identified lines.

    This function is designed to be robust against several common issues:
    1.  **Duplicate Columns:** It prevents crashes when merging multiple spectra
        by creating a single, temporary numeric key for merging and renaming
        the merge key on incoming dataframes to match.
    2.  **Data Type Preservation:** It explicitly preserves the original string
        formatting of the 'wavenumber' and 'intensity' columns from the
        `previous_ids_df` for accurate display, while using a temporary
        numeric version of the wavenumber for calculations.

    Args:
        h5_filepath: The path to the main HDF5 project file.
        previous_ids_df: A DataFrame containing the master list of lines from an
                         upper level, including their 'wavenumber' and 'intensity'.
        linelist_paths: A list of HDF5 paths pointing to the experimental
                        linelist tables to be merged.
        tolerance: The maximum distance in cm⁻¹ for a nearest-neighbor merge
                   between the master list and an experimental line.

    Returns:
        A single DataFrame with columns for the base line data, followed by
        paired 'Intensity' and 'SNR' columns for each successfully merged spectrum.
    """
    if previous_ids_df.empty:
        return pd.DataFrame()

    final_df = previous_ids_df.copy()
    
    # Preserve original string formatting for display by converting to string type.
    # A temporary numeric column is created for the sole purpose of merging.
    if 'wavenumber' in final_df.columns:
        final_df['wavenumber'] = final_df['wavenumber'].astype(str)
        final_df['wavenumber_numeric_merge_key'] = pd.to_numeric(final_df['wavenumber'], errors='coerce')
    else:
        # Merging is impossible without a wavenumber key.
        return final_df
        
    if 'intensity' in final_df.columns:
        final_df['intensity'] = final_df['intensity'].astype(str)

    # Loop through each provided experimental linelist path.
    if linelist_paths:
        for path in linelist_paths:
            try:
                # Extract a short name for the spectrum from its HDF5 path.
                spectrum_name = path.split('/')[2]
                exp_df = h5_manager.read_hdf_table_robustly(h5_filepath, path)
                exp_df.drop(columns=['index'], inplace=True, errors='ignore')

                if 'wavenumber' not in exp_df.columns:
                    continue
                
                # Rename experimental columns to be specific to this spectrum.
                rename_dict = {
                    'peak': f'{spectrum_name}\nSNR',
                    'eq_width': f'{spectrum_name}\nIntensity'
                }
                cols_to_keep = ['wavenumber', 'peak', 'eq_width']
                exp_df_subset = exp_df[[col for col in cols_to_keep if col in exp_df.columns]].copy()
                exp_df_subset.rename(columns=rename_dict, inplace=True)
                exp_df_subset['wavenumber'] = pd.to_numeric(exp_df_subset['wavenumber'], errors='coerce')
                
                # --- Robust Merging Strategy ---
                final_df.sort_values('wavenumber_numeric_merge_key', inplace=True)
                
                # Rename the key in the incoming (right) table to exactly match the key
                # in the main (left) table. This is the key to preventing pandas from
                # creating ambiguous '_x' and '_y' columns.
                exp_df_subset.rename(columns={'wavenumber': 'wavenumber_numeric_merge_key'}, inplace=True)
                exp_df_subset.sort_values('wavenumber_numeric_merge_key', inplace=True)
                
                # Perform the 'as-of' merge on the single, unambiguous key.
                final_df = pd.merge_asof(
                    final_df,
                    exp_df_subset,
                    on='wavenumber_numeric_merge_key',
                    direction='nearest',
                    tolerance=tolerance
                )

            except Exception as e:
                print(f"Warning: Could not process or merge linelist from {path}: {e}")
    
    # Clean up by removing the temporary merge key.
    final_df.drop(columns=['wavenumber_numeric_merge_key'], inplace=True, errors='ignore')

    # Define the desired final column order.
    base_cols = ['wavenumber', 'lower_level_key', 'intensity']
    spectrum_cols = [col for col in final_df.columns if '\n' in str(col)]
    spectrum_cols.sort(key=lambda name: (name.split('\n')[0], name.split('\n')[1]))
    final_order = base_cols + spectrum_cols
    existing_cols_in_order = [col for col in final_order if col in final_df.columns]
    
    return final_df[existing_cols_in_order]


def add_weighted_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the weighted average intensity and its fractional uncertainty.

    This function implements a specific weighting scheme to account for both
    statistical and systematic uncertainties in spectroscopic measurements.

    Algorithm:
    1.  For each measurement, the statistical weight is calculated from the
        Signal-to-Noise Ratio (SNR) as: `Weight = SNR²`. This is equivalent
        to `1 / (fractional_uncertainty)²`.
    2.  A **maximum weight cap** of 555 is applied. This prevents single,
        extremely high-SNR lines from dominating the average. It acts as a
        floor for systematic uncertainty, effectively treating all lines with
        SNR > ~23.5 (sqrt(555)) as having the same high quality.
    3.  The weighted average intensity is calculated: `Σ(Intensity * Weight) / Σ(Weight)`.
    4.  The final uncertainty is the **fractional uncertainty** of the weighted
        mean: `1 / sqrt( Σ(Weight) )`.

    Args:
        df: The DataFrame containing aggregated line data with 'Intensity' and 'SNR'
            columns for multiple spectra.

    Returns:
        The input DataFrame with two new columns appended:
        - 'Mean Intensity': The calculated weighted average intensity.
        - 'Mean Uncertainty': The calculated fractional uncertainty of the mean.
    """
    if df.empty:
        return df

    df_out = df.copy()
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df.columns if '\n' in col])))
    
    sum_of_weights = pd.Series(0.0, index=df.index)
    sum_of_val_x_weight = pd.Series(0.0, index=df.index)
    max_weight = 555  # Cap on the statistical weight.

    for name in spectrum_names:
        intensity_col = f'{name}\nIntensity'
        snr_col = f'{name}\nSNR'
        
        if intensity_col not in df_out.columns or snr_col not in df_out.columns:
            continue
            
        intensities = pd.to_numeric(df_out[intensity_col], errors='coerce')
        snrs = pd.to_numeric(df_out[snr_col], errors='coerce')
        
        # Weight is based on SNR^2 (the inverse of fractional uncertainty squared).
        weights = snrs ** 2
        weights = weights.fillna(0)
        
        # Apply the maximum weight cap to account for systematic uncertainty.
        weights = weights.clip(upper=max_weight)
        
        sum_of_weights += weights
        sum_of_val_x_weight += intensities.multiply(weights).fillna(0)
        
    # Calculate the final weighted average intensity.
    mean_intensity = sum_of_val_x_weight.divide(sum_of_weights).replace([np.inf, -np.inf], np.nan)
    
    # Calculate the final fractional uncertainty of the weighted mean.
    mean_fractional_uncertainty = (1.0 / np.sqrt(sum_of_weights)).replace([np.inf, -np.inf], np.nan)
    
    df_out['Mean Intensity'] = mean_intensity
    df_out['Mean Uncertainty'] = mean_fractional_uncertainty
    
    return df_out


def match_wavenumbers(experimental_linelist: pd.DataFrame,
                      previous_ids: pd.DataFrame,
                      tolerance: float = 0.02) -> pd.DataFrame:
    """
    Finds matches between an experimental linelist and a list of known lines.

    For each line in the experimental list, this function searches for the closest
    match in the 'previous_ids' list within a given tolerance.

    Args:
        experimental_linelist: DataFrame of observed lines. Must have a 'wavenumber' column.
        previous_ids: DataFrame of known lines. Must have a 'wavenumber' column.
        tolerance: The maximum allowed difference in cm⁻¹ for a match to be considered valid.

    Returns:
        A new DataFrame containing the combined data for all matched lines.
        Experimental columns are suffixed with '_exp' and ID columns with '_id'.
    """
    if experimental_linelist.empty or previous_ids.empty: return pd.DataFrame()
    if 'wavenumber' not in experimental_linelist.columns or 'wavenumber' not in previous_ids.columns:
        raise ValueError("Both input DataFrames must contain a 'wavenumber' column.")
    
    # Ensure wavenumbers are numeric for comparison.
    exp_df, ids_df = experimental_linelist.copy(), previous_ids.copy()
    exp_df['wavenumber'] = pd.to_numeric(exp_df['wavenumber'], errors='coerce')
    ids_df['wavenumber'] = pd.to_numeric(ids_df['wavenumber'], errors='coerce')
    exp_df.dropna(subset=['wavenumber'], inplace=True); ids_df.dropna(subset=['wavenumber'], inplace=True)
    if exp_df.empty or ids_df.empty: return pd.DataFrame()

    matches = []
    id_wavenumbers = ids_df['wavenumber'].values
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
    """
    A wrapper function that performs wavenumber matching and saves the results
    to the HDF5 file.

    Args:
        h5_filepath: Path to the main HDF5 project file.
        exp_path: HDF5 path to the experimental linelist table.
        ids_path: HDF5 path to the previous identifications table.
        tolerance: The wavenumber tolerance in cm⁻¹ for the match.
        output_name: The desired name for the output table in the HDF5 file.

    Returns:
        The number of matches found and saved.
    """
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
    """
    Calculates the branching fraction for each line based on mean intensities.

    Args:
        lines_for_calculation: A DataFrame containing the lines from a single
                               upper level, including a 'Mean Intensity' column.
        upper_level_key: The identifier for the upper energy level.
        energy_levels_df: The master DataFrame of all energy levels, used to
                          potentially include other atomic data in the future.

    Returns:
        A DataFrame containing the final results, including the calculated
        'Branching Fraction' for each line.
    """
    if lines_for_calculation.empty: return pd.DataFrame()
    df = lines_for_calculation.copy()
    if 'Mean Intensity' not in df.columns: df = add_weighted_averages(df)
        
    df.dropna(subset=['Mean Intensity'], inplace=True)
    if df.empty: return pd.DataFrame()
        
    total_intensity = df['Mean Intensity'].sum()
    
    if total_intensity > 0:
        df['Branching Fraction'] = df['Mean Intensity'] / total_intensity
    else:
        df['Branching Fraction'] = 0.0

    result_cols = ['wavenumber', 'lower_level_key', 'Mean Intensity', 'Mean Uncertainty', 'Branching Fraction']
    results = df[[col for col in result_cols if col in df.columns]].copy()
    return results

def normalize_intensities_by_reference_line(
    master_df: pd.DataFrame,
    reference_line_index: int
) -> pd.DataFrame:
    """
    Rescales intensities in each spectrum so a reference line has an intensity of 1000.

    This function iterates through each spectrum's intensity column. For each one,
    it finds the intensity of the specified reference line and calculates a
    rescaling factor to make that intensity equal to 1000. It then applies this
    factor to all lines within that same spectrum.

    Args:
        master_df: The DataFrame containing all line data.
        reference_line_index: The integer row number of the line to be used
                              as the normalization reference.

    Returns:
        A new DataFrame with the normalized intensity values.
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
            print(f"Warning: Reference line has no valid intensity in '{spectrum_name}'. Skipping normalization for this spectrum.")
            
    return normalized_df