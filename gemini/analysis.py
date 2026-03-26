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


        # --- NEW: Exclude lines by forcing their weight to 0 ---
        excluded_col = f'{name}\nExcluded'
        if excluded_col in df_out.columns:
            excluded_mask = df_out[excluded_col].fillna(False).astype(bool)
            weights = weights.mask(excluded_mask, 0.0)

        weights = weights.fillna(0)
        
        # Apply the maximum weight cap to account for systematic uncertainty.
        weights = weights.clip(upper=max_weight)
#        print(name,intensities,weights)

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
                                  energy_levels_df: pd.DataFrame,
                                  calculations_df: pd.DataFrame = None,
                                  wavenumber_tolerance: float = 0.02) -> pd.DataFrame:
    """
    Calculates branching fractions, estimates an unobserved residual from theoretical 
    transition probabilities, and calculates uncertainties according to Sikström et al.
    """
    if lines_for_calculation.empty: return pd.DataFrame()
    df = lines_for_calculation.copy()
    if 'Mean Intensity' not in df.columns: df = add_weighted_averages(df)
        
    df.dropna(subset=['Mean Intensity'], inplace=True)
    if df.empty: return pd.DataFrame()

    # --- Forcefully clean the target key of any asterisks or spaces ---
    clean_target_key = str(upper_level_key).replace('*', '').strip()

    # 1. Fetch the Lifetime for the upper level
    lifetime = 0.0
    if not energy_levels_df.empty and 'key' in energy_levels_df.columns and 'lifetime' in energy_levels_df.columns:
        energy_levels_df['key_clean'] = energy_levels_df['key'].astype(str).str.replace('*', '', regex=False).str.strip()
        matches = energy_levels_df[energy_levels_df['key_clean'] == clean_target_key]
        if not matches.empty:
            try:
                lifetime = float(matches.iloc[0]['lifetime'])
            except ValueError:
                pass

    print(f"\n--- Residual Calculation for Level: '{clean_target_key}' ---")
    print(f"Lifetime: {lifetime} ns")

    # 2. Calculate the Residual from Theoretical Calculations
    frac_resid = 0.0
    unobserved_A_sum = 0.0
    matched_theo_A = {}
    
    if calculations_df is not None and not calculations_df.empty and 'upper_level_key' in calculations_df.columns:
        # Sanitize keys for consistent matching (removing asterisks and whitespace)
        calculations_df['upper_level_key_clean'] = calculations_df['upper_level_key'].astype(str).str.replace('*', '', regex=False).str.strip()
        theo_lines = calculations_df[calculations_df['upper_level_key_clean'] == clean_target_key].copy()
        
        print(f"Found {len(theo_lines)} theoretical lines for this level in the Calculations file.")

        if not theo_lines.empty and 'wavenumber' in theo_lines.columns and 'transition_probability' in theo_lines.columns:
            theo_lines['wavenumber'] = pd.to_numeric(theo_lines['wavenumber'], errors='coerce')
            theo_lines['transition_probability'] = pd.to_numeric(theo_lines['transition_probability'], errors='coerce')
            theo_lines.dropna(subset=['wavenumber', 'transition_probability'], inplace=True)
            
            observed_wns = pd.to_numeric(df['wavenumber'], errors='coerce').dropna().values
            
            for idx, row in theo_lines.iterrows():
                t_wn = row['wavenumber']
                t_A = row['transition_probability']
                
                # Check if this theoretical line was observed
                diffs = np.abs(observed_wns - t_wn)
                if len(diffs) > 0 and np.min(diffs) <= wavenumber_tolerance:
                    # It's an observed line, store the A-value for display
                    best_match_idx = np.argmin(diffs)
                    matched_wn = observed_wns[best_match_idx]
                    matched_theo_A[matched_wn] = t_A
                else:
                    # Unobserved line: add to residual sum
                    unobserved_A_sum += t_A
                    print(f"  -> Unobserved Residual Line: Wavenumber = {t_wn:>10.3f} cm⁻¹ | Trans. Prob. = {t_A:>10.4f} (10⁶ s⁻¹)")
                    
            if lifetime > 0:
                # Assumes Transition Probability is in 10^6 s^-1 and lifetime in ns
                # (A * 10^6) * (tau * 10^-9) = A * tau / 1000
                frac_resid = unobserved_A_sum * lifetime / 1000.0
                print(f"Total Unobserved A-value sum: {unobserved_A_sum:.4f} (10⁶ s⁻¹)")
                print(f"Calculated Residual Fraction: {frac_resid * 100.0:.3f} %")
            else:
                print("Warning: Lifetime is 0. Cannot calculate residual fraction.")
    else:
        print("Warning: No calculations dataframe provided, or missing 'upper_level_key' column.")
    print("---------------------------------------------------\n")

    # 3. Calculate BF and Uncertainties
    # Excluded lines (or NaN) default to 0.0 intensity for the sum
    valid_intensities = pd.to_numeric(df['Mean Intensity'], errors='coerce').fillna(0.0)
    base_total_int = valid_intensities.sum()
    
    # Scale total intensity to account for the missing residual branches
    total_int = base_total_int * (1.0 + frac_resid)
    
    if total_int > 0:
        df['Branching Fraction'] = valid_intensities / total_int
        
        fractional_unc = pd.to_numeric(df['Mean Uncertainty'], errors='coerce').fillna(0.0)
        BF_array = df['Branching Fraction'].values
        
        # Calculate the BFsq term (Sum of BF_k^2 * rel_var_k)
        BFsq = np.sum( (BF_array ** 2) * (fractional_unc ** 2) )
        
        # Add a 50% relative uncertainty estimate for the residual correction itself
        BFsq += (frac_resid ** 2) * 0.25 
        
        df['BF Uncertainty (%)'] = 0.0
        df['Trans. Prob. (10^6 s^-1)'] = 0.0
        df['Trans. Prob. Unc. (%)'] = 0.0
        df['Theoretical Trans. Prob.'] = np.nan
        
        life_unc_frac = 0.0 # Assuming 0 for now unless added to energy levels schema
        
        for index, row in df.iterrows():
            BF = row['Branching Fraction']
            delta_I = fractional_unc.get(index, 0.0)
            
            # Sikström Eq 7 for variance
            rel_var_BF = (delta_I ** 2) * (1.0 - 2.0 * BF) + BFsq
            rel_var_BF = max(rel_var_BF, 0.0) # Prevent float rounding errors below 0
            
            bf_unc_frac = np.sqrt(rel_var_BF)
            df.at[index, 'BF Uncertainty (%)'] = bf_unc_frac * 100.0
            
            if lifetime > 0:
                A_val = (1000.0 * BF) / lifetime
                unc_A_frac = np.sqrt(rel_var_BF + life_unc_frac**2)
                df.at[index, 'Trans. Prob. (10^6 s^-1)'] = A_val
                df.at[index, 'Trans. Prob. Unc. (%)'] = unc_A_frac * 100.0
                
            # Match back the theoretical A-value for the table display
            wn = pd.to_numeric(row['wavenumber'], errors='coerce')
            if not pd.isna(wn):
                closest_theo = None
                min_diff = 1e9
                for t_wn, t_A in matched_theo_A.items():
                    if abs(t_wn - wn) < min_diff and abs(t_wn - wn) <= wavenumber_tolerance:
                        min_diff = abs(t_wn - wn)
                        closest_theo = t_A
                if closest_theo is not None:
                    df.at[index, 'Theoretical Trans. Prob.'] = closest_theo
    else:
        df['Branching Fraction'] = 0.0

    # Format final columns and attach metadata
    result_cols =['wavenumber', 'lower_level_key', 'Mean Intensity', 'Mean Uncertainty', 
                   'Branching Fraction', 'BF Uncertainty (%)', 'Trans. Prob. (10^6 s^-1)', 
                   'Trans. Prob. Unc. (%)', 'Theoretical Trans. Prob.']
                   
    results = df[[col for col in result_cols if col in df.columns]].copy()
    
    # Store metadata in the DataFrame so the Results Dialog can display it
    results.attrs['residual_fraction'] = frac_resid
    results.attrs['lifetime'] = lifetime
    
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

def transfer_calibration(df: pd.DataFrame, transfer_line_index: int, target_spectrum: str) -> pd.DataFrame:
    """
    Transfers the intensity calibration to a target spectrum that lacks the primary reference line.
    
    This calculates the weighted average intensity of the selected 'transfer line' across all 
    OTHER spectra. It then scales the entire target spectrum so its transfer line intensity 
    matches this average, and correctly propagates the added uncertainty to all lines in the 
    target spectrum.
    """
    if df.empty or not (0 <= transfer_line_index < len(df)):
        return df
        
    df_out = df.copy()
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df_out.columns if '\n' in col])))
    
    if target_spectrum not in spectrum_names:
        raise ValueError(f"Target spectrum '{target_spectrum}' not found.")
        
    # Get the DataFrame index label for the selected row
    transfer_label = df_out.index[transfer_line_index]
    
    sum_W = 0.0
    sum_IW = 0.0
    
    # 1. Calculate weighted average from ALL OTHER spectra
    for spec in spectrum_names:
        if spec == target_spectrum:
            continue

        # --- NEW: Ignore excluded lines for transfer calibration ---
        excluded_col = f'{spec}\nExcluded'
        if excluded_col in df_out.columns and df_out.at[transfer_label, excluded_col] == True:
            continue
        # ---------------------------------------------------------
        
        i_col = f'{spec}\nIntensity'
        snr_col = f'{spec}\nSNR'
        
        if i_col not in df_out.columns or snr_col not in df_out.columns:
            continue
            
        I_S = pd.to_numeric(df_out.at[transfer_label, i_col], errors='coerce')
        SNR_S = pd.to_numeric(df_out.at[transfer_label, snr_col], errors='coerce')
        
        # If the other spectrum has a valid measurement for this line
        if pd.notna(I_S) and pd.notna(SNR_S) and I_S > 0 and SNR_S > 0:
            W_S = min(SNR_S**2, 555)  # Cap max weight (1 / 0.06^2/2)
            sum_W += W_S
            sum_IW += I_S * W_S
            
    if sum_W == 0:
        raise ValueError("No valid data in other spectra to compute a weighted average for this transfer line.")
        
    avg_I = sum_IW / sum_W
    
    # 2. Extract target spectrum transfer line properties
    target_i_col = f'{target_spectrum}\nIntensity'
    target_snr_col = f'{target_spectrum}\nSNR'
    
    I_target = pd.to_numeric(df_out.at[transfer_label, target_i_col], errors='coerce')
    SNR_target = pd.to_numeric(df_out.at[transfer_label, target_snr_col], errors='coerce')
    
    if pd.isna(I_target) or I_target <= 0 or pd.isna(SNR_target) or SNR_target <= 0:
        raise ValueError(f"The target spectrum '{target_spectrum}' does not have a valid measurement for the selected transfer line.")
        
    # 3. Calculate scaling factor and new transfer uncertainty
    scale = avg_I / I_target
    unc_renorm = ( (SNR_target**2) + sum_W )**(-0.5)
    
    # 4. Apply to the target spectrum
    for r_label in df_out.index:
        I_old = pd.to_numeric(df_out.at[r_label, target_i_col], errors='coerce')
        SNR_old = pd.to_numeric(df_out.at[r_label, target_snr_col], errors='coerce')
        
        if pd.notna(I_old) and I_old > 0:
            df_out.at[r_label, target_i_col] = I_old * scale
            
        if pd.notna(SNR_old) and SNR_old > 0:
            # If it's the transfer level, SNR = 1/unc_renorm
            if r_label == transfer_label:
                df_out.at[r_label, target_snr_col] = 1.0 / unc_renorm
            # Otherwise add uncertainties to unc of ref level
            else:
                U_old = 1.0 / SNR_old
                U_new = (U_old**2 + unc_renorm**2)**0.5
                df_out.at[r_label, target_snr_col] = 1.0 / U_new
                
    return df_out