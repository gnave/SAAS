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
                                        tolerance: float = 0.1) -> pd.DataFrame:
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
                
                # Extract width alongside peak and eq_width for legacy variance calculations
                rename_dict = {
                    'peak': f'{spectrum_name}\nSNR',
                    'eq_width': f'{spectrum_name}\nIntensity',
                    'width': f'{spectrum_name}\nWidth'
                }
                cols_to_keep =['wavenumber', 'peak', 'eq_width', 'width']
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

    base_cols =['wavenumber', 'lower_level_key', 'intensity']
    spectrum_cols =[col for col in final_df.columns if '\n' in str(col)]
    spectrum_cols.sort(key=lambda name: (name.split('\n')[0], name.split('\n')[1]))
    final_order = base_cols + spectrum_cols
    existing_cols_in_order =[col for col in final_order if col in final_df.columns]
    
    return final_df[existing_cols_in_order]


def add_weighted_averages(df: pd.DataFrame, h5_filepath: str = None) -> pd.DataFrame:
    """
    Calculates the weighted average intensity and its fractional uncertainty
    using the legacy dynamic calibration and FWHM-based variance algorithms.
    """
    if df.empty: return df
    df_out = df.copy()
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df.columns if '\n' in col])))
    
    sum_of_weights = pd.Series(0.0, index=df.index)
    sum_of_val_x_weight = pd.Series(0.0, index=df.index)
    max_weight = 555.0

    for name in spectrum_names:
        intensity_col = f'{name}\nIntensity'
        snr_col = f'{name}\nSNR'
        width_col = f'{name}\nWidth'
        excluded_col = f'{name}\nExcluded'
        
        if intensity_col not in df_out.columns or snr_col not in df_out.columns:
            continue
            
        intensities = pd.to_numeric(df_out[intensity_col], errors='coerce')
        snrs = pd.to_numeric(df_out[snr_col], errors='coerce')
        widths = pd.to_numeric(df_out.get(width_col, pd.Series(np.nan, index=df.index)), errors='coerce')
        wavenumbers = pd.to_numeric(df_out['wavenumber'], errors='coerce')
        
        # Determine the strongest valid line in this spectrum (w_maxI)
        valid_mask = intensities.notna() & snrs.notna()
        if excluded_col in df_out.columns:
            valid_mask &= ~df_out[excluded_col].fillna(False).astype(bool)
            
        if not valid_mask.any(): continue
        
        idx_max_I = intensities[valid_mask].idxmax()
        w_maxI = wavenumbers.loc[idx_max_I]
        
        # Fetch spectrum header parameters (Updated to resolutn, bandlo, bandhi)
        resolutn, bandlo, bandhi = 0.05, 0.0, 30000.0
        if h5_filepath:
            try:
                with h5py.File(h5_filepath, 'r') as f:
                    spec_path = f"/Spectra/{name}/Raw_Data/spectrum"
                    if spec_path in f:
                        attrs = f[spec_path].attrs
                        resolutn = float(attrs.get('resolutn', 0.05))
                        bandlo = float(attrs.get('bandlo', attrs.get('wstart', 0.0)))
                        bandhi = float(attrs.get('bandhi', attrs.get('wend', bandlo + 30000.0)))
                        print(spec_path,resolutn,bandlo,bandhi)
                        if bandhi <= bandlo: bandhi = bandlo + 30000.0
            except Exception: pass
            
        calunc_per_1000 = 70.0 / (bandhi - bandlo) if bandhi > bandlo else 0.0
        
        # root_npts (Width in mK converted to cm-1 divided by resolutn)
        root_npts = (widths / 1000.0) / resolutn
        root_npts = root_npts.fillna(1.0).replace(0, 1.0)
        
        # Variance = 2.25 / (snr^2 * root_npts)
        snr_sq = snrs**2
        snr_sq = snr_sq.replace(0, np.nan)
        variance = 2.25 / (snr_sq * root_npts)
        
        # Calibration Uncertainty
        calunc = calunc_per_1000 * (wavenumbers - w_maxI).abs() / 1000.0
        
        # Final combined weight
        total_variance = (calunc**2) + variance
        weights = 1.0 / total_variance
        weights = weights.fillna(0.0)
        
        if excluded_col in df_out.columns:
            excluded_mask = df_out[excluded_col].fillna(False).astype(bool)
            weights = weights.mask(excluded_mask, 0.0)
            
        weights = weights.clip(upper=max_weight)
        
        sum_of_weights += weights
        sum_of_val_x_weight += intensities.multiply(weights).fillna(0)
        
    mean_intensity = sum_of_val_x_weight.divide(sum_of_weights).replace([np.inf, -np.inf], np.nan)
    mean_fractional_uncertainty = (1.0 / np.sqrt(sum_of_weights)).replace([np.inf, -np.inf], np.nan)
    
    df_out['Mean Intensity'] = mean_intensity
    df_out['Mean Uncertainty'] = mean_fractional_uncertainty
    
    return df_out


def transfer_calibration(df: pd.DataFrame, transfer_line_index: int, target_spectrum: str, h5_filepath: str = None) -> pd.DataFrame:
    """
    Transfers the intensity calibration to a target spectrum that lacks the primary reference line,
    using the legacy FWHM-based variance calculations.
    """
    if df.empty or not (0 <= transfer_line_index < len(df)): return df
    df_out = df.copy()
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df_out.columns if '\n' in col])))
    
    if target_spectrum not in spectrum_names: raise ValueError("Target spectrum not found.")
    transfer_label = df_out.index[transfer_line_index]
    
    sum_W, sum_IW = 0.0, 0.0
    
    # 1. Calculate weighted average from ALL OTHER spectra
    for spec in spectrum_names:
        if spec == target_spectrum: continue
        
        excluded_col = f'{spec}\nExcluded'
        if excluded_col in df_out.columns and df_out.at[transfer_label, excluded_col] == True: continue
            
        i_col, snr_col, width_col = f'{spec}\nIntensity', f'{spec}\nSNR', f'{spec}\nWidth'
        if i_col not in df_out.columns or snr_col not in df_out.columns: continue
            
        I_S = pd.to_numeric(df_out.at[transfer_label, i_col], errors='coerce')
        SNR_S = pd.to_numeric(df_out.at[transfer_label, snr_col], errors='coerce')
        Width_S = pd.to_numeric(df_out.at[transfer_label, width_col], errors='coerce')
        Wnum_S = pd.to_numeric(df_out.at[transfer_label, 'wavenumber'], errors='coerce')
        
        if pd.notna(I_S) and pd.notna(SNR_S) and I_S > 0 and SNR_S > 0:
            intensities = pd.to_numeric(df_out[i_col], errors='coerce')
            snrs = pd.to_numeric(df_out[snr_col], errors='coerce')
            wavenumbers = pd.to_numeric(df_out['wavenumber'], errors='coerce')
            
            valid_mask = intensities.notna() & snrs.notna()
            if excluded_col in df_out.columns:
                valid_mask &= ~df_out[excluded_col].fillna(False).astype(bool)
                
            w_maxI = wavenumbers.loc[intensities[valid_mask].idxmax()] if valid_mask.any() else Wnum_S
                
            # Updated to resolutn, bandlo, bandhi
            resolutn, bandlo, bandhi = 0.05, 0.0, 30000.0
            if h5_filepath:
                try:
                    with h5py.File(h5_filepath, 'r') as f:
                        spec_path = f"/Spectra/{spec}/Raw_Data/spectrum"
                        if spec_path in f:
                            attrs = f[spec_path].attrs
                            resolutn = float(attrs.get('resolutn', 0.05))
                            bandlo = float(attrs.get('bandlo', attrs.get('wstart', 0.0)))
                            bandhi = float(attrs.get('bandhi', attrs.get('wend', bandlo + 30000.0)))
                            if bandhi <= bandlo: bandhi = bandlo + 30000.0
                except Exception: pass
                
            calunc_per_1000 = 70.0 / (bandhi - bandlo) if bandhi > bandlo else 0.0
            root_npts = (Width_S / 1000.0) / resolutn if pd.notna(Width_S) else 1.0
            if root_npts <= 0: root_npts = 1.0
            
            variance = 2.25 / ((SNR_S**2) * root_npts)
            calunc = calunc_per_1000 * abs(Wnum_S - w_maxI) / 1000.0
            
            W_S = min(1.0 / (calunc**2 + variance), 555.0)
            sum_W += W_S
            sum_IW += I_S * W_S
            
    if sum_W == 0: raise ValueError("No valid data in other spectra to compute a weighted average.")
    avg_I = sum_IW / sum_W
    
    # 2. Extract target spectrum transfer line properties and its specific weight
    target_i_col, target_snr_col, target_width_col = f'{target_spectrum}\nIntensity', f'{target_spectrum}\nSNR', f'{target_spectrum}\nWidth'
    
    I_target = pd.to_numeric(df_out.at[transfer_label, target_i_col], errors='coerce')
    SNR_target = pd.to_numeric(df_out.at[transfer_label, target_snr_col], errors='coerce')
    Width_target = pd.to_numeric(df_out.at[transfer_label, target_width_col], errors='coerce')
    Wnum_target = pd.to_numeric(df_out.at[transfer_label, 'wavenumber'], errors='coerce')
    
    if pd.isna(I_target) or I_target <= 0 or pd.isna(SNR_target) or SNR_target <= 0:
        raise ValueError("Target spectrum does not have a valid measurement for the selected transfer line.")
        
    intensities_target = pd.to_numeric(df_out[target_i_col], errors='coerce')
    snrs_target = pd.to_numeric(df_out[target_snr_col], errors='coerce')
    wavenumbers_target = pd.to_numeric(df_out['wavenumber'], errors='coerce')
    
    valid_mask_target = intensities_target.notna() & snrs_target.notna()
    if f'{target_spectrum}\nExcluded' in df_out.columns:
        valid_mask_target &= ~df_out[f'{target_spectrum}\nExcluded'].fillna(False).astype(bool)
        
    w_maxI_target = wavenumbers_target.loc[intensities_target[valid_mask_target].idxmax()] if valid_mask_target.any() else Wnum_target

    # Updated to resolutn, bandlo, bandhi
    resolutn_target, bandlo_target, bandhi_target = 0.05, 0.0, 30000.0
    if h5_filepath:
        try:
            with h5py.File(h5_filepath, 'r') as f:
                spec_path = f"/Spectra/{target_spectrum}/Raw_Data/spectrum"
                if spec_path in f:
                    attrs = f[spec_path].attrs
                    resolutn_target = float(attrs.get('resolutn', 0.05))
                    bandlo_target = float(attrs.get('bandlo', attrs.get('wstart', 0.0)))
                    bandhi_target = float(attrs.get('bandhi', attrs.get('wend', bandlo_target + 30000.0)))
                    if bandhi_target <= bandlo_target: bandhi_target = bandlo_target + 30000.0
        except Exception: pass
        
    calunc_per_1000_t = 70.0 / (bandhi_target - bandlo_target) if bandhi_target > bandlo_target else 0.0
    root_npts_t = (Width_target / 1000.0) / resolutn_target if pd.notna(Width_target) else 1.0
    if root_npts_t <= 0: root_npts_t = 1.0
    
    variance_t = 2.25 / ((SNR_target**2) * root_npts_t)
    calunc_t = calunc_per_1000_t * abs(Wnum_target - w_maxI_target) / 1000.0
    W_target = 1.0 / (calunc_t**2 + variance_t)

    # 3. Calculate scaling factor and root sum of squares uncertainty
    scale = avg_I / I_target
    unc_renorm = ( W_target + sum_W )**(-0.5)
    
    # 4. Apply to the target spectrum
    for r_label in df_out.index:
        I_old = pd.to_numeric(df_out.at[r_label, target_i_col], errors='coerce')
        SNR_old = pd.to_numeric(df_out.at[r_label, target_snr_col], errors='coerce')
        
        if pd.notna(I_old) and I_old > 0:
            df_out.at[r_label, target_i_col] = I_old * scale
            
        if pd.notna(SNR_old) and SNR_old > 0:
            if r_label == transfer_label:
                df_out.at[r_label, target_snr_col] = 1.0 / unc_renorm
            else:
                U_old = 1.0 / SNR_old
                U_new = (U_old**2 + unc_renorm**2)**0.5
                df_out.at[r_label, target_snr_col] = 1.0 / U_new
                
    return df_out

def match_wavenumbers(experimental_linelist: pd.DataFrame,
                      previous_ids: pd.DataFrame,
                      tolerance: float = 0.1) -> pd.DataFrame:
    if experimental_linelist.empty or previous_ids.empty: return pd.DataFrame()
    if 'wavenumber' not in experimental_linelist.columns or 'wavenumber' not in previous_ids.columns:
        raise ValueError("Both input DataFrames must contain a 'wavenumber' column.")
    
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
    exp_df = h5_manager.read_hdf_table_robustly(h5_filepath, exp_path)
    ids_df = h5_manager.read_hdf_table_robustly(h5_filepath, ids_path)
    target_spectrum_group = '/'.join(exp_path.split('/')[:3])
    matched_df = match_wavenumbers(exp_df, ids_df, tolerance)
    if matched_df.empty: return 0
    output_group_path = f"{target_spectrum_group}/Identified_Lines"
    sanitized_output_name = output_name.replace('.', '_').replace('-', '_')
    metadata = {'analysis_date': datetime.now().isoformat(), 'analysis_type': 'Wavenumber Matching'}
    h5_manager.add_pandas_table(h5_filepath, output_group_path, sanitized_output_name, matched_df, metadata_dict=metadata)
    return len(matched_df)

def calculate_branching_fractions(lines_for_calculation: pd.DataFrame, 
                                  upper_level_key: str,
                                  energy_levels_df: pd.DataFrame,
                                  calculations_df: pd.DataFrame = None,
                                  wavenumber_tolerance: float = 0.1) -> pd.DataFrame:
    if lines_for_calculation.empty: return pd.DataFrame()
    df = lines_for_calculation.copy()
    if 'Mean Intensity' not in df.columns: df = add_weighted_averages(df)
        
    df.dropna(subset=['Mean Intensity'], inplace=True)
    if df.empty: return pd.DataFrame()

    clean_target_key = str(upper_level_key).replace('*', '').strip()

    lifetime = 0.0
    life_unc_frac = 0.0  # Default to 0.0 instead of hard-coded 0.1
    if not energy_levels_df.empty and 'key' in energy_levels_df.columns and 'lifetime' in energy_levels_df.columns:
        energy_levels_df['key_clean'] = energy_levels_df['key'].astype(str).str.replace('*', '', regex=False).str.strip()
        matches = energy_levels_df[energy_levels_df['key_clean'] == clean_target_key]
        if not matches.empty:
            try:
                lifetime = float(matches.iloc[0]['lifetime'])
            except ValueError: pass
            
            # Look for the new fractional uncertainty column
            if 'lifetime_unc_frac' in matches.columns:
                try:
                    val = matches.iloc[0]['lifetime_unc_frac']
                    if not pd.isna(val):
                        life_unc_frac = float(val)
                except ValueError: pass

    frac_resid, unobserved_A_sum, matched_theo_A = 0.0, 0.0, {}
    
    if calculations_df is not None and not calculations_df.empty:
        if 'upper_level_key' not in calculations_df.columns and 'upper_level_designation' in calculations_df.columns:
            calculations_df['upper_level_key'] = calculations_df['upper_level_designation']
            
        if 'upper_level_key' in calculations_df.columns:
            calculations_df['upper_level_key_clean'] = calculations_df['upper_level_key'].astype(str).str.replace('*', '', regex=False).str.strip()
            theo_lines = calculations_df[calculations_df['upper_level_key_clean'] == clean_target_key].copy()
            
            if not theo_lines.empty and 'wavenumber' in theo_lines.columns and 'transition_probability' in theo_lines.columns:
                theo_lines['wavenumber'] = pd.to_numeric(theo_lines['wavenumber'], errors='coerce')
                theo_lines['transition_probability'] = pd.to_numeric(theo_lines['transition_probability'], errors='coerce')
                theo_lines.dropna(subset=['wavenumber', 'transition_probability'], inplace=True)
                observed_wns = pd.to_numeric(df['wavenumber'], errors='coerce').dropna().values
                
                for idx, row in theo_lines.iterrows():
                    t_wn, t_A = row['wavenumber'], row['transition_probability']
                    
                    diffs = np.abs(observed_wns - t_wn)
                    if len(diffs) > 0 and np.min(diffs) <= wavenumber_tolerance:
                        matched_theo_A[observed_wns[np.argmin(diffs)]] = t_A
                    else:
                        unobserved_A_sum += t_A
                        print(t_wn, t_A)
                        
                if lifetime > 0:
                    frac_resid = unobserved_A_sum * lifetime / 1000.0

    valid_intensities = pd.to_numeric(df['Mean Intensity'], errors='coerce').fillna(0.0)
    total_int = valid_intensities.sum() * (1.0 + frac_resid)

   
    if total_int > 0:
        df['Branching Fraction'] = valid_intensities / total_int
        fractional_unc = pd.to_numeric(df['Mean Uncertainty'], errors='coerce').fillna(0.0)
        BFsq = np.sum( (df['Branching Fraction'].values ** 2) * (fractional_unc ** 2) ) + ((frac_resid ** 2) * 0.25)
        
        df['BF Uncertainty (%)'], df['Trans. Prob. (10^6 s^-1)'], df['Trans. Prob. Unc. (%)'], df['Theoretical Trans. Prob.'] = 0.0, 0.0, 0.0, np.nan
        
        for index, row in df.iterrows():
            BF, delta_I = row['Branching Fraction'], fractional_unc.get(index, 0.0)
            rel_var_BF = max((delta_I ** 2) * (1.0 - 2.0 * BF) + BFsq, 0.0)
            df.at[index, 'BF Uncertainty (%)'] = np.sqrt(rel_var_BF) * 100.0
            
            if lifetime > 0:
                df.at[index, 'Trans. Prob. (10^6 s^-1)'] = (1000.0 * BF) / lifetime
                df.at[index, 'Trans. Prob. Unc. (%)'] = np.sqrt(rel_var_BF + life_unc_frac ** 2) * 100.0 
                
            wn = pd.to_numeric(row['wavenumber'], errors='coerce')
            if not pd.isna(wn):
                min_diff, closest_theo = 1e9, None
                for t_wn, t_A in matched_theo_A.items():
                    if abs(t_wn - wn) < min_diff and abs(t_wn - wn) <= wavenumber_tolerance:
                        min_diff, closest_theo = abs(t_wn - wn), t_A
                if closest_theo is not None: df.at[index, 'Theoretical Trans. Prob.'] = closest_theo
    else:
        df['Branching Fraction'] = 0.0

    result_cols =['wavenumber', 'lower_level_key', 'Mean Intensity', 'Mean Uncertainty', 
                   'Branching Fraction', 'BF Uncertainty (%)', 'Trans. Prob. (10^6 s^-1)', 
                   'Trans. Prob. Unc. (%)', 'Theoretical Trans. Prob.']
    results = df[[col for col in result_cols if col in df.columns]].copy()
    results.attrs['residual_fraction'] = frac_resid
    results.attrs['lifetime'] = lifetime
    
    return results

def normalize_intensities_by_reference_line(master_df: pd.DataFrame, reference_line_index: int) -> pd.DataFrame:
    if master_df.empty or not (0 <= reference_line_index < len(master_df)): return master_df
    normalized_df = master_df.copy()
    intensity_cols =[col for col in normalized_df.columns if isinstance(col, str) and '\nIntensity' in col]
    reference_line = normalized_df.iloc[reference_line_index]

    for col in intensity_cols:
        norm_factor = reference_line.get(col)
        if pd.notna(norm_factor) and norm_factor > 0:
            normalized_df[col] = (normalized_df[col] / norm_factor) * 1000.0
    return normalized_df

def calculate_outliers(df: pd.DataFrame, h5_filepath: str = None) -> pd.DataFrame:
    """
    Identifies intensity values that are more than 3 sigma away from the mean.
    
    The standard deviation (sigma) is calculated as the root-sum-of-squares of
    the individual line's absolute uncertainty and the mean's absolute uncertainty.
    
    Returns:
        pd.DataFrame: A boolean DataFrame of the same shape as the input,
                      with True marking the cells that are outliers.
    """
    if df.empty or 'Mean Intensity' not in df.columns:
        return pd.DataFrame()

    # Create a boolean dataframe to store the highlight flags, default to False
    highlight_df = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df.columns if '\n' in col])))

    # Pre-calculate constants for each spectrum to avoid repeating work in the main loop
    spec_params = {}
    for name in spectrum_names:
        wavenumbers = pd.to_numeric(df['wavenumber'], errors='coerce')
        intensities = pd.to_numeric(df.get(f'{name}\nIntensity'), errors='coerce')
        snrs = pd.to_numeric(df.get(f'{name}\nSNR'), errors='coerce')
        
        valid_mask = intensities.notna() & snrs.notna()
        if f'{name}\nExcluded' in df.columns:
            valid_mask &= ~df[f'{name}\nExcluded'].fillna(False).astype(bool)
            
        w_maxI = wavenumbers.loc[intensities[valid_mask].idxmax()] if valid_mask.any() else 0.0
        
        resolutn, bandlo, bandhi = 0.05, 0.0, 30000.0
        if h5_filepath:
            try:
                with h5py.File(h5_filepath, 'r') as f:
                    spec_path = f"/Spectra/{name}/Raw_Data/spectrum"
                    if spec_path in f:
                        attrs = f[spec_path].attrs
                        resolutn = float(attrs.get('resolutn', 0.05))
                        bandlo = float(attrs.get('bandlo', attrs.get('wstart', 0.0)))
                        bandhi = float(attrs.get('bandhi', attrs.get('wend', bandlo + 30000.0)))
                        if bandhi <= bandlo: bandhi = bandlo + 30000.0
            except Exception: pass
        
        calunc_per_1000 = 70.0 / (bandhi - bandlo) if bandhi > bandlo else 0.0
        spec_params[name] = {'w_maxI': w_maxI, 'resolutn': resolutn, 'calunc_per_1000': calunc_per_1000}

    # Main loop to check each cell
    for index, row in df.iterrows():
        mean_I = row.get('Mean Intensity')
        mean_unc_frac = row.get('Mean Uncertainty')
        
        if pd.isna(mean_I) or pd.isna(mean_unc_frac):
            continue
            
        sigma_mean_abs_sq = (mean_I * mean_unc_frac)**2

        for name in spectrum_names:
            intensity_col = f'{name}\nIntensity'
            I_spec = row.get(intensity_col)
            snr_spec = row.get(f'{name}\nSNR')
            width_spec = row.get(f'{name}\nWidth')
            wnum_spec = pd.to_numeric(row['wavenumber'], errors='coerce')

            if pd.isna(I_spec) or pd.isna(snr_spec) or pd.isna(wnum_spec) or snr_spec == 0:
                continue

            # Calculate the absolute uncertainty of the individual measurement
            params = spec_params[name]
            root_npts = (pd.to_numeric(width_spec, errors='coerce') / 1000.0) / params['resolutn']
            root_npts = 1.0 if pd.isna(root_npts) or root_npts <= 0 else root_npts
            
            variance_instrumental = 2.25 / ((snr_spec**2) * root_npts)
            calunc = params['calunc_per_1000'] * abs(wnum_spec - params['w_maxI']) / 1000.0
            total_variance_spec = (calunc**2) + variance_instrumental
            sigma_spec_abs_sq = total_variance_spec * (I_spec**2)

            # Calculate joint uncertainty and check condition
            joint_sigma = (sigma_spec_abs_sq + sigma_mean_abs_sq)**0.5
            deviation = abs(I_spec - mean_I)

            if deviation > (3 * joint_sigma):
                highlight_df.at[index, intensity_col] = True
                
    return highlight_df