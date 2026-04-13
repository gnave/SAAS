# analysis.py (FULLY DOCUMENTED & CLEANED)

import pandas as pd
import numpy as np
import h5py  # RE-ADDED: Essential for reading spectrum metadata
from datetime import datetime
import h5_manager
import math

def get_spectrum_header_params(h5_filepath: str, spectrum_name: str):
    """
    Helper function to fetch spectrum metadata with legacy fallbacks.
    Consolidates metadata retrieval to one place (DRY principle).
    """
    resolutn, bandlo, bandhi = 0.05, 0.0, 30000.0
    if not h5_filepath:
        return resolutn, bandlo, bandhi
    try:
        with h5py.File(h5_filepath, 'r') as f:
            spec_path = f"/Spectra/{spectrum_name}/Raw_Data/spectrum"
            if spec_path in f:
                attrs = f[spec_path].attrs
                resolutn = float(attrs.get('resolutn', 0.05))
                bandlo = float(attrs.get('bandlo', attrs.get('wstart', 0.0)))
                bandhi = float(attrs.get('bandhi', attrs.get('wend', bandlo + 30000.0)))
                if bandhi <= bandlo: bandhi = bandlo + 30000.0
    except Exception:
        pass
    if bandhi <= bandlo:
        bandhi = bandlo + 1.0
    return resolutn, bandlo, bandhi

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
                
                rename_dict = {
                    'peak': f'{spectrum_name}\nSNR',
                    'eq_width': f'{spectrum_name}\nIntensity',
                    'width': f'{spectrum_name}\nWidth'
                }
                cols_to_keep = ['wavenumber', 'peak', 'eq_width', 'width']
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
    spectrum_cols = sorted([col for col in final_df.columns if '\n' in str(col)], 
                           key=lambda name: (name.split('\n')[0], name.split('\n')[1]))
    
    return final_df[[col for col in (base_cols + spectrum_cols) if col in final_df.columns]]

def add_weighted_averages(df: pd.DataFrame, h5_filepath: str = None) -> pd.DataFrame:
    if df.empty: return df
    df_out = df.copy()
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df.columns if '\n' in col])))
    
    sum_of_weights = pd.Series(0.0, index=df.index)
    sum_of_val_x_weight = pd.Series(0.0, index=df.index)
    max_weight = 555.0

    for name in spectrum_names:
        intensity_col, snr_col, width_col = f'{name}\nIntensity', f'{name}\nSNR', f'{name}\nWidth'
        excluded_col = f'{name}\nExcluded'
        
        if intensity_col not in df_out.columns or snr_col not in df_out.columns:
            continue
            
        intensities = pd.to_numeric(df_out[intensity_col], errors='coerce')
        snrs = pd.to_numeric(df_out[snr_col], errors='coerce')
        widths = pd.to_numeric(df_out.get(width_col, pd.Series(np.nan, index=df.index)), errors='coerce')
        wavenumbers = pd.to_numeric(df_out['wavenumber'], errors='coerce')
        
        valid_mask = intensities.notna() & snrs.notna()
        if excluded_col in df_out.columns:
            valid_mask &= ~df_out[excluded_col].fillna(False).astype(bool)
            
        if not valid_mask.any(): continue
        
        idx_max_I = intensities[valid_mask].idxmax()
        w_maxI = wavenumbers.loc[idx_max_I]
        
        resolutn, bandlo, bandhi = get_spectrum_header_params(h5_filepath, name)
            
        calunc_per_1000 = 70.0 / (bandhi - bandlo) if bandhi > bandlo else 0.0
        root_npts = ((widths / 1000.0) / resolutn).fillna(1.0).replace(0, 1.0)
        
        variance = 2.25 / ((snrs**2).replace(0, np.nan) * root_npts)
        calunc = calunc_per_1000 * (wavenumbers - w_maxI).abs() / 1000.0
        
        weights = (1.0 / ((calunc**2) + variance)).fillna(0.0)
        if excluded_col in df_out.columns:
            weights = weights.mask(df_out[excluded_col].fillna(False).astype(bool), 0.0)
            
        weights = weights.clip(upper=max_weight)
        sum_of_weights += weights
        sum_of_val_x_weight += intensities.multiply(weights).fillna(0)
        
    df_out['Mean Intensity'] = sum_of_val_x_weight.divide(sum_of_weights).replace([np.inf, -np.inf], np.nan)
    df_out['Mean Uncertainty'] = (1.0 / np.sqrt(sum_of_weights)).replace([np.inf, -np.inf], np.nan)
    
    return df_out

def transfer_calibration(df: pd.DataFrame, transfer_line_index: int, target_spectrum: str, h5_filepath: str = None) -> pd.DataFrame:
    if df.empty or not (0 <= transfer_line_index < len(df)): return df
    df_out = df.copy()
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df_out.columns if '\n' in col])))
    if target_spectrum not in spectrum_names: raise ValueError("Target spectrum not found.")
    transfer_label = df_out.index[transfer_line_index]
    
    sum_W, sum_IW = 0.0, 0.0
    for spec in spectrum_names:
        if spec == target_spectrum: continue
        if f'{spec}\nExcluded' in df_out.columns and df_out.at[transfer_label, f'{spec}\nExcluded'] == True: continue
            
        i_col, snr_col, width_col = f'{spec}\nIntensity', f'{spec}\nSNR', f'{spec}\nWidth'
        if i_col not in df_out.columns or snr_col not in df_out.columns: continue
            
        I_S = pd.to_numeric(df_out.at[transfer_label, i_col], errors='coerce')
        SNR_S = pd.to_numeric(df_out.at[transfer_label, snr_col], errors='coerce')
        Width_S = pd.to_numeric(df_out.at[transfer_label, width_col], errors='coerce')
        Wnum_S = pd.to_numeric(df_out.at[transfer_label, 'wavenumber'], errors='coerce')
        
        if pd.notna(I_S) and pd.notna(SNR_S) and I_S > 0 and SNR_S > 0:
            intensities = pd.to_numeric(df_out[i_col], errors='coerce')
            valid_mask = intensities.notna() & pd.to_numeric(df_out[snr_col], errors='coerce').notna()
            if f'{spec}\nExcluded' in df_out.columns:
                valid_mask &= ~df_out[f'{spec}\nExcluded'].fillna(False).astype(bool)
            
            w_maxI = pd.to_numeric(df_out['wavenumber'], errors='coerce').loc[intensities[valid_mask].idxmax()] if valid_mask.any() else Wnum_S
            resolutn, bandlo, bandhi = get_spectrum_header_params(h5_filepath, spec)
                
            calunc_per_1000 = 70.0 / (bandhi - bandlo) if bandhi > bandlo else 0.0
            root_npts = max((Width_S / 1000.0) / resolutn if pd.notna(Width_S) else 1.0, 1.0)
            W_S = min(1.0 / ((calunc_per_1000 * abs(Wnum_S - w_maxI) / 1000.0)**2 + (2.25 / ((SNR_S**2) * root_npts))), 555.0)
            sum_W += W_S
            sum_IW += I_S * W_S
            
    if sum_W == 0: raise ValueError("No valid data in other spectra.")
    
    I_target = pd.to_numeric(df_out.at[transfer_label, f'{target_spectrum}\nIntensity'], errors='coerce')
    SNR_target = pd.to_numeric(df_out.at[transfer_label, f'{target_spectrum}\nSNR'], errors='coerce')
    Width_target = pd.to_numeric(df_out.at[transfer_label, f'{target_spectrum}\nWidth'], errors='coerce')
    Wnum_target = pd.to_numeric(df_out.at[transfer_label, 'wavenumber'], errors='coerce')
    
    intensities_t = pd.to_numeric(df_out[f'{target_spectrum}\nIntensity'], errors='coerce')
    valid_mask_t = intensities_t.notna() & pd.to_numeric(df_out[f'{target_spectrum}\nSNR'], errors='coerce').notna()
    w_maxI_t = pd.to_numeric(df_out['wavenumber'], errors='coerce').loc[intensities_t[valid_mask_t].idxmax()] if valid_mask_t.any() else Wnum_target
    
    resolutn_t, bandlo_t, bandhi_t = get_spectrum_header_params(h5_filepath, target_spectrum)
    calunc_t = (70.0 / (bandhi_t - bandlo_t) if bandhi_t > bandlo_t else 0.0) * abs(Wnum_target - w_maxI_t) / 1000.0
    W_target = 1.0 / (calunc_t**2 + (2.25 / ((SNR_target**2) * max((Width_target / 1000.0) / resolutn_t if pd.notna(Width_target) else 1.0, 1.0))))

    scale, unc_renorm = (sum_IW / sum_W) / I_target, (W_target + sum_W)**(-0.5)
    
    for r_label in df_out.index:
        I_old, SNR_old = pd.to_numeric(df_out.at[r_label, f'{target_spectrum}\nIntensity'], errors='coerce'), pd.to_numeric(df_out.at[r_label, f'{target_spectrum}\nSNR'], errors='coerce')
        if pd.notna(I_old) and I_old > 0: df_out.at[r_label, f'{target_spectrum}\nIntensity'] = I_old * scale
        if pd.notna(SNR_old) and SNR_old > 0:
            df_out.at[r_label, f'{target_spectrum}\nSNR'] = 1.0 / unc_renorm if r_label == transfer_label else 1.0 / ((1.0/SNR_old)**2 + unc_renorm**2)**0.5
    return df_out

def match_wavenumbers(experimental_linelist, previous_ids, tolerance=0.1):
    if experimental_linelist.empty or previous_ids.empty: return pd.DataFrame()
    exp_df, ids_df = experimental_linelist.copy(), previous_ids.copy()
    exp_df['wavenumber'], ids_df['wavenumber'] = pd.to_numeric(exp_df['wavenumber'], errors='coerce'), pd.to_numeric(ids_df['wavenumber'], errors='coerce')
    exp_df.dropna(subset=['wavenumber'], inplace=True); ids_df.dropna(subset=['wavenumber'], inplace=True)
    
    matches = []
    id_wavenumbers = ids_df['wavenumber'].values
    for _, exp_line in exp_df.iterrows():
        diffs = np.abs(id_wavenumbers - exp_line['wavenumber'])
        best_idx = np.argmin(diffs)
        if diffs[best_idx] <= tolerance:
            combined = {f"{col}_exp": val for col, val in exp_line.items()}
            combined.update({f"{col}_id": val for col, val in ids_df.iloc[best_idx].items()})
            combined['wavenumber'] = combined.pop('wavenumber_exp')
            matches.append(combined)
    return pd.DataFrame(matches)

def run_and_save_wavenumber_match(h5_filepath, exp_path, ids_path, tolerance, output_name):
    exp_df, ids_df = h5_manager.read_hdf_table_robustly(h5_filepath, exp_path), h5_manager.read_hdf_table_robustly(h5_filepath, ids_path)
    matched_df = match_wavenumbers(exp_df, ids_df, tolerance)
    if matched_df.empty: return 0
    h5_manager.add_pandas_table(h5_filepath, f"{'/'.join(exp_path.split('/')[:3])}/Identified_Lines", 
                                output_name.replace('.','_').replace('-','_'), matched_df, 
                                metadata_dict={'analysis_date': datetime.now().isoformat(), 'analysis_type': 'Wavenumber Matching'})
    return len(matched_df)

def calculate_branching_fractions(lines_for_calculation: pd.DataFrame, upper_level_key: str, energy_levels_df: pd.DataFrame, calculations_df: pd.DataFrame = None, wavenumber_tolerance: float = 0.1) -> pd.DataFrame:
    if lines_for_calculation.empty: return pd.DataFrame()
    df = lines_for_calculation.copy()
    if 'Mean Intensity' not in df.columns: df = add_weighted_averages(df)
    df.dropna(subset=['Mean Intensity'], inplace=True)
    if df.empty: return pd.DataFrame()

    clean_target_key = str(upper_level_key).replace('*', '').strip()
    lifetime, life_unc_frac = 0.0, 0.0
    if not energy_levels_df.empty:
        energy_levels_df['key_clean'] = energy_levels_df['key'].astype(str).str.replace('*', '', regex=False).str.strip()
        matches = energy_levels_df[energy_levels_df['key_clean'] == clean_target_key]
        if not matches.empty:
            lifetime = float(matches.iloc[0].get('lifetime', 0.0))
            life_unc_frac = float(matches.iloc[0].get('lifetime_unc_frac', 0.0))

    frac_resid, matched_theo_A = 0.0, {}
    if calculations_df is not None and not calculations_df.empty:
        col = 'upper_level_key' if 'upper_level_key' in calculations_df.columns else 'upper_level_designation'
        calculations_df['clean_key'] = calculations_df[col].astype(str).str.replace('*', '', regex=False).str.strip()
        theo_lines = calculations_df[calculations_df['clean_key'] == clean_target_key].copy()
        if not theo_lines.empty:
            obs_wns = pd.to_numeric(df['wavenumber'], errors='coerce').dropna().values
            unobserved_A_sum = 0.0
            for _, row in theo_lines.iterrows():
                diffs = np.abs(obs_wns - float(row['wavenumber']))
                if len(diffs) > 0 and np.min(diffs) <= wavenumber_tolerance: matched_theo_A[obs_wns[np.argmin(diffs)]] = float(row['transition_probability'])
                else: unobserved_A_sum += float(row['transition_probability'])
            if lifetime > 0: frac_resid = unobserved_A_sum * lifetime / 1000.0

    valid_ints = pd.to_numeric(df['Mean Intensity'], errors='coerce').fillna(0.0)
    total_int = valid_ints.sum() * (1.0 + frac_resid)
    if total_int > 0:
        df['Branching Fraction'] = valid_ints / total_int
        frac_unc = pd.to_numeric(df['Mean Uncertainty'], errors='coerce').fillna(0.0)
        BFsq = np.sum((df['Branching Fraction']**2) * (frac_unc**2)) + ((frac_resid**2) * 0.25)
        for idx, row in df.iterrows():
            BF, dI = row['Branching Fraction'], frac_unc.get(idx, 0.0)
            rel_var_BF = max((dI**2) * (1.0 - 2.0*BF) + BFsq, 0.0)
            df.at[idx, 'BF Uncertainty (%)'] = np.sqrt(rel_var_BF) * 100.0
            if lifetime > 0:
                df.at[idx, 'Trans. Prob. (10^6 s^-1)'] = (1000.0 * BF) / lifetime
                df.at[idx, 'Trans. Prob. Unc. (%)'] = np.sqrt(rel_var_BF + life_unc_frac**2) * 100.0
            wn = pd.to_numeric(row['wavenumber'], errors='coerce')
            for t_wn, t_A in matched_theo_A.items():
                if abs(t_wn - wn) <= wavenumber_tolerance: df.at[idx, 'Theoretical Trans. Prob.'] = t_A; break
    
    result_cols = ['wavenumber', 'lower_level_key', 'Mean Intensity', 'Mean Uncertainty', 'Branching Fraction', 'BF Uncertainty (%)', 'Trans. Prob. (10^6 s^-1)', 'Trans. Prob. Unc. (%)', 'Theoretical Trans. Prob.']
    results = df[[c for c in result_cols if c in df.columns]].copy()
    results.attrs.update({'residual_fraction': frac_resid, 'lifetime': lifetime})
    return results

def normalize_intensities_by_reference_line(master_df, reference_line_index):
    if master_df.empty or not (0 <= reference_line_index < len(master_df)): return master_df
    normalized_df, ref_line = master_df.copy(), master_df.iloc[reference_line_index]
    for col in [c for c in normalized_df.columns if isinstance(c, str) and '\nIntensity' in c]:
        if pd.notna(ref_line.get(col)) and ref_line[col] > 0: normalized_df[col] = (normalized_df[col] / ref_line[col]) * 1000.0
    return normalized_df

def calculate_outliers(df: pd.DataFrame, h5_filepath: str = None) -> pd.DataFrame:
    if df.empty or 'Mean Intensity' not in df.columns: return pd.DataFrame()
    highlight_df = pd.DataFrame(False, index=df.index, columns=df.columns)
    spectrum_names = sorted(list(set([col.split('\n')[0] for col in df.columns if '\n' in col])))
    spec_params = {}
    for name in spectrum_names:
        wns, ints = pd.to_numeric(df['wavenumber'], errors='coerce'), pd.to_numeric(df.get(f'{name}\nIntensity'), errors='coerce')
        valid_mask = ints.notna() & pd.to_numeric(df.get(f'{name}\nSNR'), errors='coerce').notna()
        if f'{name}\nExcluded' in df.columns: valid_mask &= ~df[f'{name}\nExcluded'].fillna(False).astype(bool)
        w_maxI = wns.loc[ints[valid_mask].idxmax()] if valid_mask.any() else 0.0
        resolutn, bandlo, bandhi = get_spectrum_header_params(h5_filepath, name)
        spec_params[name] = {'w_maxI': w_maxI, 'resolutn': resolutn, 'calunc_per_1000': 70.0 / (bandhi - bandlo) if bandhi > bandlo else 0.0}

    for idx, row in df.iterrows():
        mI, mU = row.get('Mean Intensity'), row.get('Mean Uncertainty')
        if pd.isna(mI) or pd.isna(mU): continue
        sigma_mean_abs_sq = (mI * mU)**2
        for name in spectrum_names:
            i_col = f'{name}\nIntensity'
            I_s, SNR_s, W_s = row.get(i_col), row.get(f'{name}\nSNR'), row.get(f'{name}\nWidth')
            wn_s = pd.to_numeric(row['wavenumber'], errors='coerce')
            if pd.isna(I_s) or pd.isna(SNR_s) or pd.isna(wn_s) or SNR_s == 0: continue
            p = spec_params[name]
            root_npts = max((pd.to_numeric(W_s, errors='coerce') / 1000.0) / p['resolutn'], 1.0)
            sigma_spec_abs_sq = ((p['calunc_per_1000'] * abs(wn_s - p['w_maxI']) / 1000.0)**2 + (2.25 / ((SNR_s**2) * root_npts))) * (I_s**2)
            if abs(I_s - mI) > (3 * (sigma_spec_abs_sq + sigma_mean_abs_sq)**0.5): highlight_df.at[idx, i_col] = True
    return highlight_df