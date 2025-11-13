# analysis.py (Definitive Final Version with Brute-Force Matching)

import pandas as pd
import numpy as np
import h5py
import h5_manager # NEW IMPORT
from datetime import datetime # NEW IMPORT

def match_wavenumbers(experimental_linelist: pd.DataFrame, 
                      previous_ids: pd.DataFrame, 
                      tolerance: float = 0.02) -> pd.DataFrame:
    """
    Matches lines between an experimental linelist and a list of previous identifications
    using a robust, brute-force search for the absolute nearest neighbor.
    """
    print("\n--- Running Wavenumber Matching ---")
    print(f"Experimental linelist size: {len(experimental_linelist)} rows")
    print(f"Previous IDs size: {len(previous_ids)} rows")
    print(f"Matching tolerance: {tolerance} cm⁻¹")

    if experimental_linelist.empty or previous_ids.empty:
        print("Warning: One or both input DataFrames are empty. Returning empty result.")
        return pd.DataFrame()

    if 'wavenumber' not in experimental_linelist.columns or 'wavenumber' not in previous_ids.columns:
        raise ValueError("Both input DataFrames must contain a 'wavenumber' column.")

    # Create copies and ensure data is clean
    exp_df = experimental_linelist.copy()
    ids_df = previous_ids.copy()
    exp_df['wavenumber'] = pd.to_numeric(exp_df['wavenumber'], errors='coerce')
    ids_df['wavenumber'] = pd.to_numeric(ids_df['wavenumber'], errors='coerce')
    exp_df.dropna(subset=['wavenumber'], inplace=True)
    ids_df.dropna(subset=['wavenumber'], inplace=True)
    
    if exp_df.empty or ids_df.empty:
        print(f"Warning: One or both DataFrames became empty after cleaning. Exp rows: {len(exp_df)}, IDs rows: {len(ids_df)}")
        print("--- Wavenumber Matching Complete --- (No matches found)")
        return pd.DataFrame()

    matches = []
    
    # Extract the wavenumbers from the previous IDs into a NumPy array for fast searching
    id_wavenumbers = ids_df['wavenumber'].values
    
    # Iterate through each and every line in the experimental linelist
    for index, exp_line in exp_df.iterrows():
        exp_wn = exp_line['wavenumber']
        
        # Calculate the absolute difference between this line and ALL lines in the previous IDs
        differences = np.abs(id_wavenumbers - exp_wn)
        
        # Find the index of the smallest difference
        best_match_index = np.argmin(differences)
        
        # Check if this smallest difference is within our tolerance
        if differences[best_match_index] <= tolerance:
            # It's a valid match. Get the corresponding row from the previous IDs DataFrame.
            id_line = ids_df.iloc[best_match_index]
            
            # --- START OF DEFINITIVE FIX ---
            # Combine the two series (rows) into a single dictionary.
            # Manually add suffixes to prevent column name collisions.
            combined_data = {}
            for col, value in exp_line.items():
                combined_data[f"{col}_exp"] = value
            for col, value in id_line.items():
                combined_data[f"{col}_id"] = value
            
            # The 'on' column ('wavenumber') will be duplicated with suffixes. 
            # We restore the original name for the experimental value.
            combined_data['wavenumber'] = combined_data.pop('wavenumber_exp')
            # --- END OF DEFINITIVE FIX ---
            
            matches.append(combined_data)

    if not matches:
        print("Found 0 matches.")
        print("--- Wavenumber Matching Complete ---\n")
        return pd.DataFrame()

    # Convert the list of matched dictionaries into a final DataFrame
    matched_df = pd.DataFrame(matches)
    
    # Calculate the final difference for verification
    matched_df['wavenumber_diff'] = (matched_df['wavenumber'] - matched_df['wavenumber_id']).abs()

    print(f"Found {len(matched_df)} matches.")
    print("--- Wavenumber Matching Complete ---\n")
    
    return matched_df

def run_and_save_wavenumber_match(h5_filepath, exp_path, ids_path, tolerance, output_name):
    """
    Orchestrates the full process: reading data, running the match, and saving the results.
    """
    print("--- Reading Data for Wavenumber Matching ---")
    
    # Read the selected tables into DataFrames
    with h5py.File(h5_filepath, 'r') as f:
        # Use a helper function from gui.py to robustly read tables
        # Note: This creates a dependency, which is acceptable for this application structure.
        from gui import read_hdf_table_robustly
        exp_df = read_hdf_table_robustly(f[exp_path])
        ids_df = read_hdf_table_robustly(f[ids_path])
        
        # Determine the target spectrum group from the experimental linelist path
        # e.g., /Spectra/spec_name/Raw_Linelists/table -> /Spectra/spec_name
        target_spectrum_group = '/'.join(exp_path.split('/')[:3])

    # Run the core analysis function
    matched_df = match_wavenumbers(exp_df, ids_df, tolerance)

    if matched_df.empty:
        print("No matches found. Nothing to save.")
        return 0 # Return the number of matches

    # Prepare for saving
    output_group_path = f"{target_spectrum_group}/Identified_Lines"
    sanitized_output_name = output_name.replace('.', '_').replace('-', '_')

    # Create metadata dictionary
    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'analysis_type': 'Wavenumber Matching',
        'source_experimental_linelist': exp_path,
        'source_previous_identifications': ids_path,
        'matching_tolerance_cm-1': tolerance
    }
    
    print(f"--- Saving {len(matched_df)} matched lines to: {output_group_path}/{sanitized_output_name} ---")
    h5_manager.add_pandas_table(
        h5_filepath, 
        output_group_path, 
        sanitized_output_name, 
        matched_df, 
        metadata_dict=metadata
    )
    
    return len(matched_df)