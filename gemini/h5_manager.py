# h5_manager.py (Complete, Final Version)

import h5py
import numpy as np
import pandas as pd
from datetime import datetime

# Defines only the top-level, project-wide groups.
HDF5_STRUCTURE = {
    'Calculations': [],
    'Levels': [],
    'Standard_Lamp_Calibrations': [],
    'Previous_Identifications': [],
    'Spectra': []
}

def create_experiment_file(filepath, metadata_dict):
    """Creates a new, structured HDF5 file with predefined schemas."""
    with h5py.File(filepath, 'w') as f:
        # Create the main top-level groups
        for group_name in HDF5_STRUCTURE:
            f.create_group(group_name)
        
        # Attach project-level metadata to the root of the file
        for key, value in metadata_dict.items():
            f.attrs[key] = str(value)
        f.attrs['creation_date'] = str(datetime.now())

    print(f"Successfully created HDF5 file with standard structure at {filepath}")
    
    # Define schemas for the project-level groups
    print("Defining default group schemas...")
    define_group_schema(filepath, '/Levels', 
                        ['key', 'energy', 'j_value', 'parity', 'lifetime', 'designation'])
    
    define_group_schema(filepath, '/Calculations',
                        [
                            'log_gf', 'lower_level_key', 'upper_level_key',
                            'wavenumber', 'wavelength', 'transition_probability',
                            'lower_level_energy', 'lower_level_designation', 'lower_level_j',
                            'upper_level_energy', 'upper_level_designation', 'upper_level_j'
                        ])

    define_group_schema(filepath, '/Previous_Identifications',
                        [
                            'wavenumber', 'wavelength', 'intensity',
                            'lower_level_energy', 'lower_level_designation', 'lower_level_key',
                            'upper_level_energy', 'upper_level_designation', 'upper_level_key'
                        ])

def get_all_group_paths(filepath):
    """Traverses an HDF5 file and returns a list of all group paths."""
    groups = []
    with h5py.File(filepath, 'r') as f:
        def find_groups(name, obj):
            if isinstance(obj, h5py.Group):
                groups.append('/' + name)
        f.visititems(find_groups)
    return groups

def define_group_schema(h5_filepath, group_path, schema_list):
    """Stores a schema (list of expected column names) as an attribute of a group."""
    with h5py.File(h5_filepath, 'a') as f:
        if group_path not in f:
            f.create_group(group_path)
        f[group_path].attrs['schema'] = ",".join(schema_list)

def add_dataset_to_file(h5_filepath, group_path, dataset_name, data, metadata={}):
    """Adds a dataset (like an array) to an HDF5 file."""
    with h5py.File(h5_filepath, 'a') as f:
        if group_path not in f:
            f.create_group(group_path)
        dset = f[group_path].create_dataset(dataset_name, data=data)
        for key, value in metadata.items():
            dset.attrs[key] = value
        print(f"Added dataset '{dataset_name}' to group '{group_path}'")

def add_pandas_table(h5_filepath, group_path, table_name, df, metadata_dict=None):
    """
    Adds a pandas DataFrame as a table to the HDF5 file.
    """
    full_key = f"{group_path}/{table_name}"
    
    min_itemsize = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            # Calculate max length, ensuring we handle empty columns
            max_len = df[col].str.len().max()
            if pd.isna(max_len):
                max_len = 0
            min_itemsize[col] = int(max_len) + 10

    df.to_hdf(
        h5_filepath, 
        key=full_key, 
        mode='a', 
        format='table', 
        index=False,
        min_itemsize=min_itemsize,
        data_columns=True
    )
    
    if metadata_dict:
        with h5py.File(h5_filepath, 'a') as f:
            # For pandas tables, the key points to a group containing the 'table' dataset
            if full_key in f:
                dset_group = f[full_key]
                for key, value in metadata_dict.items():
                    dset_group.attrs[key] = str(value)
                print(f"Successfully attached {len(metadata_dict)} metadata items to table group '{table_name}'.")

    print(f"Added table '{table_name}' to group '{group_path}' with named, structured columns.")


def attach_metadata_to_dataset(h5_filepath, dataset_path, metadata_dict):
    """Attaches a dictionary of metadata as attributes to a specific dataset."""
    with h5py.File(h5_filepath, 'a') as f:
        if dataset_path not in f:
            return
        dset = f[dataset_path]
        for key, value in metadata_dict.items():
            if value is not None:
                dset.attrs[key] = value
    print(f"Successfully attached {len(metadata_dict)} metadata items to '{dataset_path}'.")

def delete_object(h5_filepath: str, h5_path: str) -> bool:
    """
    Deletes a group or dataset from an HDF5 file.
    """
    try:
        with h5py.File(h5_filepath, 'a') as f:
            if h5_path in f:
                del f[h5_path]
                print(f"Successfully deleted object: {h5_path}")
                return True
            else:
                print(f"Warning: Object not found for deletion: {h5_path}")
                return False
    except Exception as e:
        print(f"Error deleting object {h5_path}: {e}")
        return False
    
def read_hdf_table_robustly(h5_filepath, h5_dataset_path):
    """
    Reads an HDF5 dataset as a Pandas DataFrame, robustly handling byte strings.
    Assumes the dataset contains structured numpy array data.
    """
    with h5py.File(h5_filepath, 'r') as f:
        if h5_dataset_path not in f:
            raise FileNotFoundError(f"Dataset not found at {h5_dataset_path} in {h5_filepath}")
        
        h5_dataset = f[h5_dataset_path]
        
        # Check if it's a scalar dataset (not a table)
        if not isinstance(h5_dataset, h5py.Dataset) or not h5_dataset.dtype.fields:
             # Handle non-table datasets, e.g., return a single value or error
             print(f"Warning: Dataset at {h5_dataset_path} is not a structured table. Returning as Series or scalar.")
             if h5_dataset.shape:
                 return pd.Series(h5_dataset[:], name=h5_dataset_path.split('/')[-1])
             else:
                 return pd.Series([h5_dataset[()]], name=h5_dataset_path.split('/')[-1])

        data = h5_dataset[:]
    
        df_data = {}
        for col_name in data.dtype.names:
            col_data = data[col_name]
#            print(col_name,col_data)
            if np.issubdtype(col_data.dtype, np.bytes_):
                df_data[col_name] = [s.decode('utf-8') for s in col_data]
            else:
                df_data[col_name] = col_data
        return pd.DataFrame(df_data)

def create_group_if_not_exists(h5_filepath, group_path):
    """
    Creates an HDF5 group if it does not already exist.
    """
    with h5py.File(h5_filepath, 'a') as f:
        if group_path not in f:
            f.create_group(group_path)
            print(f"Created HDF5 group: {group_path}")