# h5_manager.py (FULLY DOCUMENTED)

import h5py
import numpy as np
import pandas as pd
from datetime import datetime

# Defines the standard, top-level group structure for every new project file.
HDF5_STRUCTURE = {
    'Calculations': [],
    'Levels': [],
    'Standard_Lamp_Calibrations': [],
    'Previous_Identifications': [],
    'Spectra': [],
    'Branching_Fraction_Analyses': [] # Added for storing analysis results
}

def create_experiment_file(filepath, metadata_dict):
    """
    Creates a new, structured HDF5 file for a project.

    This function initializes the file with the standard top-level group
    structure, attaches project-level metadata to the root of the file,
    and defines default "schemas" for key data groups.

    Args:
        filepath (str): The path where the new HDF5 file will be created.
        metadata_dict (dict): A dictionary of project-level metadata (e.g.,
                              author, project title) to be attached as
                              attributes to the root group.
    """
    with h5py.File(filepath, 'w') as f:
        # Create the main top-level groups from the predefined structure.
        for group_name in HDF5_STRUCTURE:
            f.create_group(group_name)
        
        # Attach project-level metadata to the root of the file.
        for key, value in metadata_dict.items():
            f.attrs[key] = str(value)
        f.attrs['creation_date'] = str(datetime.now())

    print(f"Successfully created HDF5 file with standard structure at {filepath}")
    
    # Define schemas for the project-level groups that will hold tabular data.
    # These schemas are used by the import wizard to guide column mapping.
    print("Defining default group schemas...")
    define_group_schema(filepath, '/Levels',['key', 'energy', 'j_value', 'parity', 'lifetime', 'lifetime_unc_frac', 'designation'])
    
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
    """
    Traverses an HDF5 file and returns a list of all group paths.

    Args:
        filepath (str): The path to the HDF5 file.

    Returns:
        list: A list of strings, where each string is the full path to a group.
    """
    groups = []
    with h5py.File(filepath, 'r') as f:
        def find_groups(name, obj):
            if isinstance(obj, h5py.Group):
                groups.append('/' + name)
        f.visititems(find_groups)
    return groups

def define_group_schema(h5_filepath, group_path, schema_list):
    """
    Stores a schema (list of expected column names) as an attribute of a group.
    The schema is saved as a single comma-separated string.

    Args:
        h5_filepath (str): The path to the HDF5 file.
        group_path (str): The full path to the group where the schema will be stored.
        schema_list (list): A list of strings representing the column names.
    """
    with h5py.File(h5_filepath, 'a') as f:
        if group_path not in f:
            f.create_group(group_path)
        f[group_path].attrs['schema'] = ",".join(schema_list)

def add_dataset_to_file(h5_filepath, group_path, dataset_name, data, metadata={}):
    """
    Adds a raw NumPy-like dataset (e.g., a spectrum array) to a group.

    Args:
        h5_filepath (str): Path to the HDF5 file.
        group_path (str): The group where the dataset will be created.
        dataset_name (str): The name of the new dataset.
        data (np.ndarray): The data array to be saved.
        metadata (dict, optional): A dictionary of attributes to attach to the dataset.
    """
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

    This function uses `pandas.to_hdf` in 'table' format. A critical step is
    pre-calculating the maximum required string length for object columns. This
    `min_itemsize` parameter prevents `HDF5-DIAG` errors and performance issues
    that can occur with variable-length strings in HDF5 tables.

    Note: `pandas.to_hdf` creates a *group* at the `table_name` path, inside of
    which it stores the actual dataset and other metadata. Therefore, attributes
    are attached to this parent group, not the raw dataset itself.

    Args:
        h5_filepath (str): Path to the HDF5 file.
        group_path (str): The group where the new table group will be created.
        table_name (str): The name of the new table group.
        df (pd.DataFrame): The pandas DataFrame to save.
        metadata_dict (dict, optional): A dictionary of attributes to attach
                                       to the table's parent group.
    """
    full_key = f"{group_path}/{table_name}"
    
    min_itemsize = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            # Calculate max string length in the column to pre-allocate space.
            max_len = df[col].str.len().max()
            if pd.isna(max_len):
                max_len = 0
            # Add a small buffer.
            min_itemsize[col] = int(max_len) + 10

    df.to_hdf(
        h5_filepath, 
        key=full_key, 
        mode='a', 
        format='table', 
        index=False,
        min_itemsize=min_itemsize,
        data_columns=True # Allows for querying the table later.
    )
    
    if metadata_dict:
        with h5py.File(h5_filepath, 'a') as f:
            if full_key in f:
                dset_group = f[full_key]
                for key, value in metadata_dict.items():
                    dset_group.attrs[key] = str(value)
                print(f"Successfully attached {len(metadata_dict)} metadata items to table group '{table_name}'.")

    print(f"Added table '{table_name}' to group '{group_path}' with named, structured columns.")


def attach_metadata_to_dataset(h5_filepath, dataset_path, metadata_dict):
    """
    Attaches a dictionary of metadata as attributes to a specific, existing dataset.

    Args:
        h5_filepath (str): Path to the HDF5 file.
        dataset_path (str): The full path to the target dataset.
        metadata_dict (dict): A dictionary of key-value pairs to attach as attributes.
    """
    with h5py.File(h5_filepath, 'a') as f:
        if dataset_path not in f:
            return
        dset = f[dataset_path]
        for key, value in metadata_dict.items():
            if value is not None:
                dset.attrs[key] = value
    print(f"Successfully attached {len(metadata_dict)} metadata items to '{dataset_path}'.")

def attach_metadata_to_group(h5_filepath: str, group_path: str, metadata_dict: dict):
    """
    Attaches a dictionary of metadata as attributes to a specific, existing group.

    Args:
        h5_filepath (str): Path to the HDF5 file.
        group_path (str): The full path to the target group.
        metadata_dict (dict): A dictionary of key-value pairs to attach as attributes.
    """
    try:
        with h5py.File(h5_filepath, 'a') as f:
            if group_path not in f:
                print(f"Warning: Group '{group_path}' not found. Cannot attach metadata.")
                return
            group = f[group_path]
            for key, value in metadata_dict.items():
                group.attrs[key] = str(value)
        print(f"Successfully attached {len(metadata_dict)} metadata items to group '{group_path}'.")
    except Exception as e:
        print(f"Error attaching metadata to group {group_path}: {e}")

def delete_object(h5_filepath: str, h5_path: str) -> bool:
    """
    Deletes a group or dataset from an HDF5 file.

    Args:
        h5_filepath (str): Path to the HDF5 file.
        h5_path (str): The full path to the object (group or dataset) to be deleted.

    Returns:
        bool: True if deletion was successful, False otherwise.
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
    Reads an HDF5 table dataset into a Pandas DataFrame, robustly handling byte strings.

    When h5py reads a table created by pandas, string columns are often loaded
    as NumPy byte strings (e.g., `b'my_string'`). This function explicitly checks
    for this data type (`np.bytes_`) and decodes such columns into standard
    UTF-8 Python strings, ensuring a clean DataFrame is returned.

    Args:
        h5_filepath (str): Path to the HDF5 file.
        h5_dataset_path (str): The full HDF5 path to the target table dataset
                              (e.g., '/Levels/MyLevels/table').

    Returns:
        pd.DataFrame: A DataFrame containing the table data with clean string columns.
    """
    with h5py.File(h5_filepath, 'r') as f:
        if h5_dataset_path not in f:
            raise FileNotFoundError(f"Dataset not found at {h5_dataset_path} in {h5_filepath}")
        
        h5_dataset = f[h5_dataset_path]
        
        # A pandas table is a structured array. If it's not, handle it gracefully.
        if not isinstance(h5_dataset, h5py.Dataset) or not h5_dataset.dtype.fields:
             print(f"Warning: Dataset at {h5_dataset_path} is not a structured table. Returning as Series or scalar.")
             if h5_dataset.shape:
                 return pd.Series(h5_dataset[:], name=h5_dataset_path.split('/')[-1])
             else:
                 return pd.Series([h5_dataset[()]], name=h5_dataset_path.split('/')[-1])

        # Read the entire dataset into a NumPy structured array.
        data = h5_dataset[:]
    
        # Convert the structured array to a dictionary of columns, decoding byte strings.
        df_data = {}
        for col_name in data.dtype.names:
            col_data = data[col_name]
            if np.issubdtype(col_data.dtype, np.bytes_):
                df_data[col_name] = [s.decode('utf-8') for s in col_data]
            else:
                df_data[col_name] = col_data
        return pd.DataFrame(df_data)

def create_group_if_not_exists(h5_filepath, group_path):
    """
    A utility function that creates an HDF5 group if it does not already exist.

    Args:
        h5_filepath (str): Path to the HDF5 file.
        group_path (str): The full path of the group to create.
    """
    with h5py.File(h5_filepath, 'a') as f:
        if group_path not in f:
            f.create_group(group_path)
            print(f"Created HDF5 group: {group_path}")