# h5_manager.py (FULLY DOCUMENTED & CLEANED)

import h5py
import numpy as np
import pandas as pd
from datetime import datetime

HDF5_STRUCTURE = {
    'Calculations': [], 'Levels': [], 'Standard_Lamp_Calibrations': [],
    'Previous_Identifications': [], 'Spectra': [], 'Branching_Fraction_Analyses': []
}

def create_experiment_file(filepath, metadata_dict):
    with h5py.File(filepath, 'w') as f:
        for g in HDF5_STRUCTURE: f.create_group(g)
        for k, v in metadata_dict.items(): f.attrs[k] = str(v)
        f.attrs['creation_date'] = str(datetime.now())
    define_group_schema(filepath, '/Levels', ['key', 'energy', 'j_value', 'parity', 'lifetime', 'lifetime_unc_frac', 'designation'])
    define_group_schema(filepath, '/Calculations', ['log_gf', 'lower_level_key', 'upper_level_key', 'wavenumber', 'wavelength', 'transition_probability', 'lower_level_energy', 'lower_level_designation', 'lower_level_j', 'upper_level_energy', 'upper_level_designation', 'upper_level_j'])
    define_group_schema(filepath, '/Previous_Identifications', ['wavenumber', 'wavelength', 'intensity', 'lower_level_energy', 'lower_level_designation', 'lower_level_key', 'upper_level_energy', 'upper_level_designation', 'upper_level_key'])

def get_all_group_paths(filepath):
    groups = []
    with h5py.File(filepath, 'r') as f:
        def find(name, obj):
            if isinstance(obj, h5py.Group): groups.append('/' + name)
        f.visititems(find)
    return groups

def define_group_schema(h5_filepath, group_path, schema_list):
    with h5py.File(h5_filepath, 'a') as f:
        if group_path not in f: f.create_group(group_path)
        f[group_path].attrs['schema'] = ",".join(schema_list)

def add_dataset_to_file(h5_filepath, group_path, dataset_name, data, metadata={}):
    with h5py.File(h5_filepath, 'a') as f:
        if group_path not in f: f.create_group(group_path)
        dset = f[group_path].create_dataset(dataset_name, data=data)
        for k, v in metadata.items(): dset.attrs[k] = v

def add_pandas_table(h5_filepath, group_path, table_name, df, metadata_dict=None):
    full_key = f"{group_path}/{table_name}"
    min_size = {c: int(df[c].str.len().max() or 0) + 10 for c in df.columns if df[c].dtype == 'object'}
    df.to_hdf(
        h5_filepath, 
        key=full_key, 
        mode='a', 
        format='table', 
        index=False,
        min_itemsize=min_itemsize,
        data_columns=True,
        complevel=9,      # High compression
        complib='blosc'   # Fast compression
    )
    if metadata_dict:
        with h5py.File(h5_filepath, 'a') as f:
            if full_key in f:
                for k, v in metadata_dict.items(): f[full_key].attrs[k] = str(v)

def attach_metadata_to_group(h5_filepath, group_path, metadata_dict):
    with h5py.File(h5_filepath, 'a') as f:
        if group_path in f:
            for k, v in metadata_dict.items(): f[group_path].attrs[k] = str(v)

def delete_object(h5_filepath, h5_path):
    try:
        with h5py.File(h5_filepath, 'a') as f:
            if h5_path in f: del f[h5_path]; return True
    except Exception: return False
    return False
    
def read_hdf_table_robustly(h5_filepath, h5_dataset_path):
    with h5py.File(h5_filepath, 'r') as f:
        ds = f[h5_dataset_path]
        if not isinstance(ds, h5py.Dataset) or not ds.dtype.fields: return pd.DataFrame(ds[:]) if ds.shape else pd.DataFrame([ds[()]])
        data = ds[:]
        return pd.DataFrame({n: ([s.decode('utf-8') for s in data[n]] if np.issubdtype(data[n].dtype, np.bytes_) else data[n]) for n in data.dtype.names})

def create_group_if_not_exists(h5_filepath, group_path):
    with h5py.File(h5_filepath, 'a') as f:
        if group_path not in f: f.create_group(group_path)