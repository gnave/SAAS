import h5py

filename = 'test.h5'  # <-- CHANGE THIS TO YOUR FILE's NAME

with h5py.File(filename, 'a') as f:
    if '/Levels' in f:
        # Overwrite the old schema with the new one
        f['/Levels'].attrs['schema'] = 'key,energy,j_value,parity,lifetime,lifetime_unc_frac,designation'
        print(f"Successfully updated the schema for {filename}!")
    else:
        print("Error: '/Levels' group not found in this file.")
