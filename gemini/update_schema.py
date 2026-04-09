import h5py

# File to update the schema in a hdf5 file, if you decide to change a data structure within it.
# Edit as appropriate for your files.

filename = 'test.h5'  # <-- CHANGE THIS TO YOUR FILE's NAME

with h5py.File(filename, 'a') as f:
    if '/Levels' in f:
        # Overwrite the old schema with the new one. Modify according to the needed schema
        f['/Levels'].attrs['schema'] = 'key,energy,j_value,parity,lifetime,lifetime_unc_frac,designation'
        print(f"Successfully updated the schema for {filename}!")
    else:
        print("Error: '/Levels' group not found in this file.")
