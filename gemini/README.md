# SAAS - Atomic Spectra Analysis Tool

SAAS is a standalone tool for managing spectroscopic data. It currently can perform branching fraction analysis of files in a hdf5 file

Directory 'data' contains test data files for the program. The file 'test.h5' contains a complete set of data for Cr II branching fraction analysis using the files in the data directory.

## Prerequisites
* **Python 3.8+**
* **Pip** (Python package manager)
* xwayland (Linux)

## Building the Program
If you have the source code and want to build the standalone executable:

1. Open a terminal in the project folder.
2. Run the build script:
   ```bash
   chmod +x build.sh
   ./build.sh


