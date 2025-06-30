# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a spectroscopy analysis project focused on creating an HDF5 import wizard for FTS (Fourier Transform Spectrometry) data. The system handles atomic spectroscopy data including emission spectra, energy levels, and transition calculations.

## Key Data Structures

The project works with HDF5 files containing structured spectroscopy data organized into groups:

- **Calculations**: Transition probabilities with wavenumber, level keys, and probabilities
- **Levels**: Energy levels with J-values, parity, and lifetimes  
- **Standard lamp calibrations**: Wavelength and spectral radiance data
- **Previous identifications**: Known spectral line identifications
- **Spectra**: Raw and processed spectra data from .dat/.hdr file pairs

## Input File Formats

- `.dat` files: Binary spectra (4-byte float arrays)
- `.hdr` files: ASCII metadata with 'keyword = value' format, comments start with '/'
- `.csv/.txt` files: Tabular data with first line containing column metadata
- `.h5` files: HDF5 structured data storage

## Sample Data

The `cr_demo_data/` directory contains example files:
- `cr042416.005_r.dat/hdr`: Example spectrum and metadata
- `CrII_CS.csv`: Previous identifications data
- `crii_lev.csv`: Energy levels data  
- `CrII_calc.csv`: Calculated transition probabilities
- `Cr_BF2.h5`: Example HDF5 structure

## Development Notes

This appears to be a data processing project without traditional build/test commands. The main deliverable is an import wizard to convert various spectroscopy file formats into structured HDF5 format for analysis.

Key technical requirements:
- Handle binary spectral data from FTS instruments
- Parse ASCII header files with metadata
- Support incremental data import into existing HDF5 files
- Maintain referential integrity between datasets via level keys