# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None
project_root = os.path.abspath(os.getcwd())

# Collect everything from the three HDF5-related packages
d_tables, b_tables, h_tables = collect_all('tables')
d_blosc, b_blosc, h_blosc = collect_all('blosc2')
d_numexpr, b_numexpr, h_numexpr = collect_all('numexpr')

# Brute force search for libblosc2.so in the conda environment
# to ensure it's not missed
extra_binaries = []
conda_lib_path = os.path.join(os.path.dirname(sys.executable), '..', 'lib')
for f in os.listdir(conda_lib_path):
    if 'libblosc2.so' in f:
        extra_binaries.append((os.path.join(conda_lib_path, f), '.'))

a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=b_tables + b_blosc + b_numexpr + extra_binaries,
    datas=[
        ('SAAS_logo.png', '.'),
        ('analysis_window.py', '.'),
        ('analysis.py', '.'),
        ('h5_manager.py', '.'),
        ('importers.py', '.'),
        ('gui.py', '.'),
        ('versioning.py', '.')
    ] + d_tables + d_blosc + d_numexpr,
    hiddenimports=[
        'analysis_window', 'analysis', 'h5_manager', 
        'importers', 'gui', 'versioning', 'ast', 'tables',
        'tables.backends.objectextension',
        'tables.utilsextension',
        'blosc2'
        'astropy'
    ] + h_tables + h_blosc + h_numexpr,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SAAS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['SAAS_logo.png'],
)
