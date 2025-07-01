"""
HDF5 Import Wizard GUI for Spectroscopy Data
A PyQt6-based application for importing various spectroscopy file formats into HDF5.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QPushButton, QFileDialog, QTextEdit,
    QTableWidget, QTableWidgetItem, QComboBox, QLineEdit,
    QProgressBar, QSplitter, QGroupBox, QCheckBox, QSpinBox,
    QMessageBox, QStatusBar, QMenuBar, QMenu, QListWidget,
    QListWidgetItem, QFrame, QScrollArea, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap, QAction

from file_parsers import FileParserFactory, ParseError
from hdf5_manager import HDF5Manager, HDF5Error
from project_file_creation import import_spectrum_with_linelist


class FileImportWorker(QThread):
    """Worker thread for file import operations."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    import_completed = pyqtSignal(str)
    
    def __init__(self, files_to_import: List[Dict], hdf5_path: str):
        super().__init__()
        self.files_to_import = files_to_import
        self.hdf5_path = hdf5_path
        self.is_cancelled = False
    
    def cancel(self):
        """Cancel the import operation."""
        self.is_cancelled = True
    
    def run(self):
        """Run the import process."""
        try:
            total_files = len(self.files_to_import)
            
            with HDF5Manager(self.hdf5_path, 'a') as hdf5_mgr:
                # Create group structure if needed
                hdf5_mgr.create_group_structure()
                
                for i, file_info in enumerate(self.files_to_import):
                    if self.is_cancelled:
                        break
                    
                    filepath = file_info['path']
                    data_type = file_info['type']
                    dataset_name = file_info.get('dataset_name', None)
                    
                    self.status_updated.emit(f"Importing {Path(filepath).name}...")
                    
                    try:
                        if data_type == 'spectrum_pair':
                            dat_file = file_info['path']
                            hdr_file = file_info['hdr_file']
                            linelist_file = file_info.get('linelist_file', None)
                            import_spectrum_with_linelist(hdf5_mgr.file, dat_file, hdr_file, linelist_file)
                        elif data_type == 'linelist':
                            self._import_standalone_linelist(hdf5_mgr, file_info)
                        elif data_type in ['calculations', 'levels', 'identifications', 'calibration']:
                            self._import_tabular_data(hdf5_mgr, file_info)
                        
                        progress = int((i + 1) / total_files * 100)
                        self.progress_updated.emit(progress)
                        
                    except Exception as e:
                        self.error_occurred.emit(f"Error importing {filepath}: {str(e)}")
                        continue
            
            if not self.is_cancelled:
                self.import_completed.emit("Import completed successfully!")
            
        except Exception as e:
            self.error_occurred.emit(f"Critical import error: {str(e)}")
    
    def _import_spectrum_pair(self, hdf5_mgr: HDF5Manager, file_info: Dict):
        """Import spectrum .dat/.hdr file pair."""
        dat_file = file_info['path']
        hdr_file = file_info['hdr_file']
        
        # Parse files
        hdr_result = FileParserFactory.parse_file(hdr_file)
        dat_result = FileParserFactory.parse_file(dat_file, metadata=hdr_result['metadata'])
        
        # Import to HDF5
        spectrum_name = file_info.get('dataset_name') or Path(dat_file).stem
        hdf5_mgr.add_spectrum(
            spectrum_data=dat_result['spectrum'],
            wavenumbers=dat_result['wavenumbers'],
            metadata=hdr_result['metadata'],
            spectrum_name=spectrum_name
        )
    
    def _import_tabular_data(self, hdf5_mgr: HDF5Manager, file_info: Dict):
        """Import tabular data (CSV/text files)."""
        filepath = file_info['path']
        data_type = file_info['type']
        dataset_name = file_info.get('dataset_name')
        column_mapping = file_info.get('column_mapping', {})
        
        # Parse file
        result = FileParserFactory.parse_file(filepath)
        df = result['data']
        
        # Apply column mapping if provided
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Import based on data type
        if data_type == 'calculations':
            hdf5_mgr.add_calculations(df, dataset_name)
        elif data_type == 'levels':
            hdf5_mgr.add_levels(df, dataset_name)
        elif data_type == 'identifications':
            hdf5_mgr.add_identifications(df, dataset_name)
        elif data_type == 'calibration':
            hdf5_mgr.add_calibration(df, dataset_name)
    
    def _import_standalone_linelist(self, hdf5_mgr: HDF5Manager, file_info: Dict):
        """Import standalone linelist file to an existing spectrum."""
        linelist_file = file_info['path']
        target_spectrum = file_info['target_spectrum']
        
        # Import the linelist using the h5py-compatible function
        from project_file_creation import import_spectrum_with_linelist_h5py
        
        # Create a dummy spectrum entry to use the existing function
        # We'll only use the linelist part of the import
        try:
            # Check if target spectrum exists
            if 'spectra' not in hdf5_mgr.file or target_spectrum not in hdf5_mgr.file['spectra']:
                raise ValueError(f"Target spectrum '{target_spectrum}' not found in HDF5 file")
            
            # Parse linelist file
            from file_parsers import FileParserFactory
            import numpy as np
            
            parsed_data = FileParserFactory.parse_file(linelist_file)
            spectrum_group = hdf5_mgr.file['spectra'][target_spectrum]
            
            # Find next available linelist version
            version = 1
            while True:
                table_name = f'linelist_v{version}' if version > 1 else 'linelist'
                if table_name not in spectrum_group:
                    break
                version += 1
            
            # Create structured array for linelist data
            lines = parsed_data['lines']
            if lines:
                # Create structured array
                dtype = [
                    ('line_num', 'i2'),
                    ('wavenumber', 'f8'),
                    ('peak', 'f8'),
                    ('width', 'f8'),
                    ('damping', 'f8'),
                    ('eq_width', 'f8'),
                    ('itn', 'i2'),
                    ('H', 'i2'),
                    ('tags', 'S8')
                ]
                
                linelist_data = np.array([
                    (line['number'], line['wavenumber'], line['peak'], line['width'],
                     line['dmp'], line['eq_width'], line['itn'], line['H'], line['tags'].encode('utf-8'))
                    for line in lines
                ], dtype=dtype)
                
                linelist_ds = spectrum_group.create_dataset(table_name, data=linelist_data)
                
                # Store metadata
                for i, meta_line in enumerate(parsed_data['metadata']):
                    linelist_ds.attrs[f'metadata_line_{i+1}'] = meta_line
                
                # Update current linelist pointer
                spectrum_group.attrs['current_linelist'] = table_name
                print(f"Added linelist '{table_name}' to spectrum '{target_spectrum}'")
            
        except Exception as e:
            raise ValueError(f"Failed to import linelist to spectrum '{target_spectrum}': {str(e)}")


class FilePreviewWidget(QWidget):
    """Widget for previewing file contents."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # File info
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.file_info_label)
        
        # Preview area
        self.preview_area = QTextEdit()
        self.preview_area.setMaximumHeight(200)
        self.preview_area.setReadOnly(True)
        layout.addWidget(self.preview_area)
        
        # Table preview for CSV files
        self.table_preview = QTableWidget()
        self.table_preview.setMaximumHeight(200)
        layout.addWidget(self.table_preview)
        
        # Hide table initially
        self.table_preview.hide()
    
    def preview_file(self, filepath: str):
        """Preview file contents."""
        try:
            file_path = Path(filepath)
            self.file_info_label.setText(f"File: {file_path.name} ({file_path.suffix})")
            
            if file_path.suffix.lower() in ['.csv', '.txt']:
                self._preview_csv(filepath)
            elif file_path.suffix.lower() == '.hdr':
                self._preview_text(filepath)
            elif file_path.suffix.lower() == '.dat':
                self._preview_binary(filepath)
            else:
                self.preview_area.setText("Binary file - no preview available")
                self.table_preview.hide()
                self.preview_area.show()
                
        except Exception as e:
            self.preview_area.setText(f"Error previewing file: {str(e)}")
            self.table_preview.hide()
            self.preview_area.show()
    
    def _preview_csv(self, filepath: str):
        """Preview CSV file in table format."""
        try:
            result = FileParserFactory.parse_file(filepath)
            df = result['data']
            
            # Setup table
            self.table_preview.setRowCount(min(10, len(df)))
            self.table_preview.setColumnCount(len(df.columns))
            self.table_preview.setHorizontalHeaderLabels(df.columns.tolist())
            
            # Fill table with first 10 rows
            for i in range(min(10, len(df))):
                for j, col in enumerate(df.columns):
                    item = QTableWidgetItem(str(df.iloc[i, j]))
                    self.table_preview.setItem(i, j, item)
            
            self.table_preview.resizeColumnsToContents()
            self.preview_area.hide()
            self.table_preview.show()
            
        except Exception as e:
            self.preview_area.setText(f"Error parsing CSV: {str(e)}")
            self.table_preview.hide()
            self.preview_area.show()
    
    def _preview_text(self, filepath: str):
        """Preview text file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]  # First 20 lines
                preview_text = ''.join(lines)
                if len(lines) == 20:
                    preview_text += "\n... (file truncated for preview)"
                self.preview_area.setText(preview_text)
            
            self.table_preview.hide()
            self.preview_area.show()
            
        except Exception as e:
            self.preview_area.setText(f"Error reading file: {str(e)}")
    
    def _preview_binary(self, filepath: str):
        """Preview binary file info."""
        try:
            file_size = Path(filepath).stat().st_size
            self.preview_area.setText(f"Binary file\nSize: {file_size:,} bytes\n"
                                    f"Estimated points: {file_size // 4:,} (4-byte floats)")
            self.table_preview.hide()
            self.preview_area.show()
            
        except Exception as e:
            self.preview_area.setText(f"Error reading file info: {str(e)}")


class ColumnMappingWidget(QWidget):
    """Widget for mapping CSV columns to standard format."""
    
    def __init__(self):
        super().__init__()
        self.mapping_combos = {}
        self.setup_ui()
    
    def setup_ui(self):
        self.layout = QGridLayout(self)
        self.layout.addWidget(QLabel("Column Mapping:"), 0, 0, 1, 2)
    
    def setup_mapping(self, columns: List[str], data_type: str):
        """Setup column mapping interface."""
        # Clear existing widgets
        for i in reversed(range(1, self.layout.rowCount())):
            for j in range(self.layout.columnCount()):
                item = self.layout.itemAtPosition(i, j)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
        
        self.mapping_combos.clear()
        
        # Standard columns for each data type
        standard_columns = {
            'calculations': ['wavenumber', 'lower_key', 'upper_key', 'probability'],
            'levels': ['key', 'energy', 'j_value', 'parity', 'lifetime'],
            'identifications': ['wavenumber', 'lower_key', 'upper_key', 'intensity'],
            'calibration': ['wavelength', 'spectral_radiance']
        }
        
        if data_type not in standard_columns:
            return
        
        std_cols = standard_columns[data_type]
        
        # Create mapping widgets
        for i, std_col in enumerate(std_cols):
            row = i + 1
            
            # Standard column label
            std_label = QLabel(f"{std_col}:")
            self.layout.addWidget(std_label, row, 0)
            
            # File column combo
            combo = QComboBox()
            combo.addItem("(not mapped)")
            combo.addItems(columns)
            
            # Auto-map if possible
            for file_col in columns:
                if std_col.lower() in file_col.lower():
                    combo.setCurrentText(file_col)
                    break
            
            self.mapping_combos[std_col] = combo
            self.layout.addWidget(combo, row, 1)
    
    def get_mapping(self) -> Dict[str, str]:
        """Get the current column mapping."""
        mapping = {}
        for std_col, combo in self.mapping_combos.items():
            file_col = combo.currentText()
            if file_col != "(not mapped)":
                mapping[file_col] = std_col
        return mapping


class ImportWizard(QMainWindow):
    """Main import wizard application."""
    
    def __init__(self):
        super().__init__()
        self.hdf5_path = None
        self.files_to_import = []
        self.import_worker = None
        
        self.setup_ui()
        self.setup_menus()
        self.setup_status_bar()
    
    def setup_ui(self):
        """Setup the main user interface."""
        self.setWindowTitle("HDF5 Spectroscopy Import Wizard")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # HDF5 file selection
        hdf5_group = QGroupBox("HDF5 Target File")
        hdf5_layout = QHBoxLayout(hdf5_group)
        
        self.hdf5_path_edit = QLineEdit()
        self.hdf5_path_edit.setPlaceholderText("Select or create HDF5 file...")
        hdf5_layout.addWidget(self.hdf5_path_edit)
        
        self.browse_hdf5_btn = QPushButton("Browse...")
        self.browse_hdf5_btn.clicked.connect(self.browse_hdf5_file)
        hdf5_layout.addWidget(self.browse_hdf5_btn)
        
        layout.addWidget(hdf5_group)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - File selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File list
        files_group = QGroupBox("Files to Import")
        files_layout = QVBoxLayout(files_group)
        
        # File buttons
        file_buttons = QHBoxLayout()
        self.add_spectrum_btn = QPushButton("Add Spectrum and Linelist (.dat/.hdr)")
        self.add_spectrum_btn.clicked.connect(self.add_spectrum_pair)
        file_buttons.addWidget(self.add_spectrum_btn)
        
        self.add_csv_btn = QPushButton("Add CSV/Text File")
        self.add_csv_btn.clicked.connect(self.add_csv_file)
        file_buttons.addWidget(self.add_csv_btn)
        
        self.add_linelist_btn = QPushButton("Add Linelist to Existing Spectrum")
        self.add_linelist_btn.clicked.connect(self.add_standalone_linelist)
        file_buttons.addWidget(self.add_linelist_btn)
        
        self.remove_file_btn = QPushButton("Remove Selected")
        self.remove_file_btn.clicked.connect(self.remove_selected_file)
        file_buttons.addWidget(self.remove_file_btn)
        
        files_layout.addLayout(file_buttons)
        
        # File list widget
        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
        files_layout.addWidget(self.file_list)
        
        left_layout.addWidget(files_group)
        splitter.addWidget(left_panel)
        
        # Right panel - File preview and configuration
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Preview tab
        self.preview_widget = FilePreviewWidget()
        self.tab_widget.addTab(self.preview_widget, "File Preview")
        
        # Configuration tab
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # Data type selection
        type_group = QGroupBox("Data Type")
        type_layout = QHBoxLayout(type_group)
        
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            "calculations", "levels", "identifications", "calibration", "linelist"
        ])
        self.data_type_combo.currentTextChanged.connect(self.on_data_type_changed)
        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.data_type_combo)
        
        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setPlaceholderText("Dataset name (optional)")
        type_layout.addWidget(QLabel("Name:"))
        type_layout.addWidget(self.dataset_name_edit)
        
        config_layout.addWidget(type_group)
        
        # Spectrum selector (for linelist files)
        self.spectrum_selector_group = QGroupBox("Target Spectrum")
        spectrum_selector_layout = QHBoxLayout(self.spectrum_selector_group)
        
        self.spectrum_combo = QComboBox()
        self.spectrum_combo.setPlaceholderText("Select spectrum for this linelist...")
        self.refresh_spectra_btn = QPushButton("Refresh")
        self.refresh_spectra_btn.clicked.connect(self.refresh_spectrum_list)
        
        spectrum_selector_layout.addWidget(QLabel("Spectrum:"))
        spectrum_selector_layout.addWidget(self.spectrum_combo)
        spectrum_selector_layout.addWidget(self.refresh_spectra_btn)
        
        self.spectrum_selector_group.hide()  # Initially hidden
        config_layout.addWidget(self.spectrum_selector_group)
        
        # Column mapping
        self.column_mapping = ColumnMappingWidget()
        config_layout.addWidget(self.column_mapping)
        
        config_layout.addStretch()
        self.tab_widget.addTab(config_widget, "Configuration")
        
        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
        
        # Import controls
        import_group = QGroupBox("Import")
        import_layout = QVBoxLayout(import_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        import_layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        import_layout.addWidget(self.status_text)
        
        # Import buttons
        import_buttons = QHBoxLayout()
        
        self.import_btn = QPushButton("Start Import")
        self.import_btn.clicked.connect(self.start_import)
        import_buttons.addWidget(self.import_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_import)
        self.cancel_btn.setEnabled(False)
        import_buttons.addWidget(self.cancel_btn)
        
        import_buttons.addStretch()
        import_layout.addLayout(import_buttons)
        
        layout.addWidget(import_group)
    
    def setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_action = QAction('New HDF5 File', self)
        new_action.triggered.connect(self.new_hdf5_file)
        file_menu.addAction(new_action)
        
        open_action = QAction('Open HDF5 File', self)
        open_action.triggered.connect(self.browse_hdf5_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def browse_hdf5_file(self):
        """Browse for HDF5 file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select HDF5 File", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if filepath:
            self.hdf5_path = filepath
            self.hdf5_path_edit.setText(filepath)
            self.status_bar.showMessage(f"HDF5 file: {Path(filepath).name}")
    
    def new_hdf5_file(self):
        """Create new HDF5 file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Create New HDF5 File", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if filepath:
            # Ensure .h5 extension
            if not filepath.lower().endswith(('.h5', '.hdf5')):
                filepath += '.h5'
            
            self.hdf5_path = filepath
            self.hdf5_path_edit.setText(filepath)
            self.status_bar.showMessage(f"New HDF5 file: {Path(filepath).name}")
    
    def add_spectrum_pair(self):
        """Add spectrum .dat/.hdr file pair with optional linelist."""
        dat_file, _ = QFileDialog.getOpenFileName(
            self, "Select Spectrum Data File", "", "Data Files (*.dat);;All Files (*)"
        )
        if not dat_file:
            return
        
        # Look for corresponding .hdr file
        hdr_file = str(Path(dat_file).with_suffix('.hdr'))
        if not Path(hdr_file).exists():
            hdr_file, _ = QFileDialog.getOpenFileName(
                self, "Select Header File", str(Path(dat_file).parent), 
                "Header Files (*.hdr);;All Files (*)"
            )
            if not hdr_file:
                return
        
        # Ask for optional linelist file
        linelist_file, _ = QFileDialog.getOpenFileName(
            self, "Select Linelist File (Optional)", str(Path(dat_file).parent),
            "Linelist Files (*.aln *.lst *.wavcorr *.txt);;All Files (*)"
        )
        
        # Add to list
        file_info = {
            'path': dat_file,
            'hdr_file': hdr_file,
            'linelist_file': linelist_file if linelist_file else None,
            'type': 'spectrum_pair',
            'dataset_name': Path(dat_file).stem
        }
        self.files_to_import.append(file_info)
        
        # Add to list widget
        if linelist_file:
            item_text = f"Spectrum: {Path(dat_file).name} + {Path(hdr_file).name} + {Path(linelist_file).name}"
        else:
            item_text = f"Spectrum: {Path(dat_file).name} + {Path(hdr_file).name}"
        list_item = QListWidgetItem(item_text)
        list_item.setData(Qt.ItemDataRole.UserRole, len(self.files_to_import) - 1)
        self.file_list.addItem(list_item)
        
        status_msg = f"Added spectrum pair: {Path(dat_file).name}"
        if linelist_file:
            status_msg += f" with linelist: {Path(linelist_file).name}"
        self.status_bar.showMessage(status_msg)
    
    def add_csv_file(self):
        """Add CSV/text file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select CSV/Text File", "", 
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if not filepath:
            return
        
        # Add to list
        file_info = {
            'path': filepath,
            'type': 'calculations',  # Default type
            'dataset_name': Path(filepath).stem,
            'column_mapping': {}
        }
        self.files_to_import.append(file_info)
        
        # Add to list widget
        item_text = f"CSV: {Path(filepath).name}"
        list_item = QListWidgetItem(item_text)
        list_item.setData(Qt.ItemDataRole.UserRole, len(self.files_to_import) - 1)
        self.file_list.addItem(list_item)
        
        self.status_bar.showMessage(f"Added CSV file: {Path(filepath).name}")
    
    def add_standalone_linelist(self):
        """Add standalone linelist file to be associated with an existing spectrum."""
        # First, refresh the spectrum list to ensure it's up to date
        self.refresh_spectrum_list()
        
        if self.spectrum_combo.count() == 0:
            QMessageBox.warning(self, "No Spectra Available", 
                              "No spectra found in the HDF5 file. Please add spectra first or create an HDF5 file.")
            return
        
        # Select linelist file
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Linelist File", "", 
            "Linelist Files (*.aln *.lst *.wavcorr *.txt);;All Files (*)"
        )
        if not filepath:
            return
        
        # Show spectrum selector dialog
        spectrum_name, ok = self.select_target_spectrum()
        if not ok or not spectrum_name:
            return
        
        # Add to list
        file_info = {
            'path': filepath,
            'type': 'linelist',
            'target_spectrum': spectrum_name,
            'dataset_name': f"linelist_for_{spectrum_name}"
        }
        self.files_to_import.append(file_info)
        
        # Add to list widget
        item_text = f"Linelist: {Path(filepath).name} → {spectrum_name}"
        list_item = QListWidgetItem(item_text)
        list_item.setData(Qt.ItemDataRole.UserRole, len(self.files_to_import) - 1)
        self.file_list.addItem(list_item)
        
        self.status_bar.showMessage(f"Added linelist: {Path(filepath).name} for spectrum: {spectrum_name}")
    
    def remove_selected_file(self):
        """Remove selected file from list."""
        current_item = self.file_list.currentItem()
        if not current_item:
            return
        
        file_index = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Remove from list
        self.files_to_import.pop(file_index)
        self.file_list.takeItem(self.file_list.row(current_item))
        
        # Update indices
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            old_index = item.data(Qt.ItemDataRole.UserRole)
            if old_index > file_index:
                item.setData(Qt.ItemDataRole.UserRole, old_index - 1)
        
        self.status_bar.showMessage("File removed from import list")
    
    def refresh_spectrum_list(self):
        """Refresh the list of available spectra from the HDF5 file."""
        self.spectrum_combo.clear()
        
        if not self.hdf5_path or not Path(self.hdf5_path).exists():
            return
        
        try:
            with HDF5Manager(self.hdf5_path, 'r') as hdf5_mgr:
                datasets = hdf5_mgr.list_datasets('spectra')
                spectra_list = datasets.get('spectra', [])
                
                if spectra_list:
                    self.spectrum_combo.addItems(spectra_list)
                    self.spectrum_combo.setCurrentIndex(-1)  # No selection initially
                    print(f"Found {len(spectra_list)} spectra: {spectra_list}")
                else:
                    print("No spectra found in 'spectra' group")
                
        except Exception as e:
            print(f"Error reading spectra from HDF5 file: {e}")
            import traceback
            traceback.print_exc()
    
    def select_target_spectrum(self):
        """Show a dialog to select the target spectrum for a linelist."""
        if self.spectrum_combo.count() == 0:
            QMessageBox.warning(self, "No Spectra Available", 
                              "No spectra found in the HDF5 file.")
            return None, False
        
        # Create a simple selection dialog
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QComboBox, QPushButton, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Spectrum")
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select the spectrum to associate this linelist with:"))
        
        spectrum_selector = QComboBox()
        spectrum_selector.addItems([self.spectrum_combo.itemText(i) for i in range(self.spectrum_combo.count())])
        layout.addWidget(spectrum_selector)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return spectrum_selector.currentText(), True
        else:
            return None, False
    
    def on_file_selected(self):
        """Handle file selection in list."""
        current_item = self.file_list.currentItem()
        if not current_item:
            return
        
        file_index = current_item.data(Qt.ItemDataRole.UserRole)
        file_info = self.files_to_import[file_index]
        
        # Preview file
        self.preview_widget.preview_file(file_info['path'])
        
        # Update configuration tab
        if file_info['type'] != 'spectrum_pair':
            self.data_type_combo.setCurrentText(file_info['type'])
            self.dataset_name_edit.setText(file_info.get('dataset_name', ''))
            
            # Handle linelist files specially
            if file_info['type'] == 'linelist':
                self.spectrum_selector_group.show()
                self.refresh_spectrum_list()
                # Set the target spectrum if already selected
                target_spectrum = file_info.get('target_spectrum')
                if target_spectrum:
                    index = self.spectrum_combo.findText(target_spectrum)
                    if index >= 0:
                        self.spectrum_combo.setCurrentIndex(index)
            else:
                self.spectrum_selector_group.hide()
                # Setup column mapping for CSV files (not applicable for linelist)
                try:
                    result = FileParserFactory.parse_file(file_info['path'])
                    columns = result['columns']
                    self.column_mapping.setup_mapping(columns, file_info['type'])
                except:
                    pass
    
    def on_data_type_changed(self):
        """Handle data type change."""
        current_item = self.file_list.currentItem()
        if not current_item:
            # Show/hide spectrum selector based on type selection
            new_type = self.data_type_combo.currentText()
            if new_type == 'linelist':
                self.spectrum_selector_group.show()
                self.refresh_spectrum_list()
            else:
                self.spectrum_selector_group.hide()
            return
        
        file_index = current_item.data(Qt.ItemDataRole.UserRole)
        file_info = self.files_to_import[file_index]
        
        if file_info['type'] != 'spectrum_pair':
            new_type = self.data_type_combo.currentText()
            file_info['type'] = new_type
            
            # Show/hide spectrum selector for linelist type
            if new_type == 'linelist':
                self.spectrum_selector_group.show()
                self.refresh_spectrum_list()
            else:
                self.spectrum_selector_group.hide()
            
            # Update column mapping (not applicable for linelist type)
            if new_type != 'linelist':
                try:
                    result = FileParserFactory.parse_file(file_info['path'])
                    columns = result['columns']
                    self.column_mapping.setup_mapping(columns, new_type)
                except:
                    pass
    
    def start_import(self):
        """Start the import process."""
        if not self.hdf5_path:
            QMessageBox.warning(self, "Error", "Please select an HDF5 file first.")
            return
        
        if not self.files_to_import:
            QMessageBox.warning(self, "Error", "Please add files to import.")
            return
        
        # Update file configurations
        self._update_file_configurations()
        
        # Disable controls
        self.import_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Clear status
        self.status_text.clear()
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self.import_worker = FileImportWorker(self.files_to_import.copy(), self.hdf5_path)
        self.import_worker.progress_updated.connect(self.progress_bar.setValue)
        self.import_worker.status_updated.connect(self.append_status)
        self.import_worker.error_occurred.connect(self.show_error)
        self.import_worker.import_completed.connect(self.import_finished)
        self.import_worker.start()
        
        self.status_bar.showMessage("Import in progress...")
    
    def cancel_import(self):
        """Cancel the import process."""
        if self.import_worker:
            self.import_worker.cancel()
            self.import_worker.wait()
        
        self.import_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_bar.showMessage("Import cancelled")
    
    def import_finished(self, message: str):
        """Handle import completion."""
        self.import_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.append_status(message)
        self.status_bar.showMessage("Import completed")
        
        QMessageBox.information(self, "Import Complete", message)
    
    def show_error(self, error_message: str):
        """Show error message."""
        self.append_status(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Import Error", error_message)
    
    def append_status(self, message: str):
        """Append message to status text."""
        self.status_text.append(message)
        self.status_text.ensureCursorVisible()
    
    def _update_file_configurations(self):
        """Update file configurations from UI."""
        current_item = self.file_list.currentItem()
        if not current_item:
            return
        
        file_index = current_item.data(Qt.ItemDataRole.UserRole)
        file_info = self.files_to_import[file_index]
        
        if file_info['type'] != 'spectrum_pair':
            file_info['type'] = self.data_type_combo.currentText()
            file_info['dataset_name'] = self.dataset_name_edit.text() or None
            
            # Handle linelist configuration
            if file_info['type'] == 'linelist':
                file_info['target_spectrum'] = self.spectrum_combo.currentText()
            else:
                file_info['column_mapping'] = self.column_mapping.get_mapping()
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About HDF5 Import Wizard",
            "HDF5 Spectroscopy Import Wizard\n\n"
            "A tool for importing various spectroscopy file formats into HDF5.\n"
            "Supports .dat/.hdr spectrum pairs and CSV/text tabular data.\n\n"
            "Built with PyQt6 and h5py."
        )


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("HDF5 Import Wizard")
    
    # Set application style
    app.setStyle('Fusion')
    
    wizard = ImportWizard()
    wizard.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()