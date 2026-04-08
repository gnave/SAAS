# gui.py (FULLY DOCUMENTED)

import sys
import os
import pandas as pd
import h5py
from datetime import date
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QTableView, QRadioButton, QLineEdit, QCheckBox, QComboBox,
    QFormLayout, QLabel, QDialogButtonBox, QMessageBox, QWidget, QTextEdit,
    QTreeView, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QMenu, QAction
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon, QDoubleValidator
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex

import numpy as np

# Matplotlib integration for plotting within the PyQt5 GUI
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Import other project modules
from analysis_window import AnalysisWindow
import importers
import h5_manager
import analysis

#==============================================================================
# A model to display a Pandas DataFrame in a QTableView
#==============================================================================
class PandasModel(QAbstractTableModel):
    """
    A Qt Abstract Table Model specifically designed to display a Pandas DataFrame.
    This acts as a bridge between the DataFrame's structure and the QTableView's
    display requirements.
    """
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        """Returns the number of rows in the DataFrame."""
        return self._data.shape[0]

    def columnCount(self, parent=None):
        """Returns the number of columns in the DataFrame."""
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        """
        Retrieves the data for a specific cell.
        Provides basic formatting for floating-point numbers.
        """
        if index.isValid() and role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (float, np.floating)):
                if pd.isna(value):
                    return ""
                return f"{value:.4f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Retrieves the header (column or row index name) for a given section."""
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None

class FullTableWindow(QDialog):
    """A standalone window for viewing and searching an entire dataset."""
    def __init__(self, df, title, parent=None):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle(f"Full Data View: {title}")
        self.setMinimumSize(1000, 600)
        
        layout = QVBoxLayout(self)
        
        # Search Bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search/Filter:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to filter rows...")
        self.search_edit.textChanged.connect(self._apply_filter)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)
        
        # Table View
        self.table_view = QTableView()
        self.model = PandasModel(df)
        self.table_view.setModel(self.model)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout.addWidget(self.table_view)
        
        # Footer
        self.status_label = QLabel(f"Showing {len(df)} rows")
        layout.addWidget(self.status_label)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _apply_filter(self, text):
        """Simple search filter across all columns."""
        if not text:
            self.table_view.setModel(PandasModel(self.df))
            self.status_label.setText(f"Showing {len(self.df)} rows")
            return
        
        # Filter rows where any column contains the search text
        mask = self.df.apply(lambda row: row.astype(str).str.contains(text, case=False).any(), axis=1)
        filtered_df = self.df[mask]
        self.table_view.setModel(PandasModel(filtered_df))
        self.status_label.setText(f"Matches: {len(filtered_df)} / {len(self.df)}")

#==============================================================================
# Standalone Dialog Windows for User Input
#==============================================================================

class NewProjectDialog(QDialog):
    """A dialog window for creating a new project, collecting essential metadata."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setWindowFlags(Qt.Window)
        layout = QFormLayout(self)
        self.title_edit = QLineEdit()
        self.author_edit = QLineEdit()
        self.institution_edit = QLineEdit()
        self.supervisor_edit = QLineEdit()
        self.supervisor_edit.setPlaceholderText("(Optional)")
        layout.addRow("Project Title:", self.title_edit)
        layout.addRow("Author:", self.author_edit)
        layout.addRow("Institution:", self.institution_edit)
        layout.addRow("Supervisor:", self.supervisor_edit)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def get_data(self):
        """Returns a dictionary of the entered project metadata."""
        return {
            'project_title': self.title_edit.text().strip(),
            'author': self.author_edit.text().strip(),
            'institution': self.institution_edit.text().strip(),
            'supervisor': self.supervisor_edit.text().strip()
        }

    def accept(self):
        """Validates that required fields are filled before closing the dialog."""
        if not self.title_edit.text().strip() or not self.author_edit.text().strip():
            QMessageBox.warning(self, "Missing Information", "Project Title and Author are required fields.")
            return
        super().accept()

class ImportSpectrumDialog(QDialog):
    """A dialog for importing a primary spectrum (a .raw/.dat data file and a .hdr header file)."""
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Main Spectrum")
        self.setWindowFlags(Qt.Window); self.setMinimumWidth(500)
        layout = QFormLayout(self)
        self.data_file_edit = QLineEdit()
        browse_data_btn = QPushButton("Browse..."); browse_data_btn.clicked.connect(self._browse_data_file)
        data_layout = QHBoxLayout(); data_layout.addWidget(self.data_file_edit); data_layout.addWidget(browse_data_btn)
        self.header_file_edit = QLineEdit()
        browse_header_btn = QPushButton("Browse..."); browse_header_btn.clicked.connect(self._browse_header_file)
        header_layout = QHBoxLayout(); header_layout.addWidget(self.header_file_edit); header_layout.addWidget(browse_header_btn)
        layout.addRow("Spectrum Data File (*.raw, *.dat):", data_layout)
        layout.addRow("Spectrum Header File (*.hdr):", header_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _browse_data_file(self):
        """Opens a file dialog to select the spectrum data file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "Data Files (*.raw *.dat);;All Files (*)")
        if filepath: self.data_file_edit.setText(filepath)

    def _browse_header_file(self):
        """Opens a file dialog to select the spectrum header file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Header File", "", "Header Files (*.hdr);;All Files (*)")
        if filepath: self.header_file_edit.setText(filepath)

    def accept(self):
        """Validates input and calls the importer function from the importers module."""
        data_file = self.data_file_edit.text().strip()
        header_file = self.header_file_edit.text().strip()
        if not data_file or not header_file:
            QMessageBox.warning(self, "Missing Files", "Please select both a data file and a header file.")
            return
        try:
            importers.import_spectrum_pair(self.h5_filepath, data_file, header_file)
            QMessageBox.information(self, "Success", f"Main spectrum imported successfully.")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{e}")

class ImportCalibSpectrumDialog(QDialog):
    """A dialog for importing a calibration spectrum and associating it with a main spectrum."""
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Calibration Spectrum")
        self.setWindowFlags(Qt.Window); self.setMinimumWidth(500)
        layout = QFormLayout(self)
        self.spectrum_combo = QComboBox(); self._populate_spectrum_combo()
        self.data_file_edit = QLineEdit()
        browse_data_btn = QPushButton("Browse..."); browse_data_btn.clicked.connect(self._browse_data_file)
        data_layout = QHBoxLayout(); data_layout.addWidget(self.data_file_edit); data_layout.addWidget(browse_data_btn)
        self.header_file_edit = QLineEdit()
        browse_header_btn = QPushButton("Browse..."); browse_header_btn.clicked.connect(self._browse_header_file)
        header_layout = QHBoxLayout(); header_layout.addWidget(self.header_file_edit); header_layout.addWidget(browse_header_btn)
        layout.addRow("Associate with Main Spectrum:", self.spectrum_combo)
        layout.addRow("Calibration Data File (*.raw, *.dat):", data_layout)
        layout.addRow("Calibration Header File (*.hdr):", header_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _populate_spectrum_combo(self):
        """Fills the dropdown with the names of existing main spectra."""
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Spectra' in f:
                    spectra_names = list(f['/Spectra'].keys())
                    self.spectrum_combo.addItems(spectra_names)
        except Exception as e:
            print(f"Error reading spectra from HDF5 file: {e}")

    def _browse_data_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Calibration Data File", "", "Data Files (*.raw *.dat);;All Files (*)")
        if filepath: self.data_file_edit.setText(filepath)

    def _browse_header_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Calibration Header File", "", "Header Files (*.hdr);;All Files (*)")
        if filepath: self.header_file_edit.setText(filepath)

    def accept(self):
        """Validates input and calls the spectrum importer with calibration flags."""
        data_file = self.data_file_edit.text().strip()
        header_file = self.header_file_edit.text().strip()
        target_spectrum_name = self.spectrum_combo.currentText()
        if not data_file or not header_file or not target_spectrum_name:
            QMessageBox.warning(self, "Missing Information", "Please select a target spectrum and both calibration files.")
            return
        target_spectrum_group = f"/Spectra/{target_spectrum_name}"
        try:
            importers.import_spectrum_pair(
                self.h5_filepath, data_file, header_file, 
                is_calibration_spectrum=True, 
                target_spectrum_group=target_spectrum_group
            )
            QMessageBox.information(self, "Success", "Calibration spectrum imported successfully.")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{e}")

class ImportLampCalDialog(QDialog):
    """A dialog for importing a standard lamp calibration file and its metadata."""
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Lamp Calibration File")
        self.setWindowFlags(Qt.Window); self.setMinimumWidth(500)
        layout = QFormLayout(self)
        self.cal_file_edit = QLineEdit()
        browse_btn = QPushButton("Browse..."); browse_btn.clicked.connect(self._browse_file)
        file_layout = QHBoxLayout(); file_layout.addWidget(self.cal_file_edit); file_layout.addWidget(browse_btn)
        self.date_edit = QLineEdit(date.today().isoformat())
        self.author_edit = QLineEdit()
        self.notes_edit = QTextEdit(); self.notes_edit.setFixedHeight(60)
        layout.addRow("Calibration File (*.txt):", file_layout); layout.addRow(QLabel("-" * 60)); layout.addRow(QLabel("<b>Dataset Metadata:</b>"))
        layout.addRow("Date:", self.date_edit); layout.addRow("Author:", self.author_edit); layout.addRow("Notes/Comments:", self.notes_edit)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Lamp Calibration File", "", "Text Files (*.txt);;All Files (*)")
        if filepath: self.cal_file_edit.setText(filepath)

    def accept(self):
        """Collects metadata and calls the lamp calibration importer."""
        cal_file = self.cal_file_edit.text().strip()
        if not cal_file:
            QMessageBox.warning(self, "Missing File", "Please select a calibration file."); return
        user_metadata = {
            'import_date': self.date_edit.text(),
            'author': self.author_edit.text().strip(),
            'notes': self.notes_edit.toPlainText().strip()
        }
        try:
            importers.import_lamp_calibration(self.h5_filepath, cal_file, user_metadata)
            QMessageBox.information(self, "Success", "Lamp calibration imported successfully.")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{e}")

class ImportLinelistDialog(QDialog):
    """A dialog for importing a raw, binary linelist (.lin) file."""
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Binary Linelist")
        self.setWindowFlags(Qt.Window); self.setMinimumWidth(500)
        layout = QFormLayout(self)
        self.spectrum_combo = QComboBox(); self._populate_spectrum_combo()
        self.lin_file_edit = QLineEdit()
        browse_btn = QPushButton("Browse..."); browse_btn.clicked.connect(self._browse_lin_file)
        file_layout = QHBoxLayout(); file_layout.addWidget(self.lin_file_edit); file_layout.addWidget(browse_btn)
        layout.addRow("Target Spectrum:", self.spectrum_combo)
        layout.addRow("Linelist File (*.lin):", file_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _populate_spectrum_combo(self):
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Spectra' in f:
                    self.spectrum_combo.addItems(list(f['/Spectra'].keys()))
        except Exception as e:
            print(f"Error reading spectra from HDF5 file: {e}")

    def _browse_lin_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Linelist File", "", "Linelist Files (*.lin);;All Files (*)")
        if filepath: self.lin_file_edit.setText(filepath)

    def accept(self):
        """Validates input and calls the binary linelist importer."""
        lin_file = self.lin_file_edit.text().strip()
        target_spectrum_name = self.spectrum_combo.currentText()
        if not lin_file or not target_spectrum_name:
            QMessageBox.warning(self, "Missing Information", "Please select a target spectrum and a .lin file.")
            return
        target_spectrum_group = f"/Spectra/{target_spectrum_name}"
        try:
            importers.import_binary_linelist(self.h5_filepath, lin_file, target_spectrum_group)
            QMessageBox.information(self, "Success", "Binary linelist imported successfully.")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{e}")

class ImportCalibratedLinelistDialog(QDialog):
    """A dialog for importing a calibrated, text-based linelist (.txt) file."""
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Calibrated Text Linelist")
        self.setWindowFlags(Qt.Window); self.setMinimumWidth(500)
        layout = QFormLayout(self)
        self.spectrum_combo = QComboBox(); self._populate_spectrum_combo()
        self.txt_file_edit = QLineEdit()
        browse_btn = QPushButton("Browse..."); browse_btn.clicked.connect(self._browse_file)
        file_layout = QHBoxLayout(); file_layout.addWidget(self.txt_file_edit); file_layout.addWidget(browse_btn)
        layout.addRow("Target Spectrum:", self.spectrum_combo)
        layout.addRow("Calibrated Linelist File (*.txt):", file_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _populate_spectrum_combo(self):
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Spectra' in f: self.spectrum_combo.addItems(list(f['/Spectra'].keys()))
        except Exception as e: print(f"Error reading spectra from HDF5 file: {e}")

    def _browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Calibrated Linelist", "", "Text Files (*.txt);;All Files (*)")
        if filepath: self.txt_file_edit.setText(filepath)

    def accept(self):
        """Validates input and calls the calibrated linelist importer."""
        txt_file = self.txt_file_edit.text().strip()
        target_spectrum_name = self.spectrum_combo.currentText()
        if not txt_file or not target_spectrum_name:
            QMessageBox.warning(self, "Missing Information", "Please select a target spectrum and a text file.")
            return
        target_spectrum_group = f"/Spectra/{target_spectrum_name}"
        try:
            importers.import_calibrated_linelist(self.h5_filepath, txt_file, target_spectrum_group)
            QMessageBox.information(self, "Success", "Calibrated linelist imported successfully.")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{e}")

class ImportWizardDialog(QDialog):
    """
    A comprehensive wizard for importing generic text files (e.g., .csv, .txt, .dat).

    This dialog allows the user to specify parsing options (delimiter, fixed-width),
    preview the parsed data, map the source columns to a predefined schema in the
    HDF5 file, and provide metadata for the new dataset.
    """
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.df_preview = pd.DataFrame()
        self.setWindowTitle("Text File Import Wizard"); self.setMinimumSize(800, 700); self.setWindowFlags(Qt.Window)
        main_layout = QVBoxLayout(self); self.form_layout = QFormLayout()
        self.filepath_edit = QLineEdit(); browse_btn = QPushButton("Browse..."); browse_btn.clicked.connect(self._browse_file)
        file_layout = QHBoxLayout(); file_layout.addWidget(self.filepath_edit); file_layout.addWidget(browse_btn)
        self.delimited_radio = QRadioButton("Delimited"); self.delimited_radio.setChecked(True)
        self.fixed_radio = QRadioButton("Fixed Width")
        self.delimiter_combo = QComboBox(); self.delimiter_combo.addItems(["Comma (,)", "Space", "Tab"])
        self.fixed_widths_edit = QLineEdit(); self.fixed_widths_edit.setPlaceholderText("e.g., 12, 8, 10, 5"); self.fixed_widths_edit.setEnabled(False)
        self.header_checkbox = QCheckBox("First row is a header")
        self.preview_table = QTableView()
        self.group_combo = QComboBox()
        self.mapping_layout = QHBoxLayout()
        self.table_name_edit = QLineEdit()
        self.orig_filename_label = QLabel("N/A"); self.date_edit = QLineEdit(date.today().isoformat()); self.author_edit = QLineEdit()
        self.notes_edit = QTextEdit(); self.notes_edit.setFixedHeight(80)
        self.form_layout.addRow("Source File:", file_layout); self.form_layout.addRow("File Type:", self.delimited_radio); self.form_layout.addRow("", self.fixed_radio)
        self.form_layout.addRow("Delimiter:", self.delimiter_combo); self.form_layout.addRow("Column Widths:", self.fixed_widths_edit); self.form_layout.addRow(self.header_checkbox)
        self.form_layout.addRow(QLabel("Data Preview (first 100 rows):"), self.preview_table); self.form_layout.addRow("Destination Group:", self.group_combo)
        self.form_layout.addRow(QLabel("Assign Columns to Schema Fields:"), self.mapping_layout); self.form_layout.addRow("HDF5 Table Name:", self.table_name_edit)
        self.form_layout.addRow(QLabel("-" * 80)); self.form_layout.addRow(QLabel("<b>Dataset Metadata:</b>")); self.form_layout.addRow("Original Filename:", self.orig_filename_label)
        self.form_layout.addRow("Date:", self.date_edit); self.form_layout.addRow("Author:", self.author_edit); self.form_layout.addRow("Notes/Comments:", self.notes_edit)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)
        main_layout.addLayout(self.form_layout); main_layout.addWidget(self.button_box)
        self.delimited_radio.toggled.connect(self._update_ui_state); self.filepath_edit.textChanged.connect(self._update_preview)
        self.delimiter_combo.currentIndexChanged.connect(self._update_preview); self.fixed_widths_edit.textChanged.connect(self._update_preview)
        self.header_checkbox.stateChanged.connect(self._update_preview); self.group_combo.currentIndexChanged.connect(self._on_group_selected)
        self._populate_group_combo(); self._update_ui_state()

    def _browse_file(self):
        """Opens a file dialog, then populates the table name and filename fields."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt *.csv *.dat);;All Files (*)")
        if filepath:
            self.filepath_edit.setText(filepath)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            sanitized_name = base_name.replace('.', '_').replace('-', '_')
            self.table_name_edit.setText(sanitized_name)
            self.orig_filename_label.setText(os.path.basename(filepath))

    def _populate_group_combo(self):
        """Fills the destination group dropdown with valid schema-defined groups."""
        group_paths = h5_manager.get_all_group_paths(self.h5_filepath)
        # Only allow importing to these specific top-level groups.
        input_groups = [p for p in group_paths if p in ['/Calculations', '/Levels', '/Previous_Identifications']]
        self.group_combo.addItems(input_groups)

    def _update_ui_state(self):
        """Enables/disables UI elements based on whether delimited or fixed-width is selected."""
        is_delimited = self.delimited_radio.isChecked()
        self.delimiter_combo.setEnabled(is_delimited)
        self.fixed_widths_edit.setEnabled(not is_delimited)
        self._update_preview()

    def _on_group_selected(self):
        """When a destination group is chosen, this reads its schema and sets up the column mapping UI."""
        selected_group = self.group_combo.currentText()
        if not selected_group: return
        schema_types = ["(ignore)"]
        with h5py.File(self.h5_filepath, 'r') as f:
            if selected_group in f and 'schema' in f[selected_group].attrs:
                schema_str = f[selected_group].attrs['schema']
                schema_types.extend(schema_str.split(','))
        self._setup_column_mapping(schema_types)

    def _setup_column_mapping(self, possible_types):
        """Dynamically creates a dropdown menu for each column in the preview table."""
        # Clear any existing mapping widgets.
        for i in reversed(range(self.mapping_layout.count())): 
            self.mapping_layout.itemAt(i).widget().setParent(None)
        self.mapping_combos = []
        for col_name in self.df_preview.columns:
            combo = QComboBox(); combo.addItems(possible_types)
            self.mapping_layout.addWidget(combo)
            self.mapping_combos.append(combo)

    def _update_preview(self):
        """
        Called whenever a parsing option changes. It re-parses the top 100 lines
        of the file and updates the preview table.
        """
        filepath = self.filepath_edit.text()
        if not os.path.exists(filepath): self.df_preview = pd.DataFrame()
        else:
            file_type = 'delimited' if self.delimited_radio.isChecked() else 'fixed'
            delimiter = self.delimiter_combo.currentText().split()[0].lower()
            has_header = self.header_checkbox.isChecked()
            col_widths = None
            if file_type == 'fixed':
                try:
                    if self.fixed_widths_edit.text():
                        col_widths = [int(w.strip()) for w in self.fixed_widths_edit.text().split(',')]
                except ValueError: col_widths = None
            self.df_preview = importers.parse_generic_text_file(
                filepath, file_type, delimiter, has_header, col_widths
            ).head(100)
        self.preview_table.setModel(PandasModel(self.df_preview))
        self._on_group_selected()

    def accept(self):
        """
        The main logic for the importer. It parses the entire file, applies column
        mappings, performs data type conversions, and saves the final DataFrame.
        """
        table_name = self.table_name_edit.text().strip(); group_path = self.group_combo.currentText(); filepath = self.filepath_edit.text()
        if not all([table_name, group_path, filepath]):
            QMessageBox.warning(self, "Missing Information", "Please provide all required fields."); return

        file_type = 'delimited' if self.delimited_radio.isChecked() else 'fixed'; delimiter = self.delimiter_combo.currentText().split()[0].lower()
        has_header = self.header_checkbox.isChecked(); col_widths = None
        if file_type == 'fixed':
            try:
                if self.fixed_widths_edit.text(): col_widths = [int(w.strip()) for w in self.fixed_widths_edit.text().split(',')]
            except ValueError: QMessageBox.critical(self, "Error", "Invalid column widths."); return
        
        full_df = importers.parse_generic_text_file(filepath, file_type, delimiter, has_header, col_widths)
        final_df = pd.DataFrame()
        for i, combo in enumerate(self.mapping_combos):
            col_type = combo.currentText()
            if col_type != "(ignore)":
                final_df[col_type] = full_df.iloc[:, i]
        
        numeric_cols = ['j_value', 'energy', 'parity', 'lifetime', 'lifetime_unc_frac', 'wavenumber', 'wavelength', 'intensity', 'lower_level_energy', 'upper_level_energy', 'log_gf', 'transition_probability', 'lower_level_j', 'upper_level_j', 'snr', 'epstot', 'epsran']
        
        # --- Data Type Conversion Logic ---
        for col in final_df.columns:
            # DEFINITIVE FIX: When importing Previous IDs, wavenumber and intensity must be
            # preserved as strings to maintain original formatting.
            if group_path == '/Previous_Identifications' and col in ['wavenumber', 'intensity']:
                final_df[col] = final_df[col].astype(str)
                continue

            # For all other cases, convert known numeric columns to numbers.
            if col in numeric_cols:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            else:
                final_df[col] = final_df[col].astype(str)
                
        metadata_to_save = {'original_filename': self.orig_filename_label.text(), 'import_date': self.date_edit.text(), 'author': self.author_edit.text().strip(), 'notes': self.notes_edit.toPlainText().strip()}
        h5_manager.add_pandas_table(self.h5_filepath, group_path, table_name, final_df, metadata_dict=metadata_to_save)
        QMessageBox.information(self, "Success", f"Table '{table_name}' imported successfully.")
        super().accept()

class WavenumberMatchDialog(QDialog):
    """A dialog to gather inputs for running the wavenumber matching analysis."""
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Run Wavenumber Matching"); self.setWindowFlags(Qt.Window); self.setMinimumWidth(500)
        layout = QFormLayout(self)
        self.exp_linelist_combo = QComboBox(); self.prev_ids_combo = QComboBox(); self._populate_combos()
        self.tolerance_edit = QLineEdit("0.02"); self.tolerance_edit.setValidator(QDoubleValidator(0.0, 10.0, 3, self))
        self.output_name_edit = QLineEdit()
        layout.addRow("Experimental Linelist:", self.exp_linelist_combo); layout.addRow("Previous Identifications Table:", self.prev_ids_combo)
        layout.addRow("Tolerance (cm⁻¹):", self.tolerance_edit); layout.addRow("Output Dataset Name:", self.output_name_edit)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        self.exp_linelist_combo.currentIndexChanged.connect(self._suggest_output_name); self._suggest_output_name()

    def _populate_combos(self):
        """Fills the dropdowns with available experimental and identification tables."""
        linelists, prev_ids = [], []
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Spectra' in f:
                    for spec_group_name in f['/Spectra'].keys():
                        spec_group_path = f'/Spectra/{spec_group_name}'
                        if 'Raw_Linelists' in f[spec_group_path]:
                            for table_group_name in f[f'{spec_group_path}/Raw_Linelists'].keys(): linelists.append(f'{spec_group_path}/Raw_Linelists/{table_group_name}/table')
                        if 'Calibrated_Linelists' in f[spec_group_path]:
                            for table_group_name in f[f'{spec_group_path}/Calibrated_Linelists'].keys(): linelists.append(f'{spec_group_path}/Calibrated_Linelists/{table_group_name}/table')
                if '/Previous_Identifications' in f:
                    for table_group_name in f['/Previous_Identifications'].keys(): prev_ids.append(f'/Previous_Identifications/{table_group_name}/table')
            self.exp_linelist_combo.addItems(linelists); self.prev_ids_combo.addItems(prev_ids)
        except Exception as e: print(f"Error reading tables from HDF5 file: {e}")

    def _suggest_output_name(self):
        """Auto-generates a suggested name for the output dataset."""
        current_text = self.exp_linelist_combo.currentText()
        if current_text:
            base_name = current_text.split('/')[-3]; today = date.today().strftime("%Y%m%d")
            self.output_name_edit.setText(f"matched_{base_name}_{today}")

    def accept(self):
        """Validates input and calls the analysis function for wavenumber matching."""
        exp_path = self.exp_linelist_combo.currentText(); ids_path = self.prev_ids_combo.currentText()
        output_name = self.output_name_edit.text().strip()
        try:
            tolerance = float(self.tolerance_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Tolerance must be a valid number."); return
        if not all([exp_path, ids_path, output_name]):
            QMessageBox.warning(self, "Missing Information", "Please select all inputs and provide an output name."); return
        try:
            num_matches = analysis.run_and_save_wavenumber_match(self.h5_filepath, exp_path, ids_path, tolerance, output_name)
            QMessageBox.information(self, "Success", f"Wavenumber matching complete.\nFound and saved {num_matches} matches.")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during matching:\n{e}")   

#==============================================================================
# Main Application Window
#==============================================================================
class MainWindow(QMainWindow):
    """
    The main window of the SAAS application.

    This window serves as the central hub for all project operations. It contains:
    - A toolbar for creating/opening projects and importing all supported file types.
    - A tree view for visualizing the hierarchical structure of the open HDF5 project file.
    - A tabbed pane for previewing data, viewing plots, and inspecting metadata.
    - A button to launch the separate, specialized Analysis Window.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAAS - Spectroscopy Data Manager")
        self.setMinimumSize(1000, 800)
        self.current_h5_file = None
        
        # --- UI Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create the Menu Bar instead of buttons
        self._setup_menus()
        
        # Main splitter divides the window into the HDF5 tree (left) and data preview (right)
        splitter = QSplitter(Qt.Horizontal)
        self.tree_view = QTreeView()
        self.tree_model = QStandardItemModel()
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._show_tree_context_menu)
        splitter.addWidget(self.tree_view)
        
        # Right-hand side is a tab widget
        self.tabs = QTabWidget()
        data_preview_widget = QWidget()
        self.data_preview_layout = QVBoxLayout(data_preview_widget)
        self.data_table_view = QTableView()
        self.plot_figure = Figure()
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        self.plot_axes = self.plot_figure.add_subplot(111)
        self.data_preview_layout.addWidget(self.data_table_view)
        self.data_preview_layout.addWidget(self.plot_toolbar)
        self.data_preview_layout.addWidget(self.plot_canvas)
        self.data_table_view.hide(); self.plot_toolbar.hide(); self.plot_canvas.hide()
        
        self.attr_view = QTableWidget()
        self.attr_view.setColumnCount(2)
        self.attr_view.setHorizontalHeaderLabels(["Attribute Name", "Value"])
        self.tabs.addTab(data_preview_widget, "Data Preview")
        self.tabs.addTab(self.attr_view, "Attributes / Metadata")
        splitter.addWidget(self.tabs)
        
        splitter.setSizes([300, 700])
        main_layout.addWidget(splitter)
        
        # --- Initialize ---
        self.set_file_loaded_state(False)
        self.tree_view.clicked.connect(self._on_tree_item_selected)
        
    def set_file_loaded_state(self, is_loaded):
        """Enables or disables UI actions based on whether a project file is open."""
        # Enable/Disable the top-level menus
        self.import_menu.setEnabled(is_loaded)
        self.analysis_menu.setEnabled(is_loaded)
        
        # Explicitly setting individual actions (optional, since menu is disabled)
        self.import_spectrum_action.setEnabled(is_loaded)
        self.import_calib_spec_action.setEnabled(is_loaded)
        self.import_lamp_cal_action.setEnabled(is_loaded)
        self.import_table_action.setEnabled(is_loaded)
        self.import_linelist_action.setEnabled(is_loaded)
        self.import_cal_linelist_action.setEnabled(is_loaded)
        self.run_match_action.setEnabled(is_loaded)
        self.run_bf_action.setEnabled(is_loaded)

    def _create_file(self):
        """Launches the new project dialog and creates a new HDF5 file."""
        project_dialog = NewProjectDialog(self)
        if project_dialog.exec_():
            project_metadata = project_dialog.get_data()
            filepath, _ = QFileDialog.getSaveFileName(self, "Save New HDF5 Project File", "", "HDF5 Files (*.h5 *.hdf5)")
            if filepath:
                h5_manager.create_experiment_file(filepath, project_metadata)
                self.set_current_file(filepath)

    def _open_file(self):
        """Launches a file dialog to open an existing HDF5 project file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Open HDF5 File", "", "HDF5 Files (*.h5 *.hdf5)")
        if filepath:
            self.set_current_file(filepath)

    def set_current_file(self, filepath):
        """Sets the active project file and refreshes the main UI."""
        self.current_h5_file = filepath
        
        # Update the Window Title instead of a label
        self.setWindowTitle(f"SAAS - Project: {filepath}")
        
        self.set_file_loaded_state(True)
        self._populate_tree_view()

    def _populate_tree_view(self):
        """Clears and re-populates the HDF5 tree view based on the current file's structure."""
        self.tree_model.clear()
        if not self.current_h5_file: return
        try:
            with h5py.File(self.current_h5_file, 'r') as f:
                root_item = self.tree_model.invisibleRootItem()
                self._add_items_to_tree_recursively(root_item, f)
        except Exception as e:
            QMessageBox.critical(self, "Error Reading File", f"Could not read HDF5 structure:\n{e}")

    def _add_items_to_tree_recursively(self, parent_item, h5_object):
        """A recursive helper function to build the QStandardItemModel for the tree view."""
        for name, item in h5_object.items():
            if isinstance(item, h5py.Group):
                child_item = QStandardItem(QIcon.fromTheme("folder"), name)
            elif isinstance(item, h5py.Dataset):
                child_item = QStandardItem(QIcon.fromTheme("text-x-generic"), name)
            else:
                child_item = QStandardItem(name)
            # Store the full HDF5 path in the item's UserRole for easy access later.
            child_item.setData(item.name, Qt.UserRole)
            parent_item.appendRow(child_item)
            if isinstance(item, h5py.Group):
                self._add_items_to_tree_recursively(child_item, item)

    def _on_tree_item_selected(self, index):
        """
        The primary slot for handling user interaction with the HDF5 tree.

        When an item is clicked, this function determines what it is (a group,
        a pandas table, a spectrum dataset) and updates the right-hand tab pane
        accordingly, showing either a table preview, a plot, or just metadata.
        """
        h5_path = index.data(Qt.UserRole)
        if not h5_path: return

        # Reset all preview widgets.
        self.data_table_view.hide(); self.plot_toolbar.hide(); self.plot_canvas.hide()
        self.data_table_view.setModel(None)

        try:
            with h5py.File(self.current_h5_file, 'r') as f:
                h5_object = f[h5_path]

                # Always display the metadata/attributes for any selected object.
                self.attr_view.setRowCount(0)
                self.attr_view.setRowCount(len(h5_object.attrs))
                for i, (key, value) in enumerate(h5_object.attrs.items()):
                    self.attr_view.setItem(i, 0, QTableWidgetItem(str(key)))
                    self.attr_view.setItem(i, 1, QTableWidgetItem(str(value)))
                self.attr_view.resizeColumnsToContents()
                
                if isinstance(h5_object, h5py.Dataset):
                    # Case 1: The dataset is a pandas table (identified by attributes on its parent group).
                    if h5_object.parent and 'pandas_type' in h5_object.parent.attrs:
                        df = h5_manager.read_hdf_table_robustly(self.current_h5_file, h5_path)
                        self.data_table_view.setModel(PandasModel(df.head(200))) # Show first 200 rows.
                        self.data_table_view.show()
                    # Case 2: The dataset is a 1D array, which we assume is a spectrum to be plotted.
                    elif h5_object.ndim == 1:
                        self._plot_spectrum_data(h5_object)
                        self.plot_toolbar.show(); self.plot_canvas.show()
                        
        except Exception as e:
            print(f"Error accessing HDF5 object at path '{h5_path}': {e}")
            self.attr_view.setRowCount(0)

    def _plot_spectrum_data(self, h5_dataset):
        """Extracts data and metadata from a spectrum dataset and renders it on the plot canvas."""
        data = h5_dataset[:]
        attrs = h5_dataset.attrs
        # Get plotting parameters from dataset attributes, with safe defaults.
        wstart = attrs.get('wstart', 0.0); delw = attrs.get('delw', 1.0); rdsclfct = attrs.get('rdsclfct', 1.0)
        y = data * rdsclfct
        indices = np.arange(len(y))
        x = wstart + indices * delw
        self.plot_axes.clear()
        self.plot_axes.plot(x, y)
        self.plot_axes.set_xlabel("Wavenumber (cm⁻¹)"); self.plot_axes.set_ylabel("Intensity"); self.plot_axes.set_title(f"Spectrum Plot: {h5_dataset.name}")
        self.plot_axes.grid(True); self.plot_figure.tight_layout(); self.plot_canvas.draw()
            
    # --- Methods for launching dialogs ---
    def _show_table_import_wizard(self):
        if self.current_h5_file:
            dialog = ImportWizardDialog(self.current_h5_file, self)
            if dialog.exec_(): self._populate_tree_view()

    def _show_spectrum_import_dialog(self):
        if self.current_h5_file:
            dialog = ImportSpectrumDialog(self.current_h5_file, self)
            if dialog.exec_(): self._populate_tree_view()

    def _show_calib_spec_import_dialog(self):
        if self.current_h5_file:
            dialog = ImportCalibSpectrumDialog(self.current_h5_file, self)
            if dialog.exec_(): self._populate_tree_view()

    def _show_lamp_cal_import_dialog(self):
        if self.current_h5_file:
            dialog = ImportLampCalDialog(self.current_h5_file, self)
            if dialog.exec_(): self._populate_tree_view()

    def _show_linelist_import_dialog(self):
        if self.current_h5_file:
            dialog = ImportLinelistDialog(self.current_h5_file, self)
            if dialog.exec_(): self._populate_tree_view()

    def _show_cal_linelist_import_dialog(self):
        if self.current_h5_file:
            dialog = ImportCalibratedLinelistDialog(self.current_h5_file, self)
            if dialog.exec_(): self._populate_tree_view()

    def _show_match_dialog(self):
        if self.current_h5_file:
            dialog = WavenumberMatchDialog(self.current_h5_file, self)
            if dialog.exec_(): self._populate_tree_view()

    def _launch_branching_fraction_analysis(self):
        """Launches the dedicated, separate window for interactive analysis."""
        if self.current_h5_file:
            self.branching_fraction_analysis_window = AnalysisWindow(self.current_h5_file, self)
            self.branching_fraction_analysis_window.show()
            # When the analysis window is closed, refresh the tree to show any new saved results.
            self.branching_fraction_analysis_window.destroyed.connect(self._populate_tree_view)
        else:
            QMessageBox.warning(self, "No HDF5 File", "Please open an HDF5 project file first.")

    def _show_tree_context_menu(self, position):
        index = self.tree_view.indexAt(position)
        if not index.isValid(): return
        
        h5_path = index.data(Qt.UserRole)
        menu = QMenu()
        
        # Only show "View Full Table" if it's a dataset
        view_full_action = None
        with h5py.File(self.current_h5_file, 'r') as f:
            if isinstance(f[h5_path], h5py.Dataset):
                view_full_action = menu.addAction("View Full Table (Searchable)")
                menu.addSeparator()

        delete_action = menu.addAction("Delete Selected Item")
        
        action = menu.exec_(self.tree_view.viewport().mapToGlobal(position))
        
        if action == delete_action:
            self._delete_selected_item(index)
        elif action == view_full_action:
            self._open_full_table_viewer(h5_path)

    def _delete_selected_item(self, index):
        """Handles the logic for deleting an item from the HDF5 file."""
        h5_path = index.data(Qt.UserRole)
        if not h5_path or h5_path == '/':
            QMessageBox.warning(self, "Action Not Allowed", "The root of the file cannot be deleted."); return
        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Are you sure you want to permanently delete this item and all its contents?\n\n<b>{h5_path}</b>",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            success = h5_manager.delete_object(self.current_h5_file, h5_path)
            if success:
                self._populate_tree_view()
                self.data_table_view.setModel(None); self.attr_view.setRowCount(0)
            else:
                QMessageBox.critical(self, "Error", f"Failed to delete the item at {h5_path}.")

    def _setup_menus(self):
        menubar = self.menuBar()

        # --- FILE MENU ---
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Project...", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._create_file)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- IMPORT MENU ---
        self.import_menu = menubar.addMenu("&Import")
        
        # Submenu for Spectra
        spec_menu = self.import_menu.addMenu("Spectra")
        self.import_spectrum_action = QAction("Main Spectrum...", self)
        self.import_spectrum_action.triggered.connect(self._show_spectrum_import_dialog)
        self.import_calib_spec_action = QAction("Calibration Spectrum...", self)
        self.import_calib_spec_action.triggered.connect(self._show_calib_spec_import_dialog)
        spec_menu.addAction(self.import_spectrum_action)
        spec_menu.addAction(self.import_calib_spec_action)

        # Submenu for Linelists
        line_menu = self.import_menu.addMenu("Linelists")
        self.import_linelist_action = QAction("Raw Binary (.lin)...", self)
        self.import_linelist_action.triggered.connect(self._show_linelist_import_dialog)
        self.import_cal_linelist_action = QAction("Calibrated Text (.txt)...", self)
        self.import_cal_linelist_action.triggered.connect(self._show_cal_linelist_import_dialog)
        line_menu.addAction(self.import_linelist_action)
        line_menu.addAction(self.import_cal_linelist_action)

        self.import_table_action = QAction("Generic Table (Wizard)...", self)
        self.import_table_action.triggered.connect(self._show_table_import_wizard)
        self.import_menu.addAction(self.import_table_action)

        self.import_lamp_cal_action = QAction("Standard Lamp Calibration...", self)
        self.import_lamp_cal_action.triggered.connect(self._show_lamp_cal_import_dialog)
        self.import_menu.addAction(self.import_lamp_cal_action)

        # --- ANALYSIS MENU ---
        self.analysis_menu = menubar.addMenu("&Analysis")
        
        self.run_match_action = QAction("Wavenumber Matching...", self)
        self.run_match_action.triggered.connect(self._show_match_dialog)
        self.analysis_menu.addAction(self.run_match_action)
        
        self.run_bf_action = QAction("Interactive Branching Fraction Analysis...", self)
        self.run_bf_action.setShortcut("Ctrl+R")
        self.run_bf_action.triggered.connect(self._launch_branching_fraction_analysis)
        self.analysis_menu.addAction(self.run_bf_action)

    def _open_full_table_viewer(self, h5_path):
        """Loads the full dataset and opens it in the FullTableWindow."""
        try:
            df = h5_manager.read_hdf_table_robustly(self.current_h5_file, h5_path)
            # Create a modeless window so user can keep it open while browsing others
            self.full_view = FullTableWindow(df, h5_path, self)
            self.full_view.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load full table:\n{e}")