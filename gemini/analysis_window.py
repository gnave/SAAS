# analysis_window.py (MODIFIED to remove 'Include in Fit')

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableView, QTreeView, QSplitter, QDockWidget, QPushButton, QLineEdit,
    QAbstractItemView, QSizePolicy, QHeaderView, QMenuBar, QAction, QMessageBox,
    QDialog, QDialogButtonBox, QInputDialog, QFormLayout, QTextEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QMenu
)
from PyQt5.QtCore import Qt, QModelIndex, QAbstractTableModel, pyqtSignal, QItemSelectionModel
from PyQt5.QtGui import QColor, QFont, QStandardItemModel, QStandardItem, QIcon, QDoubleValidator, QBrush

import pandas as pd
import numpy as np
import h5py
import os
from datetime import date

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import h5_manager
import analysis

class PandasTableModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = data
    def rowCount(self, parent=QModelIndex()): return self.df.shape[0]
    def columnCount(self, parent=QModelIndex()): return self.df.shape[1]
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            value = self.df.iloc[index.row(), index.column()]
            if isinstance(value, (float, np.floating)):
                if pd.isna(value): return ""
                return f"{value:.4f}"
            return str(value)
        return None
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal: return str(self.df.columns[section])
            if orientation == Qt.Vertical: return str(self.df.index[section])
        return None

# --- MODIFICATION START: 'Include in Fit' logic is fully removed ---
class LineDataTableModel(PandasTableModel):
    """A simplified table model that removes all 'Include in Fit' logic."""
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(data, parent)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        
        value = self.df.iloc[index.row(), index.column()]
        col_name = str(self.df.columns[index.column()])
        
        if isinstance(value, (float, np.floating)):
            if pd.isna(value): return ""
            if '\nSNR' in col_name or '\nIntensity' in col_name:
                return f"{int(round(value))}"
            return f"{value:.4f}"
        return str(value)

    def flags(self, index: QModelIndex):
        # Just return the base flags, as no columns are checkable/editable.
        return super().flags(index)
# --- MODIFICATION END ---

class PlotPopupDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

class AnalysisWindow(QMainWindow):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Branching Fraction Analysis")
        self.setGeometry(100, 100, 1400, 800)
        self.extra_plot_windows = []
        self.h5_filepath = h5_filepath
        self.h5_manager = h5_manager
        self.analysis_module = analysis
        self.current_energy_levels_df = pd.DataFrame()
        self.current_previous_ids_df = pd.DataFrame()
        self.filtered_levels_df = pd.DataFrame()
        self.master_line_data_df = pd.DataFrame()
        self.result_df = pd.DataFrame()
        self.DATA_SOURCE_COLUMNS = {
            "Cal. Linelists": "Calibrated_Linelists",
            "Ident. Lines": "Identified_Lines",
            "Raw Spectrum": "Raw_Data"
        }
        self._create_menu_bar()
        self._create_main_layout()
        self._populate_initial_comboboxes()
        self._populate_data_source_table()
        self._clear_plot()
        self.main_splitter.setSizes([350, 1050])
        self.side_panel_splitter.setSizes([self.height() // 2, self.height() // 2])

    def _create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        exit_action = QAction("Exit", self); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)
        debug_menu = menubar.addMenu("&Debug"); run_diagnostics_action = QAction("Run Diagnostics...", self)
        run_diagnostics_action.triggered.connect(self._run_debug_diagnostics); debug_menu.addAction(run_diagnostics_action)
        help_menu = menubar.addMenu("&Help"); help_action = QAction("About", self)
        help_action.triggered.connect(lambda: QMessageBox.information(self, "About", "SAAS")); help_menu.addAction(help_action)

    def _create_main_layout(self):
        self.main_splitter = QSplitter(Qt.Horizontal)
        side_panel_widget = self._create_side_panel()
        central_content_widget = self._create_central_content_widget()
        self.main_splitter.addWidget(side_panel_widget)
        self.main_splitter.addWidget(central_content_widget)
        self.setCentralWidget(self.main_splitter)

    def _create_side_panel(self):
        self.side_panel_splitter = QSplitter(Qt.Vertical)
        
        level_selector_container = QWidget()
        level_selector_layout = QVBoxLayout(level_selector_container)
        
        data_source_container = QWidget()
        data_source_layout = QVBoxLayout(data_source_container)
        
        self.level_file_combo = QComboBox(); self.level_file_combo.addItem("Select Energy Level File...")
        self.level_file_combo.currentIndexChanged.connect(self._on_level_file_selected)
        level_selector_layout.addWidget(QLabel("Master Energy Level File:")); level_selector_layout.addWidget(self.level_file_combo)
        self.level_table = QTableView(); self.level_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.level_table.setSelectionMode(QAbstractItemView.SingleSelection); self.level_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.level_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive); self.level_table.clicked.connect(self._on_level_selected_in_table)
        level_selector_layout.addWidget(QLabel("Available Upper Levels:")); level_selector_layout.addWidget(self.level_table)
        header_height = self.level_table.horizontalHeader().height()
        row_height = self.level_table.verticalHeader().defaultSectionSize()
        self.level_table.setMaximumHeight(int(header_height + 5.5 * row_height))
        self.level_details_group = QWidget(); level_details_layout = QFormLayout(self.level_details_group)
        self.level_key_display, self.level_energy_display, self.level_j_display, self.level_parity_display, self.level_lifetime_display = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        for editor in [self.level_key_display, self.level_energy_display, self.level_j_display, self.level_parity_display, self.level_lifetime_display]: editor.setReadOnly(True)
        level_details_layout.addRow("key:", self.level_key_display); level_details_layout.addRow("energy (cm⁻¹):", self.level_energy_display)
        level_details_layout.addRow("j_value:", self.level_j_display); level_details_layout.addRow("parity:", self.level_parity_display)
        level_details_layout.addRow("lifetime (ns):", self.level_lifetime_display)
        level_selector_layout.addWidget(QLabel("Selected Level Details:")); level_selector_layout.addWidget(self.level_details_group)
        
        self.prev_id_combo = QComboBox(); self.prev_id_combo.addItem("Select Previous IDs File...")
        self.prev_id_combo.currentIndexChanged.connect(self._on_prev_id_file_selected)
        data_source_layout.addWidget(QLabel("Master Previous IDs File:")); data_source_layout.addWidget(self.prev_id_combo)
        data_source_layout.addWidget(QLabel("Select Data for Comparison/Plotting:"))
        self.data_source_table = QTableWidget(); self.data_source_table.itemChanged.connect(self._on_data_source_table_item_changed)
        data_source_layout.addWidget(self.data_source_table)
        self.analysis_controls_group = QWidget(); analysis_controls_layout = QVBoxLayout(self.analysis_controls_group)
        self.separate_plots_checkbox = QCheckBox("Plot Spectra in Separate Windows"); analysis_controls_layout.addWidget(self.separate_plots_checkbox)
        self.tolerance_edit = QLineEdit("0.02"); self.tolerance_edit.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        analysis_controls_layout.addWidget(QLabel("Wavenumber Matching Tolerance (cm⁻¹):")); analysis_controls_layout.addWidget(self.tolerance_edit)
        self.run_analysis_btn = QPushButton("Calculate Branching Fractions"); self.run_analysis_btn.clicked.connect(self._calculate_clicked)
        analysis_controls_layout.addWidget(self.run_analysis_btn)
        self.save_results_btn = QPushButton("Save Results to HDF5"); self.save_results_btn.clicked.connect(self._save_results_clicked)
        self.save_results_btn.setEnabled(False); analysis_controls_layout.addWidget(self.save_results_btn)
        data_source_layout.addWidget(self.analysis_controls_group)
        
        self.side_panel_splitter.addWidget(level_selector_container); self.side_panel_splitter.addWidget(data_source_container)
        return self.side_panel_splitter

    def _create_central_content_widget(self):
        self.central_splitter = QSplitter(Qt.Vertical)
        self.line_data_table = QTableView()
        self.line_data_table.setSelectionBehavior(QAbstractItemView.SelectRows); self.line_data_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.line_data_table.setAlternatingRowColors(True); self.line_data_table.clicked.connect(self._on_line_selected)
        self.line_data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.line_data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.line_data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.line_data_table.customContextMenuRequested.connect(self._show_line_table_context_menu)
        self.central_splitter.addWidget(self.line_data_table)
        main_plot_widget = QWidget()
        plot_layout = QVBoxLayout(main_plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 4), dpi=100); self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self); self.ax = self.figure.add_subplot(111)
        plot_layout.addWidget(self.toolbar); plot_layout.addWidget(self.canvas)
        self.central_splitter.addWidget(main_plot_widget)
        return self.central_splitter

    def _show_line_table_context_menu(self, position):
        index = self.line_data_table.indexAt(position)
        if not index.isValid():
            return
        menu = QMenu()
        normalize_action = menu.addAction("Set as Intensity Reference (Normalize to 1000)")
        action = menu.exec_(self.line_data_table.viewport().mapToGlobal(position))
        if action == normalize_action:
            self._normalize_intensities(index.row())

    def _normalize_intensities(self, reference_line_row: int):
        if self.master_line_data_df.empty:
            QMessageBox.warning(self, "Normalization Error", "No data loaded to normalize.")
            return
        try:
            normalized_df = self.analysis_module.normalize_intensities_by_reference_line(
                self.master_line_data_df,
                reference_line_row
            )
            self.master_line_data_df = normalized_df
            model = LineDataTableModel(self.master_line_data_df)
            self.line_data_table.setModel(model)
            
            new_index_to_select = model.index(reference_line_row, 0)
            if new_index_to_select.isValid():
                self.line_data_table.setCurrentIndex(new_index_to_select)
                selection_model = self.line_data_table.selectionModel()
                selection_model.select(new_index_to_select, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                self._on_line_selected(new_index_to_select)

            ref_level_key = self.master_line_data_df.iloc[reference_line_row].get('lower_level_key', 'Unknown')
            QMessageBox.information(self, "Success", 
                                    f"Intensities have been normalized using the line for lower level '{ref_level_key}' as the reference.")
        except Exception as e:
            QMessageBox.critical(self, "Normalization Error", f"An error occurred during normalization:\n{e}")

    def _populate_data_source_table(self):
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Spectra' not in f: self.data_source_table.clear(); return
                spectra_names, column_labels = sorted(list(f['/Spectra'].keys())), list(self.DATA_SOURCE_COLUMNS.keys())
                self.data_source_table.setRowCount(len(spectra_names)); self.data_source_table.setColumnCount(len(column_labels))
                self.data_source_table.setVerticalHeaderLabels(spectra_names); self.data_source_table.setHorizontalHeaderLabels(column_labels)
                for r, spectrum_name in enumerate(spectra_names):
                    for c, col_label in enumerate(column_labels):
                        hdf5_group_name = self.DATA_SOURCE_COLUMNS[col_label]
                        base_path = f"/Spectra/{spectrum_name}/{hdf5_group_name}"
                        item = QTableWidgetItem()
                        item.setFlags(item.flags() & ~Qt.ItemIsEnabled); item.setBackground(QBrush(QColor('lightGray')))
                        if base_path in f:
                            dset_path, item_text = "", ""
                            if hdf5_group_name == "Raw_Data":
                                dset_path = f"{base_path}/spectrum"
                                if dset_path in f: item_text = "Exists"
                            else:
                                sub_datasets = list(f[base_path].keys())
                                if sub_datasets:
                                    first_dset_name = sub_datasets[0]
                                    dset_path = f"{base_path}/{first_dset_name}/table"
                                    if dset_path in f: item_text = first_dset_name
                            if item_text and dset_path:
                                item.setText(""), item.setToolTip(dset_path), item.setData(Qt.UserRole, dset_path)
                                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                                item.setBackground(QBrush(QColor('white'))); item.setCheckState(Qt.Unchecked)
                        self.data_source_table.setItem(r, c, item)
            self.data_source_table.resizeColumnsToContents()
            self.data_source_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        except Exception as e:
            QMessageBox.critical(self, "HDF5 Scan Error", f"Failed to populate data source table: {e}")

    def _get_checked_data_paths(self):
        checked_paths = []
        for r in range(self.data_source_table.rowCount()):
            for c in range(self.data_source_table.columnCount()):
                item = self.data_source_table.item(r, c)
                if item and item.checkState() == Qt.Checked:
                    path = item.data(Qt.UserRole)
                    if path: checked_paths.append(path)
        return checked_paths
        
    def _populate_initial_comboboxes(self):
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Levels' in f:
                    self.level_file_combo.addItems([name for name in f['/Levels'].keys() if isinstance(f['/Levels'][name], h5py.Group)])
                if '/Previous_Identifications' in f:
                    self.prev_id_combo.addItems([name for name in f['/Previous_Identifications'].keys() if isinstance(f['/Previous_Identifications'][name], h5py.Group)])
        except Exception as e:
            QMessageBox.critical(self, "HDF5 Error", f"Failed to read HDF5 structure: {e}")

    def _on_level_file_selected(self):
        selected_file = self.level_file_combo.currentText()
        if selected_file == "Select Energy Level File...":
            self.level_table.setModel(None); self._clear_level_details(); self.current_energy_levels_df, self.filtered_levels_df = pd.DataFrame(), pd.DataFrame(); return
        path = f"/Levels/{selected_file}/table"
        try:
            self.current_energy_levels_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
            if not self.current_energy_levels_df.empty and 'key' in self.current_energy_levels_df.columns:
                self.current_energy_levels_df['key'] = self.current_energy_levels_df['key'].astype(str).str.replace('*', '', regex=False).str.strip()
            if not self.current_energy_levels_df.empty and 'lifetime' in self.current_energy_levels_df.columns:
                self.filtered_levels_df = self.current_energy_levels_df[self.current_energy_levels_df['lifetime'] > 0].copy()
                self.level_table.setModel(PandasTableModel(self.filtered_levels_df[['key', 'energy']]))
                self.level_table.resizeColumnsToContents()
            else:
                self.level_table.setModel(None); QMessageBox.warning(self, "Data Error", f"Table at {path} is empty or missing required columns.")
        except Exception as e:
            self.level_table.setModel(None); self.current_energy_levels_df, self.filtered_levels_df = pd.DataFrame(), pd.DataFrame()
            QMessageBox.critical(self, "HDF5 Read Error", f"Could not read energy levels from {path}:\n{e}")
        finally:
            self._clear_level_details()

    def _on_prev_id_file_selected(self):
        selected_file = self.prev_id_combo.currentText()
        if selected_file == "Select Previous IDs File...":
            self.current_previous_ids_df = pd.DataFrame(); self.line_data_table.setModel(None); self._clear_plot(); return
        path = f"/Previous_Identifications/{selected_file}/table"
        try:
            self.current_previous_ids_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
            if not self.current_previous_ids_df.empty and 'upper_level_key' in self.current_previous_ids_df.columns:
                self.current_previous_ids_df['normalized_key'] = self.current_previous_ids_df['upper_level_key'].astype(str).str.replace('*', '', regex=False).str.strip()
            if self.level_table.selectionModel() and self.level_table.selectionModel().hasSelection():
                self._on_level_selected_in_table()
        except Exception as e:
            self.current_previous_ids_df = pd.DataFrame(); QMessageBox.critical(self, "HDF5 Read Error", f"Could not read Previous IDs from {path}:\n{e}")
            self.line_data_table.setModel(None); self._clear_plot()
            
    def _on_level_selected_in_table(self):
        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes or self.filtered_levels_df.empty:
            self._clear_level_details(); self.line_data_table.setModel(None); self._clear_plot(); return
        row = selected_indexes[0].row()
        selected_level_data = self.filtered_levels_df.iloc[row]
        self.level_key_display.setText(str(selected_level_data.get('key', 'N/A')))
        self.level_energy_display.setText(f"{selected_level_data.get('energy', 0.0):.3f}")
        self.level_j_display.setText(str(selected_level_data.get('j_value', 'N/A')))
        self.level_parity_display.setText(str(selected_level_data.get('parity', 'N/A')))
        self.level_lifetime_display.setText(f"{selected_level_data.get('lifetime', 0.0):.3f}")
        self._populate_line_data_table(selected_level_data['key'])
        
    def _populate_line_data_table(self, upper_level_key: str):
        if self.current_previous_ids_df.empty:
            self.line_data_table.setModel(None); self._clear_plot(); return
        if 'normalized_key' not in self.current_previous_ids_df.columns: return
        lines_from_level = self.current_previous_ids_df[self.current_previous_ids_df['normalized_key'] == upper_level_key]
        if lines_from_level.empty:
            self.line_data_table.setModel(None); self._clear_plot(); return
        all_checked_paths = self._get_checked_data_paths()
        linelist_paths_to_merge = [p for p in all_checked_paths if ('Identified_Lines' in p or 'Calibrated_Linelists' in p)]
        try:
            df_to_pass = lines_from_level.drop(columns=['normalized_key'], errors='ignore')
            self.master_line_data_df = self.analysis_module.aggregate_observed_data_for_display(
                h5_filepath=self.h5_filepath, previous_ids_df=df_to_pass,
                linelist_paths=linelist_paths_to_merge, tolerance=float(self.tolerance_edit.text())
            )
            if self.master_line_data_df.empty:
                self.line_data_table.setModel(None); self._clear_plot(); return
            
            model = LineDataTableModel(self.master_line_data_df)
            self.line_data_table.setModel(model)
            self.line_data_table.horizontalHeader().setFixedHeight(40)
            self._clear_plot()
            current_height = self.central_splitter.height()
            self.central_splitter.setSizes([current_height // 2, current_height // 2])
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error in _populate_line_data_table: {e}")
            self.line_data_table.setModel(None); self._clear_plot()
            
    def _clear_level_details(self):
        self.level_key_display.clear(); self.level_energy_display.clear(); self.level_j_display.clear(); self.level_parity_display.clear(); self.level_lifetime_display.clear()
        
    def _on_data_source_table_item_changed(self, item):
        if self.level_table.selectionModel() and self.level_table.selectionModel().hasSelection():
            selected_indexes = self.level_table.selectionModel().selectedRows()
            row = selected_indexes[0].row()
            selected_level_data = self.filtered_levels_df.iloc[row]
            self._populate_line_data_table(selected_level_data['key'])
        else:
            self.line_data_table.setModel(None)
            
    def _on_line_selected(self, index: QModelIndex):
        if not index.isValid() or self.master_line_data_df.empty:
            self._clear_plot(); return

        row = index.row()
        line_data = self.master_line_data_df.iloc[row]
        wavenumber = line_data.get('wavenumber')
        if wavenumber is not None:
            self._update_plot(wavenumber, self._get_checked_data_paths())
        else:
            self._clear_plot()
            
    def _update_plot(self, target_wavenumber: float, all_checked_paths: list):
        self.figure.clear()
        plot_in_separate_windows = self.separate_plots_checkbox.isChecked()
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linelist_paths = [p for p in all_checked_paths if 'Calibrated_Linelists' in p or 'Identified_Lines' in p]
        spectrum_data_paths = [p for p in all_checked_paths if 'Raw_Data' in p]
        max_fwhm = 0.0
        tolerance = float(self.tolerance_edit.text())
        for path in linelist_paths:
            try:
                linelist_df = h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
                if 'wavenumber' not in linelist_df.columns or 'width' not in linelist_df.columns: continue
                linelist_df['wavenumber'] = pd.to_numeric(linelist_df['wavenumber'], errors='coerce')
                differences = np.abs(linelist_df['wavenumber'] - target_wavenumber)
                best_match_index = differences.idxmin()
                if differences[best_match_index] <= tolerance:
                    max_fwhm = max(max_fwhm, linelist_df.loc[best_match_index, 'width'])
            except Exception as e:
                print(f"Could not read FWHM for line {target_wavenumber} in {path}: {e}")
        if max_fwhm > 0: max_fwhm /= 1000.0
        plot_range = (5.0 * max_fwhm) if max_fwhm > 0 else 5.0
        spectrum_data_loaded = False
        
        self.ax = self.figure.add_subplot(1, 1, 1)
        
        for i, spec_path in enumerate(spectrum_data_paths):
            try:
                line_color = color_cycle[i % len(color_cycle)]
                vline_color = 'red'
                with h5py.File(self.h5_filepath, 'r') as f:
                    h5_dataset = f[spec_path]
                    attrs = h5_dataset.attrs
                    wavcorr, wstart, delw, rdsclfct = attrs.get('wavcorr', 0.0), attrs.get('wstart', 0.0), attrs.get('delw', 1.0), attrs.get('rdsclfct', 1.0)
                    data = h5_dataset[:]
                    spectrum_name = spec_path.split('/')[2]
                    y, indices = data * rdsclfct, np.arange(len(data))
                    x = wstart + indices * delw
                    x_corrected = x * (1.0 + wavcorr)
                    mask = (x_corrected >= target_wavenumber - plot_range) & (x_corrected <= target_wavenumber + plot_range)
                    if np.any(mask):
                        self.ax.plot(x_corrected[mask], y[mask], color=line_color, alpha=0.7, label=spectrum_name)
                        spectrum_data_loaded = True
            except Exception as e:
                print(f"Error loading spectrum data for plot from {spec_path}: {e}")
        
        if spectrum_data_loaded:
            self.ax.axvline(target_wavenumber, color=vline_color, linestyle='--')
            self.ax.set_title(f"Spectra around {target_wavenumber:.3f} cm⁻¹")
            self.ax.legend()
            self.ax.grid(True)
            self.figure.supxlabel(r'$\sigma$ (cm$^{-1}$)')
            self.figure.supylabel('Intensity')
            self.figure.tight_layout()
        else:
            self.ax.text(0.5, 0.5, "No Spectrum Data Selected or Loaded", ha='center', va='center', transform=self.ax.transAxes, fontsize=12, color='darkred')
        
        self.canvas.draw()
            
    def _close_extra_plot_windows(self):
        for window in self.extra_plot_windows: window.close()
        self.extra_plot_windows = []
        
    def _clear_plot(self):
        if self.figure.get_axes():
            ax = self.figure.get_axes()[0]
            ax.clear()
            ax.text(0.5, 0.5, "Select an upper level and a line to view spectrum", ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw()

    def _calculate_clicked(self):
        if self.master_line_data_df.empty:
            QMessageBox.warning(self, "Calculation Error", "No lines loaded."); return
        
        # --- MODIFICATION: Use all lines for calculation ---
        lines_for_calculation = self.master_line_data_df
        if lines_for_calculation.empty:
            QMessageBox.information(self, "Calculation", "No lines available for calculation.");
            self.result_df, self.save_results_btn.setEnabled(pd.DataFrame(), False); return

        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes:
            QMessageBox.warning(self, "Calculation Error", "Please select an upper level."); return
        row = selected_indexes[0].row()
        selected_level_data = self.filtered_levels_df.iloc[row]
        upper_level_key = selected_level_data['key']
        try:
            self.result_df = self.analysis_module.calculate_branching_fractions(lines_for_calculation, upper_level_key=upper_level_key, energy_levels_df=self.current_energy_levels_df)
            if not self.result_df.empty:
                QMessageBox.information(self, "Calculation Complete", "Branching fractions calculated successfully!")
                self.save_results_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Calculation Error", "Calculation returned no results.")
                self.save_results_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"An error occurred: {e}")
            self.result_df, self.save_results_btn.setEnabled(pd.DataFrame(), False)
            
    def _save_results_clicked(self):
        if self.result_df.empty:
            QMessageBox.warning(self, "Save Error", "No results to save."); return
        results_name, ok = QInputDialog.getText(self, "Save Results", "Enter a name for this analysis dataset:")
        if ok and results_name:
            h5_manager.create_group_if_not_exists(self.h5_filepath, '/Calculated_Branching_Fractions')
            metadata_to_save = {
                'analysis_date': date.today().isoformat(), 'source_level_file': self.level_file_combo.currentText(),
                'source_previous_ids_file': self.prev_id_combo.currentText(), 'source_linelists': self._get_checked_data_paths(),
                'wavenumber_tolerance': float(self.tolerance_edit.text()), 'upper_level_key': self.level_key_display.text(),
                'notes': f"Branching fractions for {self.level_key_display.text()} calculated using SAAS."
            }
            try:
                self.h5_manager.add_pandas_table(
                    self.h5_filepath, '/Calculated_Branching_Fractions', results_name,
                    self.result_df, metadata_dict=metadata_to_save
                )
                QMessageBox.information(self, "Save Complete", f"Results saved to HDF5 at: /Calculated_Branching_Fractions/{results_name}")
            except Exception as e:
                QMessageBox.critical(self, "HDF5 Save Error", f"Failed to save results:\n{e}")
        else:
            QMessageBox.information(self, "Save Cancelled", "Saving results cancelled.")
            
    def _run_debug_diagnostics(self):
        report = []
        report.append("--- 1. Master DataFrames ---")
        if self.current_energy_levels_df.empty:
            report.append("WARNING: Energy Levels DataFrame is EMPTY.")
        else:
            report.append(f"OK: Energy Levels DataFrame loaded ({len(self.current_energy_levels_df)} rows).")
        report.append("\n")
        if self.current_previous_ids_df.empty:
            report.append("WARNING: Previous IDs DataFrame is EMPTY.")
        else:
            report.append(f"OK: Previous IDs DataFrame loaded ({len(self.current_previous_ids_df)} rows).")
            if 'normalized_key' in self.current_previous_ids_df.columns:
                 report.append("OK: 'normalized_key' column was successfully created.")
            else:
                 report.append("ERROR: 'normalized_key' column was NOT created.")
        report.append("\n--- 2. Level Selection & Filtering ---")
        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes:
            report.append("INFO: No level is currently selected in the table.")
            self._show_debug_report(report); return
        row = selected_indexes[0].row()
        selected_level_data = self.filtered_levels_df.iloc[row]
        upper_level_key = selected_level_data.get('key')
        if not upper_level_key:
            report.append("ERROR: A level is selected, but could not get its 'key' value!")
            self._show_debug_report(report); return
        report.append(f"OK: A level is selected. The key being used for filtering is: '{upper_level_key}'")
        report.append("\n--- 3. Filtering Previous IDs ---")
        if 'normalized_key' not in self.current_previous_ids_df.columns:
            report.append(f"FATAL ERROR: The Previous IDs DataFrame does NOT have the 'normalized_key' column.")
            self._show_debug_report(report); return
        lines_from_level = self.current_previous_ids_df[self.current_previous_ids_df['normalized_key'] == upper_level_key]
        if lines_from_level.empty:
            report.append(f"\nRESULT: CRITICAL FAILURE! Found 0 matching lines for key '{upper_level_key}'.")
        else:
            report.append(f"\nRESULT: SUCCESS! Found {len(lines_from_level)} matching lines for key '{upper_level_key}'.")
        self._show_debug_report(report)
        
    def _show_debug_report(self, report_lines):
        dialog = QDialog(self)
        dialog.setWindowTitle("Debug Diagnostics Report")
        dialog.setMinimumSize(700, 500)
        layout = QVBoxLayout(dialog)
        report_text = QTextEdit()
        report_text.setReadOnly(True)
        report_text.setFont(QFont("Monospace", 10))
        report_text.setText("\n".join(report_lines))
        layout.addWidget(report_text)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        dialog.exec_()