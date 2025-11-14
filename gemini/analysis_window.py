# analysis_window.py (FINAL CRASH FIX)

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableView, QTreeView, QSplitter, QDockWidget, QPushButton, QLineEdit,
    QAbstractItemView, QSizePolicy, QHeaderView, QMenuBar, QAction, QMessageBox,
    QDialog, QDialogButtonBox, QInputDialog, QFormLayout, QTextEdit, QCheckBox
)
from PyQt5.QtCore import Qt, QModelIndex, QAbstractTableModel, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QStandardItemModel, QStandardItem, QIcon, QDoubleValidator

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
# --- FIX 1: Add the required import for rcParams ---
import matplotlib.pyplot as plt

import h5_manager 
import analysis 

# ... (The model classes and PlotPopupDialog class at the top of your file are unchanged) ...
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

class LineDataTableModel(PandasTableModel):
    include_in_fit_changed = pyqtSignal(pd.Series) 
    def __init__(self, data: pd.DataFrame, parent=None):
        if 'Include_in_Fit' not in data.columns:
            data['Include_in_Fit'] = True
        super().__init__(data, parent)
        self.include_col_index = self.df.columns.get_loc('Include_in_Fit')
    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid(): return None
        if index.column() == self.include_col_index:
            if role == Qt.CheckStateRole:
                return Qt.Checked if self.df.iloc[index.row(), index.column()] else Qt.Unchecked
            elif role == Qt.DisplayRole: return None
            elif role == Qt.ForegroundRole:
                if not self.df.iloc[index.row(), self.include_col_index]: return QColor(Qt.gray)
        else:
            if role == Qt.ForegroundRole:
                if not self.df.iloc[index.row(), self.include_col_index]: return QColor(Qt.gray)
            return super().data(index, role)
        return None
    def setData(self, index: QModelIndex, value, role=Qt.EditRole):
        if not index.isValid(): return False
        if index.column() == self.include_col_index and role == Qt.CheckStateRole:
            new_value = (value == Qt.Checked)
            if self.df.iloc[index.row(), index.column()] != new_value:
                self.df.iloc[index.row(), index.column()] = new_value
                self.dataChanged.emit(index, index, [Qt.CheckStateRole, Qt.ForegroundRole])
                self.include_in_fit_changed.emit(self.df.iloc[index.row()]) 
                return True
        return False
    def flags(self, index: QModelIndex):
        if not index.isValid(): return Qt.NoItemFlags
        base_flags = super().flags(index)
        if index.column() == self.include_col_index:
            return base_flags | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
        return base_flags | Qt.ItemIsEnabled

class HDF5TreeModel(QStandardItemModel):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setHorizontalHeaderLabels(['Item'])
        self._populate_tree()
    def _populate_tree(self):
        self.clear()
        self.setHorizontalHeaderLabels(['Item'])
        if not self.h5_filepath or not os.path.exists(self.h5_filepath): return
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                root_item = self.invisibleRootItem()
                self._add_items_to_tree_recursively(root_item, f)
        except Exception as e:
            QMessageBox.critical(None, "Error Reading HDF5", f"Could not read HDF5 structure: {e}")
    def _add_items_to_tree_recursively(self, parent_item, h5_object):
        for name, item in h5_object.items():
            child_item = QStandardItem(name)
            child_item.setData(item.name, Qt.UserRole)
            if isinstance(item, h5py.Dataset):
                if ('Identified_Lines' in item.name or 'Calibrated_Linelists' in item.name or 'Raw_Data' in item.name):
                    child_item.setCheckable(True)
                    child_item.setCheckState(Qt.Unchecked)
                else:
                    child_item.setCheckable(False)
            else:
                 child_item.setCheckable(False)
            parent_item.appendRow(child_item)
            if isinstance(item, h5py.Group):
                self._add_items_to_tree_recursively(child_item, item)
    def get_checked_items(self, parent_item=None):
        checked_paths = []
        if parent_item is None: parent_item = self.invisibleRootItem()
        for row in range(parent_item.rowCount()):
            item = parent_item.child(row)
            if item.isCheckable() and item.checkState() == Qt.Checked: checked_paths.append(item.data(Qt.UserRole))
            if item.hasChildren(): checked_paths.extend(self.get_checked_items(item))
        return checked_paths

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
        self.setGeometry(100, 100, 1400, 900)
        self.extra_plot_windows = []
        self.h5_filepath = h5_filepath
        self.h5_manager = h5_manager 
        self.analysis_module = analysis 
        self.current_energy_levels_df = pd.DataFrame()
        self.current_previous_ids_df = pd.DataFrame()
        self.filtered_levels_df = pd.DataFrame()
        self.master_line_data_df = pd.DataFrame()
        self.result_df = pd.DataFrame()
        self._create_menu_bar()
        self._create_main_layout()
        self._create_docks()
        self._create_central_widget()
        self._populate_initial_comboboxes()
        self._clear_plot()
        self.central_splitter.setSizes([int(self.height() * 0.5), int(self.height() * 0.5)])
        self.main_splitter.setSizes([int(self.width() * 0.25), int(self.width() * 0.5), int(self.width() * 0.25)])

    def _create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        debug_menu = menubar.addMenu("&Debug")
        run_diagnostics_action = QAction("Run Diagnostics...", self)
        run_diagnostics_action.triggered.connect(self._run_debug_diagnostics)
        debug_menu.addAction(run_diagnostics_action)
        help_menu = menubar.addMenu("&Help")
        help_action = QAction("About", self)
        help_action.triggered.connect(lambda: QMessageBox.information(self, "About", "Interactive Branching Fraction Analysis Tool v0.1"))
        help_menu.addAction(help_action)

    def _create_main_layout(self):
        self.central_widget_container = QWidget()
        self.setCentralWidget(self.central_widget_container)
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout(self.central_widget_container)
        main_layout.addWidget(self.main_splitter)

    def _create_docks(self):
        self.left_dock = QDockWidget("Level Selector", self)
        # ... (rest of method is unchanged) ...
        self.left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        left_dock_widget = QWidget()
        self.left_dock_layout = QVBoxLayout(left_dock_widget)
        self.level_file_combo = QComboBox()
        self.level_file_combo.addItem("Select Energy Level File...")
        self.level_file_combo.currentIndexChanged.connect(self._on_level_file_selected)
        self.left_dock_layout.addWidget(QLabel("Master Energy Level File:"))
        self.left_dock_layout.addWidget(self.level_file_combo)
        self.level_table = QTableView()
        self.level_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.level_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.level_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.level_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.level_table.clicked.connect(self._on_level_selected_in_table)
        self.left_dock_layout.addWidget(QLabel("Available Upper Levels (lifetime > 0):"))
        self.left_dock_layout.addWidget(self.level_table)
        self.level_details_group = QWidget()
        level_details_layout = QFormLayout(self.level_details_group)
        self.level_key_display = QLineEdit()
        self.level_energy_display = QLineEdit()
        self.level_j_display = QLineEdit()
        self.level_parity_display = QLineEdit()
        for editor in [self.level_key_display, self.level_energy_display, self.level_j_display, self.level_parity_display]:
            editor.setReadOnly(True)
            editor.setFont(QFont("Monospace", 10))
        level_details_layout.addRow("key:", self.level_key_display)
        level_details_layout.addRow("energy (cm⁻¹):", self.level_energy_display)
        level_details_layout.addRow("j_value:", self.level_j_display)
        level_details_layout.addRow("parity:", self.level_parity_display)
        self.left_dock_layout.addWidget(QLabel("Selected Level Details:"))
        self.left_dock_layout.addWidget(self.level_details_group)
        left_dock_widget.setLayout(self.left_dock_layout)
        self.left_dock.setWidget(left_dock_widget)
        self.main_splitter.addWidget(self.left_dock) 

        self.right_dock = QDockWidget("Data Source Selector", self)
        self.right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        right_dock_widget = QWidget()
        self.right_dock_layout = QVBoxLayout(right_dock_widget)
        self.prev_id_combo = QComboBox()
        self.prev_id_combo.addItem("Select Previous IDs File...")
        self.prev_id_combo.currentIndexChanged.connect(self._on_prev_id_file_selected)
        self.right_dock_layout.addWidget(QLabel("Master Previous IDs File:"))
        self.right_dock_layout.addWidget(self.prev_id_combo)
        self.right_dock_layout.addWidget(QLabel("Select Data for Comparison/Plotting:"))
        self.data_source_tree = QTreeView()
        self.data_source_model = HDF5TreeModel(self.h5_filepath)
        self.data_source_tree.setModel(self.data_source_model)
        self.data_source_tree.setHeaderHidden(True)
        self.data_source_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_source_model.itemChanged.connect(self._on_data_source_tree_item_changed)
        self.right_dock_layout.addWidget(self.data_source_tree)
        self.analysis_controls_group = QWidget()
        analysis_controls_layout = QVBoxLayout(self.analysis_controls_group)
        self.separate_plots_checkbox = QCheckBox("Plot Spectra in Separate Windows")
        analysis_controls_layout.addWidget(self.separate_plots_checkbox)
        self.tolerance_edit = QLineEdit("0.02")
        self.tolerance_edit.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        analysis_controls_layout.addWidget(QLabel("Wavenumber Matching Tolerance (cm⁻¹):"))
        analysis_controls_layout.addWidget(self.tolerance_edit)
        self.run_analysis_btn = QPushButton("Calculate Branching Fractions")
        self.run_analysis_btn.clicked.connect(self._calculate_clicked)
        analysis_controls_layout.addWidget(self.run_analysis_btn)
        self.save_results_btn = QPushButton("Save Results to HDF5")
        self.save_results_btn.clicked.connect(self._save_results_clicked)
        self.save_results_btn.setEnabled(False)
        analysis_controls_layout.addWidget(self.save_results_btn)
        self.right_dock_layout.addWidget(self.analysis_controls_group)
        right_dock_widget.setLayout(self.right_dock_layout)
        self.right_dock.setWidget(right_dock_widget)
        self.main_splitter.addWidget(self.right_dock)

    def _create_central_widget(self):
        self.central_splitter = QSplitter(Qt.Vertical)
        self.line_data_table = QTableView()
        self.line_data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.line_data_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.line_data_table.setAlternatingRowColors(True)
        self.line_data_table.clicked.connect(self._on_line_selected)
        self.line_data_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.line_data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.central_splitter.addWidget(self.line_data_table)
        main_plot_widget = QWidget()
        plot_layout = QVBoxLayout(main_plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.central_splitter.addWidget(main_plot_widget)
        self.main_splitter.addWidget(self.central_splitter)
    
    def _populate_initial_comboboxes(self):
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Levels' in f:
                    level_names = [name for name in f['/Levels'].keys() if isinstance(f['/Levels'][name], h5py.Group)]
                    self.level_file_combo.addItems(level_names)
                if '/Previous_Identifications' in f:
                    prev_id_names = [name for name in f['/Previous_Identifications'].keys() if isinstance(f['/Previous_Identifications'][name], h5py.Group)]
                    self.prev_id_combo.addItems(prev_id_names)
        except Exception as e:
            QMessageBox.critical(self, "HDF5 Error", f"Failed to read HDF5 structure: {e}")

    def _on_level_file_selected(self):
        selected_file = self.level_file_combo.currentText()
        if selected_file == "Select Energy Level File...":
            self.level_table.setModel(None)
            self._clear_level_details()
            self.current_energy_levels_df = pd.DataFrame()
            self.filtered_levels_df = pd.DataFrame()
            return
        path = f"/Levels/{selected_file}/table"
        try:
            self.current_energy_levels_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
            if not self.current_energy_levels_df.empty and 'lifetime' in self.current_energy_levels_df.columns:
                self.filtered_levels_df = self.current_energy_levels_df[(self.current_energy_levels_df['lifetime'] > 0)].copy()
                model = PandasTableModel(self.filtered_levels_df[['key', 'energy', 'j_value', 'lifetime']])
                self.level_table.setModel(model)
                self.level_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            else:
                self.level_table.setModel(None)
                QMessageBox.warning(self, "Data Error", f"Energy levels table at {path} is empty or missing 'lifetime' column.")
        except Exception as e:
            self.level_table.setModel(None)
            self.current_energy_levels_df = pd.DataFrame()
            self.filtered_levels_df = pd.DataFrame()
            QMessageBox.critical(self, "HDF5 Read Error", f"Could not read energy levels from {path}:\n{e}")
        finally:
            self._clear_level_details()

    def _on_prev_id_file_selected(self):
        selected_file = self.prev_id_combo.currentText()
        if selected_file == "Select Previous IDs File...":
            self.current_previous_ids_df = pd.DataFrame()
            self.line_data_table.setModel(None)
            self._clear_plot()
            return
        path = f"/Previous_Identifications/{selected_file}/table"
        try:
            self.current_previous_ids_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
            if not self.current_previous_ids_df.empty and 'upper_level_key' in self.current_previous_ids_df.columns:
                self.current_previous_ids_df['normalized_key'] = self.current_previous_ids_df['upper_level_key'].astype(str).str.replace('*', '', regex=False).str.strip()
            if self.level_table.selectionModel() and self.level_table.selectionModel().hasSelection():
                self._on_level_selected_in_table()
        except Exception as e:
            self.current_previous_ids_df = pd.DataFrame()
            QMessageBox.critical(self, "HDF5 Read Error", f"Could not read Previous IDs from {path}:\n{e}")
            self.line_data_table.setModel(None)
            self._clear_plot()
            
    def _on_level_selected_in_table(self):
        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes or self.filtered_levels_df.empty:
            self._clear_level_details()
            self.line_data_table.setModel(None)
            self._clear_plot()
            return
        row = selected_indexes[0].row()
        selected_level_data = self.filtered_levels_df.iloc[row]
        self.level_key_display.setText(str(selected_level_data.get('key', 'N/A')))
        self.level_energy_display.setText(f"{selected_level_data.get('energy', 0.0):.3f}")
        self.level_j_display.setText(str(selected_level_data.get('j_value', 'N/A')))
        self.level_parity_display.setText(str(selected_level_data.get('parity', 'N/A')))
        self._populate_line_data_table(selected_level_data['key'])
        
    def _populate_line_data_table(self, upper_level_key: str):
        if self.current_previous_ids_df.empty:
            self.line_data_table.setModel(None)
            self._clear_plot()
            return
        if 'normalized_key' not in self.current_previous_ids_df.columns:
             return
        lines_from_level = self.current_previous_ids_df[self.current_previous_ids_df['normalized_key'] == upper_level_key]
        if lines_from_level.empty:
            self.line_data_table.setModel(None)
            self._clear_plot()
            return
        selected_linelist_paths = self.data_source_model.get_checked_items()
        linelist_paths_to_merge = [p for p in selected_linelist_paths if ('Identified_Lines' in p or 'Calibrated_Linelists' in p) and 'table' in p]
        try:
            df_to_pass = lines_from_level.drop(columns=['normalized_key'], errors='ignore')
            self.master_line_data_df = self.analysis_module.aggregate_observed_data_for_display(
                h5_filepath=self.h5_filepath,
                previous_ids_df=df_to_pass,
                linelist_paths=linelist_paths_to_merge,
                tolerance=float(self.tolerance_edit.text())
            )
            if self.master_line_data_df.empty:
                self.line_data_table.setModel(None)
                self._clear_plot()
                return
            model = LineDataTableModel(self.master_line_data_df)
            self.line_data_table.setModel(model)
            model.include_in_fit_changed.connect(self._on_line_include_changed)
            self._clear_plot()
            current_height = self.central_splitter.height()
            self.central_splitter.setSizes([current_height // 2, current_height // 2])
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred in _populate_line_data_table: {e}")
            self.line_data_table.setModel(None)
            self._clear_plot()
            
    def _clear_level_details(self):
        self.level_key_display.clear()
        self.level_energy_display.clear()
        self.level_j_display.clear()
        self.level_parity_display.clear()
        
    def _on_data_source_tree_item_changed(self, item: QStandardItem):
        if self.level_table.selectionModel() and self.level_table.selectionModel().hasSelection():
            selected_indexes = self.level_table.selectionModel().selectedRows()
            row = selected_indexes[0].row()
            selected_level_data = self.filtered_levels_df.iloc[row]
            self._populate_line_data_table(selected_level_data['key'])
        else:
            self.line_data_table.setModel(None)
            
    def _on_line_selected(self):
        selected_indexes = self.line_data_table.selectionModel().selectedRows()
        if not selected_indexes or self.master_line_data_df.empty:
            self._clear_plot()
            return
        row = selected_indexes[0].row()
        line_data = self.master_line_data_df.iloc[row]
        wavenumber = line_data.get('wavenumber_id')
        if wavenumber is None: wavenumber = line_data.get('wavenumber')
        is_excluded = not line_data.get('Include_in_Fit', True)
        if wavenumber is not None:
            selected_spectrum_paths = [p for p in self.data_source_model.get_checked_items() if 'Raw_Data' in p]
            self._update_plot(wavenumber, selected_spectrum_paths, is_excluded)
        else:
            self._clear_plot()
            
    def _on_line_include_changed(self, updated_row_data: pd.Series):
        current_selection_model = self.line_data_table.selectionModel()
        if current_selection_model and current_selection_model.hasSelection():
            selected_row_index = current_selection_model.selectedRows()[0]
            if self.master_line_data_df.iloc[selected_row_index.row()].equals(updated_row_data):
                wavenumber = updated_row_data.get('wavenumber_id')
                if wavenumber is None: wavenumber = updated_row_data.get('wavenumber')
                is_excluded = not updated_row_data.get('Include_in_Fit', True)
                selected_spectrum_paths = [p for p in self.data_source_model.get_checked_items() if 'Raw_Data' in p]
                self._update_plot(wavenumber, selected_spectrum_paths, is_excluded)
                
    def _update_plot(self, target_wavenumber: float, spectrum_paths: list, is_excluded: bool):
        self._close_extra_plot_windows()
        self.ax.clear()

        plot_in_separate_windows = self.separate_plots_checkbox.isChecked()
        
        # --- FIX 2: Correctly get the color cycle from rcParams ---
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        spectrum_data_loaded = False
        vline_plotted_on_main = False

        for i, spec_path in enumerate(spectrum_paths):
            try:
                line_color = 'gray' if is_excluded else color_cycle[i % len(color_cycle)]
                vline_color = 'gray' if is_excluded else 'red'

                with h5py.File(self.h5_filepath, 'r') as f:
                    h5_dataset = f[spec_path]
                    data, attrs = h5_dataset[:], h5_dataset.attrs
                    parent_group = h5_dataset.parent
                    wavcorr = parent_group.attrs.get('wavcorr', 0.0)
                    spectrum_name = spec_path.split('/')[2]
                    wstart, delw, rdsclfct = attrs.get('wstart', 0.0), attrs.get('delw', 1.0), attrs.get('rdsclfct', 1.0)
                    
                    y, indices = data * rdsclfct, np.arange(len(data))
                    x = wstart + indices * delw
                    x_corrected = x * (1.0 + wavcorr)
                    
                    plot_range = 5
                    mask = (x_corrected >= target_wavenumber - plot_range) & (x_corrected <= target_wavenumber + plot_range)

                    if np.any(mask):
                        plot_axis = self.ax
                        if plot_in_separate_windows:
                            popup = PlotPopupDialog(f"Spectrum: {spectrum_name} (around {target_wavenumber:.3f} cm⁻¹)", self)
                            self.extra_plot_windows.append(popup)
                            plot_axis = popup.ax
                            popup.show()

                        label_for_vline = 'Target Line' if (plot_in_separate_windows or not vline_plotted_on_main) else None
                        
                        plot_axis.plot(x_corrected[mask], y[mask], color=line_color, alpha=0.7, label=spectrum_name)
                        plot_axis.axvline(target_wavenumber, color=vline_color, linestyle='--', label=label_for_vline)
                        
                        if not plot_in_separate_windows: vline_plotted_on_main = True

                        plot_axis.set_title(f"Spectrum around {target_wavenumber:.3f} cm⁻¹")
                        plot_axis.set_xlabel("Corrected Wavenumber (cm⁻¹)")
                        plot_axis.set_ylabel("Intensity")
                        plot_axis.legend()
                        plot_axis.grid(True)
                        
                        if plot_in_separate_windows:
                            popup.figure.tight_layout()
                            popup.canvas.draw()
                        
                        spectrum_data_loaded = True
            except Exception as e:
                print(f"Error loading spectrum data for plot from {spec_path}: {e}")

        if not plot_in_separate_windows:
            if spectrum_data_loaded:
                self.figure.tight_layout()
            else:
                self.ax.text(0.5, 0.5, "No Spectrum Data Selected or Loaded",
                             ha='center', va='center', transform=self.ax.transAxes, fontsize=12, color='darkred')
            self.canvas.draw()
        elif spectrum_data_loaded:
            self.ax.text(0.5, 0.5, "Spectra are displayed in separate windows.",
                         ha='center', va='center', transform=self.ax.transAxes, fontsize=14, color='gray')
            self.canvas.draw()
            
    def _close_extra_plot_windows(self):
        for window in self.extra_plot_windows:
            window.close()
        self.extra_plot_windows = []
        
    def _clear_plot(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Select an upper level and a line to view spectrum",
                     ha='center', va='center', transform=self.ax.transAxes, fontsize=14, color='gray')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
        
    def _calculate_clicked(self):
        if self.master_line_data_df.empty:
            QMessageBox.warning(self, "Calculation Error", "No lines loaded.")
            return
        lines_for_calculation = self.master_line_data_df[self.master_line_data_df['Include_in_Fit'] == True]
        if lines_for_calculation.empty:
            QMessageBox.information(self, "Calculation", "No lines selected for calculation.")
            self.result_df = pd.DataFrame()
            self.save_results_btn.setEnabled(False)
            return
        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes:
            QMessageBox.warning(self, "Calculation Error", "Please select an upper level.")
            return
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
            QMessageBox.warning(self, "Save Error", "No results to save.")
            return
        results_name, ok = QInputDialog.getText(self, "Save Results", "Enter a name for this analysis dataset:")
        if ok and results_name:
            h5_manager.create_group_if_not_exists(self.h5_filepath, '/Calculated_Branching_Fractions')
            metadata_to_save = {
                'analysis_date': date.today().isoformat(), 'source_level_file': self.level_file_combo.currentText(),
                'source_previous_ids_file': self.prev_id_combo.currentText(), 'source_linelists': self.data_source_model.get_checked_items(),
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
            report.append(f"Columns: {list(self.current_energy_levels_df.columns)}")
        report.append("\n")
        if self.current_previous_ids_df.empty:
            report.append("WARNING: Previous IDs DataFrame is EMPTY.")
        else:
            report.append(f"OK: Previous IDs DataFrame loaded ({len(self.current_previous_ids_df)} rows).")
            report.append(f"Columns: {list(self.current_previous_ids_df.columns)}")
            if 'normalized_key' in self.current_previous_ids_df.columns:
                 report.append("OK: 'normalized_key' column was successfully created.")
                 report.append("First 2 rows of keys:\n" + self.current_previous_ids_df[['upper_level_key', 'normalized_key']].head(2).to_string())
            else:
                 report.append("ERROR: 'normalized_key' column was NOT created. Problem in _on_prev_id_file_selected.")
        report.append("\n--- 2. Level Selection & Filtering ---")
        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes:
            report.append("INFO: No level is currently selected in the table.")
            self._show_debug_report(report)
            return
        row = selected_indexes[0].row()
        selected_level_data = self.filtered_levels_df.iloc[row]
        upper_level_key = selected_level_data.get('key')
        if not upper_level_key:
            report.append("ERROR: A level is selected, but could not get its 'key' value!")
            self._show_debug_report(report)
            return
        report.append(f"OK: A level is selected. The key being used for filtering is: '{upper_level_key}'")
        report.append("\n--- 3. Filtering Previous IDs ---")
        report.append("Attempting to find rows in Previous IDs where the 'normalized_key' matches the selected key...")
        if 'normalized_key' not in self.current_previous_ids_df.columns:
            report.append(f"FATAL ERROR: The Previous IDs DataFrame does NOT have the 'normalized_key' column.")
            self._show_debug_report(report)
            return
        lines_from_level = self.current_previous_ids_df[self.current_previous_ids_df['normalized_key'] == upper_level_key]
        if lines_from_level.empty:
            report.append("\nRESULT: CRITICAL FAILURE!")
            report.append(f"Found 0 matching lines for key '{upper_level_key}'.")
            report.append("This is why the central table is empty.")
            report.append("\nTroubleshooting:")
            report.append("1. Verify the 'normalized_key' column in step 1 looks correct (no asterisks or spaces).")
            report.append("2. Check for hidden characters or case-sensitivity issues that were not caught.")
            if not self.current_previous_ids_df.empty:
                report.append("\nFirst 5 'normalized_key' values from your IDs file for comparison:")
                report.append(self.current_previous_ids_df['normalized_key'].head(5).to_string())
        else:
            report.append(f"\nRESULT: SUCCESS!")
            report.append(f"Found {len(lines_from_level)} matching lines for key '{upper_level_key}'.")
            report.append("If table is still empty, the problem is in the GUI rendering after `setModel` is called.")
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