# gui.py (STABLE & READABLE)
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

# Matplotlib integration
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Import project modules
from analysis_window import AnalysisWindow
import importers
import h5_manager
import analysis

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data
    def rowCount(self, parent=None):
        return self._data.shape[0]
    def columnCount(self, parent=None):
        return self._data.shape[1]
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (float, np.floating)):
                if pd.isna(value):
                    return ""
                return f"{value:.4f}"
            return str(value)
        return None
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None

class FullTableWindow(QDialog):
    """Standalone searchable window for viewing entire datasets."""
    def __init__(self, df, title, parent=None):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle(f"Full Data View: {title}")
        self.setMinimumSize(1000, 600)
        
        layout = QVBoxLayout(self)
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search/Filter:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to filter...")
        self.search_edit.textChanged.connect(self._apply_filter)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)
        
        self.table_view = QTableView()
        self.table_view.setModel(PandasModel(df))
        self.table_view.setAlternatingRowColors(True)
        layout.addWidget(self.table_view)
        
        self.status_label = QLabel(f"Showing {len(df)} rows")
        layout.addWidget(self.status_label)
        
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _apply_filter(self, text):
        if not text:
            f_df = self.df
        else:
            f_df = self.df[self.df.apply(lambda r: r.astype(str).str.contains(text, case=False).any(), axis=1)]
        self.table_view.setModel(PandasModel(f_df))
        self.status_label.setText(f"Matches: {len(f_df)} / {len(self.df)}")

class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        l = QFormLayout(self)
        self.title_edit = QLineEdit()
        self.author_edit = QLineEdit()
        self.inst_edit = QLineEdit()
        self.super_edit = QLineEdit()
        l.addRow("Title:", self.title_edit)
        l.addRow("Author:", self.author_edit)
        l.addRow("Institution:", self.inst_edit)
        l.addRow("Supervisor:", self.super_edit)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        l.addRow(bb)
    def get_data(self):
        return {'project_title': self.title_edit.text(), 'author': self.author_edit.text(), 'institution': self.inst_edit.text(), 'supervisor': self.super_edit.text()}

class ImportSpectrumDialog(QDialog):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Main Spectrum")
        self.setMinimumWidth(500)
        l = QFormLayout(self)
        self.d_edit = QLineEdit()
        self.h_edit = QLineEdit()
        db = QPushButton("Browse...")
        db.clicked.connect(self._browse_data)
        hb = QPushButton("Browse...")
        hb.clicked.connect(self._browse_hdr)
        dl = QHBoxLayout()
        dl.addWidget(self.d_edit)
        dl.addWidget(db)
        l.addRow("Data File:", dl)
        hl = QHBoxLayout()
        hl.addWidget(self.h_edit)
        hl.addWidget(hb)
        l.addRow("Header File:", hl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        l.addRow(bb)
    def _browse_data(self):
        f, _ = QFileDialog.getOpenFileName(self, "Data File", "", "Data Files (*.raw *.dat)")
        if f:
            self.d_edit.setText(f)
    def _browse_hdr(self):
        f, _ = QFileDialog.getOpenFileName(self, "Header File", "", "Header Files (*.hdr)")
        if f:
            self.h_edit.setText(f)
    def accept(self):
        try:
            importers.import_spectrum_pair(self.h5_filepath, self.d_edit.text(), self.h_edit.text())
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class ImportCalibSpectrumDialog(QDialog):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Calibration Spectrum")
        self.setMinimumWidth(500)
        l = QFormLayout(self)
        self.combo = QComboBox()
        self.d_edit = QLineEdit()
        self.h_edit = QLineEdit()
        with h5py.File(self.h5_filepath, 'r') as f:
            if '/Spectra' in f:
                self.combo.addItems(list(f['/Spectra'].keys()))
        db = QPushButton("Browse...")
        db.clicked.connect(lambda: self.d_edit.setText(QFileDialog.getOpenFileName(self, "Data File", "", "Data Files (*.raw *.dat)")[0]))
        hb = QPushButton("Browse...")
        hb.clicked.connect(lambda: self.h_edit.setText(QFileDialog.getOpenFileName(self, "Header File", "", "Header Files (*.hdr)")[0]))
        l.addRow("Main Spectrum:", self.combo)
        dl = QHBoxLayout()
        dl.addWidget(self.d_edit)
        dl.addWidget(db)
        l.addRow("Data File:", dl)
        hl = QHBoxLayout()
        hl.addWidget(self.h_edit)
        hl.addWidget(hb)
        l.addRow("Header File:", hl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        l.addRow(bb)
    def accept(self):
        try:
            importers.import_spectrum_pair(self.h5_filepath, self.d_edit.text(), self.h_edit.text(), True, f"/Spectra/{self.combo.currentText()}")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class ImportLampCalDialog(QDialog):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Lamp Calibration")
        self.setMinimumWidth(500)
        l = QFormLayout(self)
        self.f_edit = QLineEdit()
        self.a_edit = QLineEdit()
        self.n_edit = QTextEdit()
        fb = QPushButton("Browse...")
        fb.clicked.connect(lambda: self.f_edit.setText(QFileDialog.getOpenFileName(self, "Select File", "", "Text Files (*.txt)")[0]))
        fl = QHBoxLayout()
        fl.addWidget(self.f_edit)
        fl.addWidget(fb)
        l.addRow("File:", fl)
        l.addRow("Author:", self.a_edit)
        l.addRow("Notes:", self.n_edit)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        l.addRow(bb)
    def accept(self):
        try:
            importers.import_lamp_calibration(self.h5_filepath, self.f_edit.text(), {'author': self.a_edit.text(), 'notes': self.n_edit.toPlainText()})
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class ImportLinelistDialog(QDialog):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Binary Linelist")
        l = QFormLayout(self)
        self.combo = QComboBox()
        self.f_edit = QLineEdit()
        with h5py.File(self.h5_filepath, 'r') as f:
            if '/Spectra' in f:
                self.combo.addItems(list(f['/Spectra'].keys()))
        fb = QPushButton("Browse...")
        fb.clicked.connect(lambda: self.f_edit.setText(QFileDialog.getOpenFileName(self, "Select File", "", "Linelist Files (*.lin)")[0]))
        l.addRow("Target:", self.combo)
        fl = QHBoxLayout()
        fl.addWidget(self.f_edit)
        fl.addWidget(fb)
        l.addRow("File:", fl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        l.addRow(bb)
    def accept(self):
        try:
            importers.import_binary_linelist(self.h5_filepath, self.f_edit.text(), f"/Spectra/{self.combo.currentText()}")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class ImportCalibratedLinelistDialog(QDialog):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Import Calibrated Linelist")
        l = QFormLayout(self)
        self.combo = QComboBox()
        self.f_edit = QLineEdit()
        with h5py.File(self.h5_filepath, 'r') as f:
            if '/Spectra' in f:
                self.combo.addItems(list(f['/Spectra'].keys()))
        fb = QPushButton("Browse...")
        fb.clicked.connect(lambda: self.f_edit.setText(QFileDialog.getOpenFileName(self, "Select File", "", "Text Files (*.txt)")[0]))
        l.addRow("Target:", self.combo)
        fl = QHBoxLayout()
        fl.addWidget(self.f_edit)
        fl.addWidget(fb)
        l.addRow("File:", fl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        l.addRow(bb)
    def accept(self):
        try:
            importers.import_calibrated_linelist(self.h5_filepath, self.f_edit.text(), f"/Spectra/{self.combo.currentText()}")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class ImportWizardDialog(QDialog):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.df_p = pd.DataFrame()
        self.setWindowTitle("Import Wizard")
        self.setMinimumSize(800, 600)
        ml = QVBoxLayout(self)
        fl = QFormLayout()
        self.path_e = QLineEdit()
        pb = QPushButton("Browse...")
        pb.clicked.connect(self._browse)
        pl = QHBoxLayout()
        pl.addWidget(self.path_e)
        pl.addWidget(pb)
        fl.addRow("File:", pl)
        self.prev_t = QTableView()
        fl.addRow("Preview:", self.prev_t)
        self.grp_c = QComboBox()
        self.grp_c.addItems(['/Calculations', '/Levels', '/Previous_Identifications'])
        fl.addRow("Group:", self.grp_c)
        self.map_l = QHBoxLayout()
        fl.addRow("Mapping:", self.map_l)
        self.name_e = QLineEdit()
        fl.addRow("Table Name:", self.name_e)
        self.grp_c.currentIndexChanged.connect(self._update_map)
        self.path_e.textChanged.connect(self._update_p)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        ml.addLayout(fl)
        ml.addWidget(bb)
    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Data Files (*.txt *.csv *.dat)")
        if f:
            self.path_e.setText(f)
            self.name_e.setText(os.path.splitext(os.path.basename(f))[0].replace('.','_'))
    def _update_p(self):
        if os.path.exists(self.path_e.text()): 
            self.df_p = importers.parse_generic_text_file(self.path_e.text(), delimiter='space').head(100)
            self.prev_t.setModel(PandasModel(self.df_p))
            self._update_map()
    def _update_map(self):
        for i in reversed(range(self.map_l.count())):
            self.map_l.itemAt(i).widget().setParent(None)
        self.map_cs = []
        s_types = ["(ignore)"]
        with h5py.File(self.h5_filepath, 'r') as f:
            s_types.extend(f[self.grp_c.currentText()].attrs.get('schema','').split(','))
        for _ in self.df_p.columns:
            c = QComboBox()
            c.addItems(s_types)
            self.map_l.addWidget(c)
            self.map_cs.append(c)
    def accept(self):
        full_df = importers.parse_generic_text_file(self.path_e.text(), delimiter='space')
        final_df = pd.DataFrame()
        for i, c in enumerate(self.map_cs):
            if c.currentText() != "(ignore)":
                final_df[c.currentText()] = full_df.iloc[:, i]
        for col in final_df.columns:
            if self.grp_c.currentText() == '/Previous_Identifications' and col in ['wavenumber', 'intensity']:
                final_df[col] = final_df[col].astype(str)
            else:
                final_df[col] = pd.to_numeric(final_df[col], errors='ignore')
        h5_manager.add_pandas_table(self.h5_filepath, self.grp_c.currentText(), self.name_e.text(), final_df)
        super().accept()

class WavenumberMatchDialog(QDialog):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.h5_filepath = h5_filepath
        self.setWindowTitle("Wavenumber Matching")
        l = QFormLayout(self)
        self.exp_c = QComboBox()
        self.id_c = QComboBox()
        self.tol_e = QLineEdit("0.02")
        self.name_e = QLineEdit()
        with h5py.File(self.h5_filepath, 'r') as f:
            if '/Spectra' in f:
                for s in f['/Spectra'].keys():
                    for sub in ['Raw_Linelists', 'Calibrated_Linelists']:
                        path = f'/Spectra/{s}/{sub}'
                        if path in f:
                            for t in f[path].keys():
                                self.exp_c.addItem(f'{path}/{t}/table')
            if '/Previous_Identifications' in f:
                for t in f['/Previous_Identifications'].keys():
                    self.id_c.addItem(f'/Previous_Identifications/{t}/table')
        l.addRow("Exp Linelist:", self.exp_c)
        l.addRow("ID Table:", self.id_c)
        l.addRow("Tolerance:", self.tol_e)
        l.addRow("Output Name:", self.name_e)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        l.addRow(bb)
    def accept(self):
        try:
            n = analysis.run_and_save_wavenumber_match(self.h5_filepath, self.exp_c.currentText(), self.id_c.currentText(), float(self.tol_e.text()), self.name_e.text())
            QMessageBox.information(self, "Success", f"Matches: {n}")
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QApplication.setDesktopFileName("SAAS")
        self.setWindowTitle("SAAS - Spectroscopy Data Manager")
        self.setMinimumSize(1000, 800)
        self.setWindowIcon(QIcon(resource_path('SAAS_logo.png')))
        self.current_h5_file = None
        central = QWidget()
        self.setCentralWidget(central)
        main_l = QVBoxLayout(central)
        self._setup_menus()
        splitter = QSplitter(Qt.Horizontal)
        self.tree_view = QTreeView()
        self.tree_model = QStandardItemModel()
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._show_tree_context_menu)
        splitter.addWidget(self.tree_view)
        self.tabs = QTabWidget()
        prev_w = QWidget()
        self.prev_l = QVBoxLayout(prev_w)
        self.data_table = QTableView()
        self.plot_fig = Figure()
        self.plot_can = FigureCanvas(self.plot_fig)
        self.plot_tb = NavigationToolbar(self.plot_can, self)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.prev_l.addWidget(self.data_table)
        self.prev_l.addWidget(self.plot_tb)
        self.prev_l.addWidget(self.plot_can)
        self.data_table.hide()
        self.plot_tb.hide()
        self.plot_can.hide()
        self.attr_view = QTableWidget()
        self.attr_view.setColumnCount(2)
        self.attr_view.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.tabs.addTab(prev_w, "Preview")
        self.tabs.addTab(self.attr_view, "Metadata")
        splitter.addWidget(self.tabs)
        splitter.setSizes([300, 700])
        main_l.addWidget(splitter)
        self.set_file_loaded_state(False)
        self.tree_view.clicked.connect(self._on_tree_item_selected)

    def _setup_menus(self):
        m = self.menuBar()
        f_m = m.addMenu("&File")
        f_m.addAction("&New...", self._create_file, "Ctrl+N")
        f_m.addAction("&Open...", self._open_file, "Ctrl+O")
        f_m.addSeparator()
        f_m.addAction("Exit", self.close)
        self.import_menu = m.addMenu("&Import")
        s_m = self.import_menu.addMenu("Spectra")
        self.act_sp = s_m.addAction("Main...", self._show_spectrum_import_dialog)
        self.act_ca = s_m.addAction("Calibration...", self._show_calib_spec_import_dialog)
        l_m = self.import_menu.addMenu("Linelists")
        self.act_rl = l_m.addAction("Binary (.lin)...", self._show_linelist_import_dialog)
        self.act_cl = l_m.addAction("Text (.txt)...", self._show_cal_linelist_import_dialog)
        self.act_gw = self.import_menu.addAction("Generic Table...", self._show_table_import_wizard)
        self.act_lc = self.import_menu.addAction("Lamp Calibration...", self._show_lamp_cal_import_dialog)
        self.analysis_menu = m.addMenu("&Analysis")
        self.act_wm = self.analysis_menu.addAction("Matching...", self._show_match_dialog) 
        self.act_bf = self.analysis_menu.addAction("Branching Fractions...", self._launch_branching_fraction_analysis, "Ctrl+R")

    def set_file_loaded_state(self, state):
        actions = [self.import_menu, self.analysis_menu, self.act_sp, self.act_ca, self.act_rl, self.act_cl, self.act_gw, self.act_lc, self.act_wm, self.act_bf]
        for a in actions:
            a.setEnabled(state)

    def set_current_file(self, path):
        self.current_h5_file = path
        self.setWindowTitle(f"SAAS - Project: {path}")
        self.set_file_loaded_state(True)
        self._populate_tree_view()

    def _create_file(self):
        d = NewProjectDialog(self)
        if d.exec_():
            f, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "HDF5 Files (*.h5 *.hdf5)")
            if f:
                h5_manager.create_experiment_file(f, d.get_data())
                self.set_current_file(f)

    def _open_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "HDF5 Files (*.h5 *.hdf5)")
        if f:
            self.set_current_file(f)

    def _populate_tree_view(self):
        self.tree_model.clear()
        if not self.current_h5_file: 
            return
        with h5py.File(self.current_h5_file, 'r') as f:
            self._add_items_to_tree_recursively(self.tree_model.invisibleRootItem(), f)

    def _add_items_to_tree_recursively(self, parent, obj):
        for name, item in obj.items():
            child = QStandardItem(name)
            child.setData(item.name, Qt.UserRole)
            parent.appendRow(child)
            if isinstance(item, h5py.Group):
                self._add_items_to_tree_recursively(child, item)

    def _on_tree_item_selected(self, idx):
        path = idx.data(Qt.UserRole)
        self.data_table.hide()
        self.plot_tb.hide()
        self.plot_can.hide()
        with h5py.File(self.current_h5_file, 'r') as f:
            o = f[path]
            self.attr_view.setRowCount(len(o.attrs))
            for i, (k, v) in enumerate(o.attrs.items()):
                self.attr_view.setItem(i,0,QTableWidgetItem(k))
                self.attr_view.setItem(i,1,QTableWidgetItem(str(v)))
            if isinstance(o, h5py.Dataset):
                if o.parent and 'pandas_type' in o.parent.attrs:
                    df = h5_manager.read_hdf_table_robustly(self.current_h5_file, path)
                    self.data_table.setModel(PandasModel(df.head(200)))
                    self.data_table.show()
                elif o.ndim == 1:
                    self._plot_spectrum_data(o)
                    self.plot_tb.show()
                    self.plot_can.show()

    def _plot_spectrum_data(self, ds):
        d = ds[:] * ds.attrs.get('rdsclfct', 1.0)
        x = ds.attrs.get('wstart', 0.0) + np.arange(len(d)) * ds.attrs.get('delw', 1.0)
        self.plot_ax.clear()
        self.plot_ax.plot(x, d)
        self.plot_ax.grid(True)
        self.plot_can.draw()

    def _show_table_import_wizard(self):
        if ImportWizardDialog(self.current_h5_file, self).exec_():
            self._populate_tree_view()
    def _show_spectrum_import_dialog(self):
        if ImportSpectrumDialog(self.current_h5_file, self).exec_():
            self._populate_tree_view()
    def _show_calib_spec_import_dialog(self):
        if ImportCalibSpectrumDialog(self.current_h5_file, self).exec_():
            self._populate_tree_view()
    def _show_lamp_cal_import_dialog(self):
        if ImportLampCalDialog(self.current_h5_file, self).exec_():
            self._populate_tree_view()
    def _show_linelist_import_dialog(self):
        if ImportLinelistDialog(self.current_h5_file, self).exec_():
            self._populate_tree_view()
    def _show_cal_linelist_import_dialog(self):
        if ImportCalibratedLinelistDialog(self.current_h5_file, self).exec_():
            self._populate_tree_view()
    def _show_match_dialog(self):
        if WavenumberMatchDialog(self.current_h5_file, self).exec_():
            self._populate_tree_view()
    def _launch_branching_fraction_analysis(self):
        self.aw = AnalysisWindow(self.current_h5_file, self)
        self.aw.show()
        self.aw.destroyed.connect(self._populate_tree_view)

    def _show_tree_context_menu(self, pos):
        idx = self.tree_view.indexAt(pos)
        if not idx.isValid(): 
            return
        path = idx.data(Qt.UserRole)
        menu = QMenu()
        with h5py.File(self.current_h5_file, 'r') as f:
            full_act = None
            if isinstance(f[path], h5py.Dataset):
                full_act = menu.addAction("Full Table")
            del_act = menu.addAction("Delete")
            res = menu.exec_(self.tree_view.viewport().mapToGlobal(pos))
            if res == del_act:
                self._delete_selected_item(idx)
            elif full_act and res == full_act:
                self._open_full_table_viewer(path)

    def _open_full_table_viewer(self, path):
        df = h5_manager.read_hdf_table_robustly(self.current_h5_file, path)
        self.fv = FullTableWindow(df, path, self)
        self.fv.show()

    def _delete_selected_item(self, idx):
        if h5_manager.delete_object(self.current_h5_file, idx.data(Qt.UserRole)):
            self._populate_tree_view()