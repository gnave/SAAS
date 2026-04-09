# analysis_window.py (STABLE & READABLE)
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableView, QTreeView, QSplitter, QDockWidget, QPushButton, QLineEdit,
    QAbstractItemView, QSizePolicy, QHeaderView, QMenuBar, QAction, QMessageBox,
    QDialog, QDialogButtonBox, QInputDialog, QFormLayout, QTextEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QMenu, QStyle, QStyleOptionHeader, QApplication
)
from PyQt5.QtCore import Qt, QModelIndex, QAbstractTableModel, pyqtSignal, QItemSelectionModel, QRect
from PyQt5.QtGui import QColor, QFont, QStandardItemModel, QStandardItem, QIcon, QDoubleValidator, QBrush, QPainter 

import pandas as pd
import numpy as np
import h5py
import os
import ast
from datetime import date
import math

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import h5_manager
import analysis

# =============================================================================
# TABLE MODELS
# =============================================================================

class PandasTableModel(QAbstractTableModel):
    """Standard model for displaying generic DataFrames (like Energy Levels)."""
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = data

    def rowCount(self, parent=QModelIndex()):
        return self.df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self.df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role != Qt.DisplayRole:
            return None
            
        value = self.df.iloc[index.row(), index.column()]
        col_name = str(self.df.columns[index.column()]).strip().lower()

        if pd.isna(value):
            return ""

        if 'j_value' in col_name:
            try:
                return f"{float(value):.1f}"
            except Exception:
                pass
        elif 'parity' in col_name:
            try:
                return f"{int(float(value))}"
            except Exception:
                pass
        elif 'lifetime' in col_name:
            try:
                return f"{float(value):.2f}"
            except Exception:
                pass

        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
            
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.df.columns[section])
            else:
                return str(self.df.index[section])
        return None

class LineDataTableModel(PandasTableModel):
    """Specialized model for the main analysis table (Aggregation)."""
    def __init__(self, data: pd.DataFrame, highlight_df: pd.DataFrame, parent=None):
        super().__init__(data, parent)
        self.highlight_df = highlight_df

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            col_name = str(self.df.columns[section])
            if col_name == 'Mean Intensity':
                return 'Mean\nIntensity'
            if col_name == 'Mean Uncertainty':
                return 'Mean\nUncertainty'
            return col_name
        return super().headerData(section, orientation, role)
 
    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col_name = str(self.df.columns[index.column()])
        
        # Foreground: Grey text for excluded lines
        if role == Qt.ForegroundRole:
            if '\n' in col_name:
                spec = col_name.split('\n')[0]
                excluded_col = f"{spec}\nExcluded"
                if excluded_col in self.df.columns:
                    if self.df.iloc[index.row()].get(excluded_col) == True:
                        return QBrush(Qt.gray)
            return None

        # Background: Red for 3-sigma outliers
        if role == Qt.BackgroundRole:
            if not self.highlight_df.empty and col_name in self.highlight_df.columns:
                 is_outlier = self.highlight_df.iloc[index.row()].get(col_name, False)
                 if is_outlier:
                     return QBrush(QColor('#FFDDDD'))
            return None

        if role != Qt.DisplayRole:
            return None

        value = self.df.iloc[index.row(), index.column()]
        if col_name in ['wavenumber', 'intensity', 'lower_level_key']:
            return str(value)
        elif isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return ""
            if col_name == 'Mean Intensity' or '\nSNR' in col_name or '\nIntensity' in col_name:
                return f"{int(round(value))}"
            if col_name == 'Mean Uncertainty':
                return f"{(value * 100):.1f} %"
            return f"{value:.4f}"
        return str(value)

# =============================================================================
# HELPER FUNCTIONS & RESULTS DISPLAY
# =============================================================================

def _get_decimal_places(uncertainty: float) -> int:
    """Implements Rule of 20 for scientific rounding."""
    if not isinstance(uncertainty, (float, int, np.floating)):
        return 2
    if not np.isfinite(uncertainty):
        return 2
    if uncertainty <= 0:
        return 2
        
    mag = math.floor(math.log10(abs(uncertainty)))
    first_digit = uncertainty / (10**mag)
    
    if first_digit >= 2.0:
        return -int(mag)
    else:
        return -int(mag) + 1

class ResultsTableModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = data
        self.bf_dps = {}
        self.tp_dps = {}
        
        for i, row in self.df.iterrows():
            bf_val = row.get('Branching Fraction', 0)
            bf_unc_pct = row.get('BF Uncertainty (%)', 0)
            bf_abs_unc = abs(bf_val * (bf_unc_pct / 100.0))
            self.bf_dps[i] = _get_decimal_places(bf_abs_unc)

            tp_val = row.get('Trans. Prob. (10^6 s^-1)', 0)
            tp_unc_pct = row.get('Trans. Prob. Unc. (%)', 0)
            tp_abs_unc = abs(tp_val * (tp_unc_pct / 100.0))
            self.tp_dps[i] = _get_decimal_places(tp_abs_unc)

    def rowCount(self, parent=QModelIndex()):
        return self.df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self.df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role != Qt.DisplayRole:
            return None
            
        val = self.df.iloc[index.row(), index.column()]
        col = str(self.df.columns[index.column()])
        r_idx = self.df.index[index.row()]
        
        if pd.isna(val):
            return ""
            
        try:
            if col == 'Branching Fraction':
                dp = self.bf_dps.get(r_idx, 3)
                if dp >= 0:
                    return f"{val:.{max(0, dp)}f}"
                else:
                    return f"{round(val, dp):.0f}"
            elif col == 'Trans. Prob. (10^6 s^-1)':
                dp = self.tp_dps.get(r_idx, 1)
                if dp >= 0:
                    return f"{val:.{max(0, dp)}f}"
                else:
                    return f"{int(round(val, dp))}"
            elif col in ['BF Uncertainty (%)', 'Trans. Prob. Unc. (%)']:
                if val >= 10:
                    return f"{val:.0f}"
                return f"{val:.1f}"
            elif isinstance(val, (float, np.floating)):
                if 'Intensity' in col:
                    return f"{int(round(val))}"
                return f"{val:.4f}"
        except Exception:
            return f"{val:.3f}"
        return str(val)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.df.columns[section])
        return super().headerData(section, orientation, role)

class ResultsDisplayDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle("Calculation Results")
        self.setMinimumSize(1000, 500)
        
        layout = QVBoxLayout(self)
        self.table_view = QTableView()
        self.table_view.setModel(ResultsTableModel(df))
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        resid = df.attrs.get('residual_fraction', 0.0) * 100.0
        life = df.attrs.get('lifetime', 0.0)
        header_text = f"<b>Branching Fraction Results</b><br>Lifetime: {life:.3f} ns | Estimated Residual: {resid:.3f} %"
        
        header_label = QLabel(header_text)
        header_label.setTextFormat(Qt.RichText)
        layout.addWidget(header_label)
        layout.addWidget(self.table_view)
        
        button_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        button_layout.addWidget(copy_btn)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        button_layout.addWidget(button_box)
        layout.addLayout(button_layout)

    def _copy_to_clipboard(self):
        try:
            text = self.df.to_csv(sep='\t', index=False)
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "Success", "Results copied to clipboard!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to copy: {e}")

# =============================================================================
# UI COMPONENTS (HEADERS / DETAILS)
# =============================================================================

class MultiLevelHeaderView(QHeaderView):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setFixedHeight(50)

    def sectionSizeFromContents(self, logicalIndex):
        size = super().sectionSizeFromContents(logicalIndex)
        model = self.model()
        if not model:
            return size
        
        header_text = str(model.headerData(logicalIndex, self.orientation(), Qt.DisplayRole) or "")
        parts = header_text.split('\n')
        max_w = 0
        for p in parts:
            w = self.fontMetrics().boundingRect(p).width()
            if w > max_w:
                max_w = w
        
        size.setWidth(max(size.width(), max_w + 5))
        return size
 
    def paintSection(self, painter, rect, logicalIndex):
        model = self.model()
        if not model:
            super().paintSection(painter, rect, logicalIndex)
            return
            
        header_text = str(model.headerData(logicalIndex, self.orientation(), Qt.DisplayRole) or "")
        parts = header_text.split('\n')
        
        if len(parts) == 1:
            super().paintSection(painter, rect, logicalIndex)
            return
            
        top_text = parts[0]
        bottom_text = parts[1]
        v_idx = self.visualIndex(logicalIndex)
        
        l_v_idx = v_idx
        while l_v_idx > 0:
            prev_l = self.logicalIndex(l_v_idx - 1)
            prev_t = str(model.headerData(prev_l, self.orientation(), Qt.DisplayRole) or "")
            if prev_t.startswith(top_text + '\n'):
                l_v_idx -= 1
            else:
                break
                
        r_v_idx = v_idx
        while r_v_idx < self.count() - 1:
            next_l = self.logicalIndex(r_v_idx + 1)
            next_t = str(model.headerData(next_l, self.orientation(), Qt.DisplayRole) or "")
            if next_t.startswith(top_text + '\n'):
                r_v_idx += 1
            else:
                break
                
        l_off = sum(self.sectionSize(self.logicalIndex(i)) for i in range(l_v_idx, v_idx))
        span_x = rect.left() - l_off
        span_w = sum(self.sectionSize(self.logicalIndex(i)) for i in range(l_v_idx, r_v_idx + 1))
        h_h = rect.height() // 2
        
        top_r = QRect(span_x, rect.top(), span_w, h_h)
        bot_r = QRect(rect.left(), rect.top() + h_h, rect.width(), rect.height() - h_h)
        
        painter.save()
        painter.setClipRect(rect)
        opt = QStyleOptionHeader()
        self.initStyleOption(opt)
        opt.section = logicalIndex
        
        opt.rect = top_r
        opt.text = ""
        self.style().drawControl(QStyle.CE_HeaderSection, opt, painter, self)
        
        opt.rect = bot_r
        self.style().drawControl(QStyle.CE_HeaderSection, opt, painter, self)
        
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.setFont(self.font())
        painter.drawText(top_r, Qt.AlignCenter, top_text)
        painter.drawText(bot_r, Qt.AlignCenter, bottom_text)
        painter.restore()

class LineDetailsDialog(QDialog):
    def __init__(self, df, wavenumber, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Line Parameters: ~{wavenumber:.3f} cm⁻¹")
        self.setMinimumSize(900, 250)
        
        layout = QVBoxLayout(self)
        self.table_view = QTableView()
        self.table_view.setModel(PandasTableModel(df))
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        header_label = QLabel(f"<b>Raw Parameters for Line near {wavenumber:.3f} cm⁻¹</b>")
        layout.addWidget(header_label)
        layout.addWidget(self.table_view)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

# =============================================================================
# MAIN WINDOW CLASS
# =============================================================================

class AnalysisWindow(QMainWindow):
    def __init__(self, h5_filepath, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Analysis")
        self.setGeometry(100, 100, 1400, 800)
        self.h5_filepath = h5_filepath
        self.h5_manager = h5_manager
        self.analysis_module = analysis
        
        self.current_energy_levels_df = pd.DataFrame()
        self.current_previous_ids_df = pd.DataFrame()
        self.filtered_levels_df = pd.DataFrame()
        self.master_line_data_df = pd.DataFrame()
        self.highlight_df = pd.DataFrame()
        self.result_df = pd.DataFrame()
        self.current_upper_level_key = ""
        
        self.DATA_SOURCE_COLUMNS = {"Cal. Linelists": "Calibrated_Linelists", "Raw Spectrum": "Raw_Data"}
        
        self._create_menu_bar()
        self._create_main_layout()
        self._populate_initial_comboboxes()
        self._populate_data_source_table()
        self._clear_plot()
        
        self.main_splitter.setSizes([350, 1050])
        self.side_panel_splitter.setSizes([self.height() // 2, self.height() // 2])

    # --- UI CREATION ---

    def _create_menu_bar(self):
        m = self.menuBar()
        fm = m.addMenu("&File")
        
        open_act = QAction("Open Saved Analysis...", self)
        open_act.triggered.connect(self._on_open_analysis_triggered)
        fm.addAction(open_act)
        fm.addSeparator()
        
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        fm.addAction(exit_act)
        
        dm = m.addMenu("&Debug")
        diag_act = QAction("Run Diagnostics...", self)
        diag_act.triggered.connect(self._run_debug_diagnostics)
        dm.addAction(diag_act)

    def _create_main_layout(self):
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self._create_side_panel())
        self.main_splitter.addWidget(self._create_central_content_widget())
        self.setCentralWidget(self.main_splitter)

    def _create_side_panel(self):
        self.side_panel_splitter = QSplitter(Qt.Vertical)
        
        # Upper Panel: Levels
        lsc = QWidget()
        lsl = QVBoxLayout(lsc)
        self.level_file_combo = QComboBox()
        self.level_file_combo.addItem("Select Energy Level File...")
        self.level_file_combo.currentIndexChanged.connect(self._on_level_file_selected)
        lsl.addWidget(QLabel("Master Energy Level File:"))
        lsl.addWidget(self.level_file_combo)
        
        self.level_table = QTableView()
        self.level_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.level_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.level_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.level_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.level_table.clicked.connect(self._on_level_selected_in_table)
        lsl.addWidget(QLabel("Available Upper Levels:"))
        lsl.addWidget(self.level_table)
        
        # Lower Panel: Sources
        dsc = QWidget()
        dsl = QVBoxLayout(dsc)
        self.prev_id_combo = QComboBox()
        self.prev_id_combo.addItem("Select Previous IDs File...")
        self.prev_id_combo.currentIndexChanged.connect(self._on_prev_id_file_selected)
        dsl.addWidget(QLabel("Master Previous IDs File:"))
        dsl.addWidget(self.prev_id_combo)
        
        dshl = QHBoxLayout()
        dshl.addWidget(QLabel("Data Sources:"))
        self.edit_bands_btn = QPushButton("Edit Band Limits")
        self.edit_bands_btn.setMaximumWidth(120)
        self.edit_bands_btn.clicked.connect(self._on_edit_bands_btn_clicked)
        dshl.addWidget(self.edit_bands_btn)
        dsl.addLayout(dshl)
        
        self.data_source_table = QTableWidget()
        self.data_source_table.itemChanged.connect(self._on_data_source_table_item_changed)
        self.data_source_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_source_table.customContextMenuRequested.connect(self._show_data_source_context_menu)
        dsl.addWidget(self.data_source_table)
        
        acg = QWidget()
        acl = QVBoxLayout(acg)
        self.separate_plots_checkbox = QCheckBox("Plot Spectra in Separate Windows")
        acl.addWidget(self.separate_plots_checkbox)
        self.tolerance_edit = QLineEdit("0.1")
        self.tolerance_edit.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        acl.addWidget(QLabel("Wavenumber Tolerance:"))
        acl.addWidget(self.tolerance_edit)
        dsl.addWidget(acg)
        
        self.side_panel_splitter.addWidget(lsc)
        self.side_panel_splitter.addWidget(dsc)
        return self.side_panel_splitter

    def _create_central_content_widget(self):
        self.central_splitter = QSplitter(Qt.Vertical)
        
        tc = QWidget()
        tl = QVBoxLayout(tc)
        tl.setContentsMargins(0, 0, 0, 0)
        
        bl = QHBoxLayout()
        self.run_analysis_btn = QPushButton("Calculate BFs")
        self.run_analysis_btn.clicked.connect(self._calculate_clicked)
        self.save_results_btn = QPushButton("Save to HDF5")
        self.save_results_btn.clicked.connect(self._save_results_clicked)
        self.save_results_btn.setEnabled(False)
        self.copy_table_btn = QPushButton("Copy Table")
        self.copy_table_btn.clicked.connect(self._copy_table_to_clipboard)
        
        bl.addWidget(self.run_analysis_btn)
        bl.addWidget(self.save_results_btn)
        bl.addWidget(self.copy_table_btn)
        bl.addStretch()
        tl.addLayout(bl)
        
        self.line_data_table = QTableView()
        self.line_data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.line_data_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.custom_header = MultiLevelHeaderView(Qt.Horizontal, self.line_data_table)
        self.line_data_table.setHorizontalHeader(self.custom_header)
        self.line_data_table.setAlternatingRowColors(True)
        self.line_data_table.clicked.connect(self._on_line_selected)
        self.line_data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.line_data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.line_data_table.customContextMenuRequested.connect(self._show_line_table_context_menu)
        tl.addWidget(self.line_data_table)
        self.central_splitter.addWidget(tc)
        
        mpw = QWidget()
        pl = QVBoxLayout(mpw)
        pl.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        pl.addWidget(self.toolbar)
        pl.addWidget(self.canvas)
        self.central_splitter.addWidget(mpw)
        
        return self.central_splitter

    # --- ACTION HANDLERS ---

    def _normalize_intensities(self, row):
        try:
            norm_df = self.analysis_module.normalize_intensities_by_reference_line(self.master_line_data_df, row)
            self.master_line_data_df = self.analysis_module.add_weighted_averages(norm_df, self.h5_filepath)
            self._refresh_table_view(row)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _transfer_calibration(self, row, target):
        try:
            trans_df = self.analysis_module.transfer_calibration(self.master_line_data_df, row, target, self.h5_filepath)
            self.master_line_data_df = self.analysis_module.add_weighted_averages(trans_df, self.h5_filepath)
            self._refresh_table_view(row)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _toggle_exclusion(self, row, spec):
        col = f"{spec}\nExcluded"
        if col not in self.master_line_data_df.columns:
            self.master_line_data_df[col] = False
        
        current_val = self.master_line_data_df.iloc[row].get(col, False)
        self.master_line_data_df.at[self.master_line_data_df.index[row], col] = not bool(current_val)
        
        self.master_line_data_df = self.analysis_module.add_weighted_averages(self.master_line_data_df, self.h5_filepath)
        self._refresh_table_view(row)

    # --- MENU & CALLBACKS ---

    def _show_line_table_context_menu(self, pos):
        idx = self.line_data_table.indexAt(pos)
        if not idx.isValid():
            return
            
        menu = QMenu()
        norm_act = menu.addAction("Normalize to Reference")
        det_act = menu.addAction("Show Raw Parameters")
        
        # Filter spectrum names
        specs = []
        for c in self.master_line_data_df.columns:
            if '\nSNR' in c:
                specs.append(c.split('\n')[0])
        specs = sorted(list(set(specs)))
        
        t_menu = menu.addMenu("Transfer Calibration To...")
        t_actions = {}
        for s in specs:
            t_actions[t_menu.addAction(f"To: {s}")] = s
            
        e_menu = menu.addMenu("Toggle Exclusion...")
        e_actions = {}
        for s in specs:
            label = f"Spectrum: {s}"
            is_ex = bool(self.master_line_data_df.iloc[idx.row()].get(f"{s}\nExcluded", False))
            if is_ex:
                label += " (Excluded)"
            e_actions[e_menu.addAction(label)] = s
            
        act = menu.exec_(self.line_data_table.viewport().mapToGlobal(pos))
        
        if act == norm_act:
            self._normalize_intensities(idx.row())
        elif act == det_act:
            self._show_line_details(idx.row())
        elif act in t_actions:
            self._transfer_calibration(idx.row(), t_actions[act])
        elif act in e_actions:
            self._toggle_exclusion(idx.row(), e_actions[act])

    def _show_data_source_context_menu(self, pos):
        idx = self.data_source_table.indexAt(pos)
        if not idx.isValid():
            return
            
        sn = self.data_source_table.verticalHeaderItem(idx.row()).text()
        m = QMenu()
        a = m.addAction(f"Edit Band Limits for {sn}")
        if m.exec_(self.data_source_table.viewport().mapToGlobal(pos)) == a:
            self._edit_spectrum_bands(sn)

    def _on_edit_bands_btn_clicked(self):
        sel = self.data_source_table.selectedRanges()
        if sel:
            s_name = self.data_source_table.verticalHeaderItem(sel[0].topRow()).text()
            self._edit_spectrum_bands(s_name)
        else:
            specs = []
            for r in range(self.data_source_table.rowCount()):
                specs.append(self.data_source_table.verticalHeaderItem(r).text())
            if not specs:
                return
            i, ok = QInputDialog.getItem(self, "Select Spectrum", "Edit bands for:", specs, 0, False)
            if ok and i:
                self._edit_spectrum_bands(i)

    def _edit_spectrum_bands(self, s_name):
        p = f"/Spectra/{s_name}/Raw_Data/spectrum"
        try:
            with h5py.File(self.h5_filepath, 'a') as f:
                a = f[p].attrs
                lo = a.get('bandlo', a.get('wstart', 0.0))
                hi = a.get('bandhi', a.get('wend', a.get('wstart', 0.0) + 30000.0))
                
                d = QDialog(self)
                d.setWindowTitle(f"Edit Bands: {s_name}")
                fl = QFormLayout(d)
                elo = QLineEdit(str(lo))
                ehi = QLineEdit(str(hi))
                fl.addRow("bandlo:", elo)
                fl.addRow("bandhi:", ehi)
                
                bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                bb.accepted.connect(d.accept)
                bb.rejected.connect(d.reject)
                fl.addRow(bb)
                
                if d.exec_() == QDialog.Accepted:
                    f[p].attrs['bandlo'] = float(elo.text())
                    f[p].attrs['bandhi'] = float(ehi.text())
                    sm = self.level_table.selectionModel()
                    if sm and sm.hasSelection():
                        self._on_level_selected_in_table()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # --- DATA LOADING ---

    def _on_level_file_selected(self):
        fn = self.level_file_combo.currentText()
        if fn == "Select Energy Level File...":
            self.level_table.setModel(None)
            self.current_energy_levels_df = pd.DataFrame()
            return
        try:
            path = f"/Levels/{fn}/table"
            self.current_energy_levels_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
            df = self.current_energy_levels_df
            if not df.empty and 'key' in df.columns:
                df['key'] = df['key'].astype(str).str.replace('*', '', regex=False).str.strip()
                self.filtered_levels_df = df[df['lifetime'] > 0].copy()
                cols = []
                for c in ['key','energy','j_value','parity','lifetime','lifetime_unc_frac']:
                    if c in self.filtered_levels_df.columns:
                        cols.append(c)
                self.level_table.setModel(PandasTableModel(self.filtered_levels_df[cols]))
                self.level_table.resizeColumnsToContents()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_prev_id_file_selected(self):
        fn = self.prev_id_combo.currentText()
        if fn == "Select Previous IDs File...":
            self.current_previous_ids_df = pd.DataFrame()
            self.line_data_table.setModel(None)
            return
        try:
            path = f"/Previous_Identifications/{fn}/table"
            self.current_previous_ids_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
            df = self.current_previous_ids_df
            if not df.empty and 'upper_level_key' in df.columns:
                df['normalized_key'] = df['upper_level_key'].astype(str).str.replace('*', '', regex=False).str.strip()
            
            sm = self.level_table.selectionModel()
            if sm and sm.hasSelection():
                self._on_level_selected_in_table()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_level_selected_in_table(self):
        sm = self.level_table.selectionModel()
        if not sm:
            return
        if not sm.hasSelection():
            self.current_upper_level_key = ""
            return
            
        row = sm.selectedRows()[0].row()
        self.current_upper_level_key = str(self.filtered_levels_df.iloc[row].get('key', ''))
        self._populate_line_data_table(self.current_upper_level_key)

    def _populate_line_data_table(self, key):
        df_ids = self.current_previous_ids_df
        if df_ids.empty or 'normalized_key' not in df_ids.columns:
            return
            
        lines = df_ids[df_ids['normalized_key'] == key]
        if lines.empty:
            self.line_data_table.setModel(None)
            return
        try:
            tol = float(self.tolerance_edit.text() or 0.1)
            agg_df = self.analysis_module.aggregate_observed_data_for_display(
                self.h5_filepath, 
                lines.drop(columns=['normalized_key']), 
                self._get_checked_data_paths(), 
                tol
            )
            self.master_line_data_df = self.analysis_module.add_weighted_averages(agg_df, self.h5_filepath)
            self._refresh_table_view()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _refresh_table_view(self, restore_row=None):
        self.highlight_df = self.analysis_module.calculate_outliers(self.master_line_data_df, self.h5_filepath)
        model = LineDataTableModel(self.master_line_data_df, self.highlight_df)
        self.line_data_table.setModel(model)
        self._format_table_columns()
        
        if restore_row is not None:
            idx = model.index(restore_row, 0)
            if idx.isValid():
                self.line_data_table.setCurrentIndex(idx)
                self.line_data_table.selectionModel().select(idx, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                self._on_line_selected(idx)

    def _format_table_columns(self):
        m = self.line_data_table.model()
        if not m:
            return
        for c in range(m.columnCount()):
            col_name = str(m.df.columns[c])
            if "Width" in col_name or "Excluded" in col_name:
                self.line_data_table.setColumnHidden(c, True)
            elif c < 3 or "Mean" in str(m.headerData(c, Qt.Horizontal)):
                self.line_data_table.resizeColumnToContents(c)
            else:
                self.line_data_table.setColumnWidth(c, 55)

    # --- SPECTRUM SOURCES ---

    def _populate_data_source_table(self):
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Spectra' not in f:
                    return
                specs = sorted(list(f['/Spectra'].keys()))
                cols = list(self.DATA_SOURCE_COLUMNS.keys())
                self.data_source_table.setRowCount(len(specs))
                self.data_source_table.setColumnCount(len(cols))
                self.data_source_table.setVerticalHeaderLabels(specs)
                self.data_source_table.setHorizontalHeaderLabels(cols)
                for r, sn in enumerate(specs):
                    for c, cl in enumerate(cols):
                        bp = f"/Spectra/{sn}/{self.DATA_SOURCE_COLUMNS[cl]}"
                        item = QTableWidgetItem()
                        item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                        item.setBackground(QBrush(QColor('lightGray')))
                        if bp in f:
                            group = f[bp]
                            p = ""
                            if self.DATA_SOURCE_COLUMNS[cl] == "Raw_Data":
                                p = f"{bp}/spectrum"
                            elif len(group.keys()) > 0:
                                p = f"{bp}/{list(group.keys())[0]}/table"
                            
                            if p and p in f:
                                a = f[p].attrs
                                lo = a.get('bandlo', a.get('wstart', 0.0))
                                hi = a.get('bandhi', a.get('wend', lo + 30000.0))
                                item.setToolTip(f"Bands: {float(lo):.1f} to {float(hi):.1f}")
                                item.setData(Qt.UserRole, p)
                                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                                item.setBackground(QBrush(QColor('white')))
                                item.setCheckState(Qt.Unchecked)
                        self.data_source_table.setItem(r, c, item)
            self.data_source_table.resizeColumnsToContents()
        except Exception:
            pass

    def _on_data_source_table_item_changed(self, it):
        self.data_source_table.blockSignals(True)
        try:
            header = self.data_source_table.horizontalHeaderItem(it.column()).text()
            if header == "Cal. Linelists" and it.checkState() == Qt.Checked:
                for c in range(self.data_source_table.columnCount()):
                    h_text = self.data_source_table.horizontalHeaderItem(c).text()
                    if h_text == "Raw Spectrum":
                        ri = self.data_source_table.item(it.row(), c)
                        if ri:
                            ri.setCheckState(Qt.Checked)
        finally:
            self.data_source_table.blockSignals(False)
            
        sm = self.level_table.selectionModel()
        if sm and sm.hasSelection():
            self._on_level_selected_in_table()

    def _get_checked_data_paths(self):
        paths = []
        for r in range(self.data_source_table.rowCount()):
            for c in range(self.data_source_table.columnCount()):
                it = self.data_source_table.item(r, c)
                if it and it.checkState() == Qt.Checked:
                    paths.append(it.data(Qt.UserRole))
        return paths

    def _populate_initial_comboboxes(self):
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Levels' in f:
                    self.level_file_combo.addItems(list(f['/Levels'].keys()))
                if '/Previous_Identifications' in f:
                    self.prev_id_combo.addItems(list(f['/Previous_Identifications'].keys()))
        except Exception:
            pass

    # --- PLOTTING ---

    def _on_line_selected(self, idx):
        if not idx.isValid():
            return
        if self.master_line_data_df.empty:
            return
        wn = self.master_line_data_df.iloc[idx.row()].get('wavenumber')
        try:
            self._update_plot(float(wn), self._get_checked_data_paths(), self.master_line_data_df.iloc[idx.row()])
        except Exception:
            self._clear_plot()

    def _update_plot(self, t_wn, paths, ld=None):
        self.figure.clear()
        cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        raw_ps = [p for p in paths if 'Raw_Data' in p]
        linelist_paths = [p for p in paths if 'Calibrated' in p or 'Identified' in p]
        max_fwhm = 0.0
        tol = float(self.tolerance_edit.text() or 0.1)
        
        for p in linelist_paths:
            try:
                df = h5_manager.read_hdf_table_robustly(self.h5_filepath, p)
                df['w'] = pd.to_numeric(df['wavenumber'], errors='coerce')
                d = np.abs(df['w'] - t_wn)
                if d.min() <= tol:
                    max_fwhm = max(max_fwhm, df.loc[d.idxmin(), 'width'])
            except Exception:
                pass
                
        rng = (5.0 * (max_fwhm/1000.0)) if max_fwhm > 0 else 5.0
        loaded = False
        ax = self.figure.add_subplot(1, 1, 1)
        for i, p in enumerate(raw_ps):
            try:
                sn = p.split('/')[2]
                ex_key = f"{sn}\nExcluded"
                ex = bool(ld is not None and ld.get(ex_key, False))
                with h5py.File(self.h5_filepath, 'r') as f:
                    ds = f[p]
                    a = ds.attrs
                    d_raw = ds[:] * a.get('rdsclfct', 1.0)
                    x = (a.get('wstart', 0.0) + np.arange(len(d_raw)) * a.get('delw', 1.0)) * (1.0 + a.get('wavcorr', 0.0))
                    m = (x >= t_wn - rng) & (x <= t_wn + rng)
                    if np.any(m):
                        c = 'lightgray' if ex else cc[i % len(cc)]
                        al = 0.5 if ex else 0.7
                        ax.plot(x[m], d_raw[m], color=c, alpha=al, label=sn)
                        ax.grid(True)
                        loaded = True
            except Exception:
                pass
        if loaded:
            self.figure.tight_layout()
            self.canvas.draw()

    def _clear_plot(self):
        self.figure.clear()
        self.canvas.draw()

    # --- CALCULATION & SAVE ---

    def _calculate_clicked(self):
        if self.master_line_data_df.empty:
            return
        cdf = pd.DataFrame()
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Calculations' in f:
                    keys = list(f['/Calculations'].keys())
                    if keys:
                        cdf = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, f"/Calculations/{keys[0]}/table")
        except Exception:
            pass
        self.result_df = self.analysis_module.calculate_branching_fractions(
            self.master_line_data_df, 
            self.current_upper_level_key, 
            self.current_energy_levels_df, 
            cdf, 
            float(self.tolerance_edit.text() or 0.1)
        )
        if not self.result_df.empty:
            self.save_results_btn.setEnabled(True)
            self.rd = ResultsDisplayDialog(self.result_df, self)
            self.rd.show()

    def _save_results_clicked(self):
        n, ok = QInputDialog.getText(self, "Save", "Name:", text=f"BF_{self.current_upper_level_key}_{date.today().strftime('%Y%m%d')}")
        if ok and n:
            try:
                self.h5_manager.create_group_if_not_exists(self.h5_filepath, "/Branching_Fraction_Analyses")
                p = f"/Branching_Fraction_Analyses/{n}"
                self.h5_manager.add_pandas_table(self.h5_filepath, p, "calculation_input_data", self.master_line_data_df)
                self.h5_manager.add_pandas_table(self.h5_filepath, p, "branching_fraction_results", self.result_df)
                
                meta = {
                    'upper_level_key': self.current_upper_level_key,
                    'source_level_file': self.level_file_combo.currentText(),
                    'source_previous_ids_file': self.prev_id_combo.currentText(),
                    'source_linelists': str(self._get_checked_data_paths()),
                    'wavenumber_tolerance': self.tolerance_edit.text()
                }
                self.h5_manager.attach_metadata_to_group(self.h5_filepath, p, meta)
                QMessageBox.information(self, "Saved", "Analysis saved.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _copy_table_to_clipboard(self):
        try:
            dfc = self.master_line_data_df.copy()
            new_cols = []
            for c in dfc.columns:
                new_cols.append(str(c).replace('\n', ' '))
            dfc.columns = new_cols
            QApplication.clipboard().setText(dfc.to_csv(sep='\t', index=False))
            QMessageBox.information(self, "Copied", "Table copied.")
        except Exception:
            pass

    # --- PREVIOUS ANALYSIS LOADING ---

    def _on_open_analysis_triggered(self):
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if "/Branching_Fraction_Analyses" not in f:
                    return
                als = sorted(list(f["/Branching_Fraction_Analyses"].keys()))
            i, ok = QInputDialog.getItem(self, "Open Analysis", "Select:", als, 0, False)
            if ok and i:
                self._load_saved_analysis(i)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _load_saved_analysis(self, name):
        p = f"/Branching_Fraction_Analyses/{name}"
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                a = f[p].attrs
                self.tolerance_edit.setText(str(a.get('wavenumber_tolerance', "0.1")))
                self.level_file_combo.setCurrentText(str(a.get('source_level_file', '')))
                self._on_level_file_selected()
                self.prev_id_combo.setCurrentText(str(a.get('source_previous_ids_file', '')))
                self._on_prev_id_file_selected()
                k = str(a.get('upper_level_key', ''))
                m = self.level_table.model()
                for r in range(m.rowCount()):
                    if str(m.index(r, 0).data()) == k:
                        self.level_table.selectRow(r)
                        self.current_upper_level_key = k
                        break
                try:
                    checked = ast.literal_eval(str(a.get('source_linelists', '[]')))
                except:
                    checked = []
                for r in range(self.data_source_table.rowCount()):
                    for c in range(self.data_source_table.columnCount()):
                        it = self.data_source_table.item(r, c)
                        if it:
                            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in checked else Qt.Unchecked)
            self.master_line_data_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, f"{p}/calculation_input_data/table")
            self._refresh_table_view()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _show_line_details(self, row_index):
        line_data = self.master_line_data_df.iloc[row_index]
        wn = pd.to_numeric(line_data.get('wavenumber'), errors='coerce')
        if pd.isna(wn):
            return
        tol = float(self.tolerance_edit.text() or 0.1)
        paths = [p for p in self._get_checked_data_paths() if ('Identified' in p or 'Calibrated' in p)]
        details = []
        for p in paths:
            try:
                df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, p)
                df['w'] = pd.to_numeric(df['wavenumber'], errors='coerce')
                d = np.abs(df['w'] - wn)
                if not d.empty and d.min() <= tol:
                    r_idx = np.argmin(d)
                    r_dict = {'Spectrum': p.split('/')[2]}
                    r_dict.update(df.iloc[r_idx].drop(['w', 'index'], errors='ignore').to_dict())
                    details.append(r_dict)
            except Exception:
                pass
        if details:
            self.dd = LineDetailsDialog(pd.DataFrame(details), wn, self)
            self.dd.show()
        else:
            QMessageBox.information(self, "No Details", "No matching lines found.")

    def _run_debug_diagnostics(self):
        d = QDialog(self)
        d.setWindowTitle("Diagnostics")
        l = QVBoxLayout(d)
        t = QTextEdit()
        t.setReadOnly(True)
        t.setText(f"Levels: {len(self.filtered_levels_df)}\nLines: {len(self.master_line_data_df)}")
        l.addWidget(t)
        d.exec_()