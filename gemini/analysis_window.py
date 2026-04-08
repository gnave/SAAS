# analysis_window.py (FULLY DOCUMENTED)
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableView, QTreeView, QSplitter, QDockWidget, QPushButton, QLineEdit,
    QAbstractItemView, QSizePolicy, QHeaderView, QMenuBar, QAction, QMessageBox,
    QDialog, QDialogButtonBox, QInputDialog, QFormLayout, QTextEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QMenu, QStyle, QStyleOptionHeader, QApplication
)
from PyQt5.QtCore import Qt, QModelIndex, QAbstractTableModel, pyqtSignal, QItemSelectionModel, QRect # <--- Added QRect
from PyQt5.QtGui import QColor, QFont, QStandardItemModel, QStandardItem, QIcon, QDoubleValidator, QBrush, QPainter 

import pandas as pd
import numpy as np
import h5py
import os
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

class PandasTableModel(QAbstractTableModel):
    """A standard Qt Table Model for displaying a Pandas DataFrame."""
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = data

    def rowCount(self, parent=QModelIndex()):
        return self.df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self.df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        value = self.df.iloc[index.row(), index.column()]
        col_name = str(self.df.columns[index.column()]).strip().lower()

        # Safely handle missing values
        if pd.isna(value):
            return ""

        # Force specific formatting for j_value (1 decimal place)
        if 'j_value' in col_name:
            try:
                return f"{float(value):.1f}"
            except (ValueError, TypeError):
                pass
                
        # Force specific formatting for parity (integer 0 or 1)
        elif 'parity' in col_name:
            try:
                return f"{int(float(value))}"
            except (ValueError, TypeError):
                pass
                
        # Force specific formatting for lifetime and its uncertainty (2 decimal places)
        elif 'lifetime' in col_name:
            try:
                return f"{float(value):.2f}"
            except (ValueError, TypeError):
                pass

        # Default formatting for all other floats (4 decimal places)
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"

        # Fallback for strings
        return str(value)
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.df.columns[section])
            if orientation == Qt.Vertical:
                return str(self.df.index[section])
        return None

class LineDataTableModel(PandasTableModel):
    """
    A specialized table model for the main analysis table.

    This model provides custom data formatting for different columns to ensure
    readability and correctness. For example, it ensures the base 'wavenumber'
    and 'intensity' are displayed as raw strings, while formatting calculated
    values like mean uncertainty as percentages.
    """
    def __init__(self, data: pd.DataFrame, highlight_df: pd.DataFrame, parent=None):
        super().__init__(data, parent)
        self.highlight_df = highlight_df

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Intercepts column names to trigger the multi-level header."""
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            col_name = str(self.df.columns[section])
            
            # Inject newlines so our custom header splits them!
            if col_name == 'Mean Intensity':
                return 'Mean\nIntensity'
            if col_name == 'Mean Uncertainty':
                return 'Mean\nUncertainty'
                
            return col_name
            
        # Fall back to the default behavior for row numbers, etc.
        return super().headerData(section, orientation, role)
 
    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        """Overrides the base data method to provide custom formatting."""
        if not index.isValid():
            return None

        col_name = str(self.df.columns[index.column()])
        
        # Render excluded cells as grey text ---
        if role == Qt.ForegroundRole:
            if '\n' in col_name:
                spectrum_name = col_name.split('\n')[0]
                excluded_col = f"{spectrum_name}\nExcluded"
                if excluded_col in self.df.columns:
                    if self.df.iloc[index.row()].get(excluded_col) == True:
                        return QBrush(Qt.gray)
            return None

        # Render outlier cells with a red background ---
        if role == Qt.BackgroundRole:
            if not self.highlight_df.empty and col_name in self.highlight_df.columns:
                 is_outlier = self.highlight_df.iloc[index.row()].get(col_name, False)
                 if is_outlier:
                     return QBrush(QColor('#FFDDDD'))  # A light red
            return None


        if role != Qt.DisplayRole:
            return None

        value = self.df.iloc[index.row(), index.column()]
        
        # Rule 1: Always treat the original 'intensity', 'wavenumber', and 'key' columns as text
        # to preserve their original formatting from the source file.
        if col_name in['wavenumber', 'intensity', 'lower_level_key']:
            return str(value)
        
        # Rule 2: Handle formatting for calculated float columns.
        elif isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return ""
            if col_name == 'Mean Intensity':
                return f"{int(round(value))}"
            if col_name == 'Mean Uncertainty':
                # Display the fractional uncertainty as a formatted percentage.
                return f"{(value * 100):.1f} %"
            if '\nSNR' in col_name or '\nIntensity' in col_name:
                # Display individual spectrum measurements as integers.
                return f"{int(round(value))}"
            # Default formatting for any other floats.
            return f"{value:.4f}"

        # Rule 3: Default fallback for any other data types.
        else:
            return str(value)

    def flags(self, index: QModelIndex):
        """Returns the item flags for the given index. This table is read-only."""
        return super().flags(index)

class MultiLevelHeaderView(QHeaderView):
    """
    A custom QHeaderView that supports spanning multi-level headers.
    It hooks into the native paintSection method to draw filled backgrounds
    while seamlessly spanning text across adjacent grouped columns.
    """
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setFixedHeight(50)  # Make it taller to fit two lines of text cleanly

    
    def sectionSizeFromContents(self, logicalIndex):
        """
        Overrides the default size calculation to account for our custom drawn text.
        """
        size = super().sectionSizeFromContents(logicalIndex)
        
        model = self.model()
        if not model:
            return size
            
        header_text = str(model.headerData(logicalIndex, self.orientation(), Qt.DisplayRole) or "")
        parts = header_text.split('\n')
        
        font_metrics = self.fontMetrics()
        max_text_width = 0
        
        for part in parts:
            text_width = font_metrics.boundingRect(part).width()
            if text_width > max_text_width:
                max_text_width = text_width
                
        # Add 5 pixels of padding
        required_header_width = max_text_width + 5
        
        # Return whichever is wider: the table data or our custom header text
        size.setWidth(max(size.width(), required_header_width))
        
        return size
 
    def paintSection(self, painter, rect, logicalIndex):
        model = self.model()
        if not model:
            super().paintSection(painter, rect, logicalIndex)
            return

        header_text = str(model.headerData(logicalIndex, self.orientation(), Qt.DisplayRole) or "")
        parts = header_text.split('\n')

        # If it's a standard single-line header (like 'wavenumber' or 'intensity')
        if len(parts) == 1:
            super().paintSection(painter, rect, logicalIndex)
            return

        # --- Multi-level header logic ---
        top_text, bottom_text = parts[0], parts[1]
        visual_index = self.visualIndex(logicalIndex)

        # 1. Look left to find where this top-level group starts
        left_v_index = visual_index
        while left_v_index > 0:
            prev_l_index = self.logicalIndex(left_v_index - 1)
            prev_text = str(model.headerData(prev_l_index, self.orientation(), Qt.DisplayRole) or "")
            if prev_text.startswith(top_text + '\n'):
                left_v_index -= 1
            else:
                break
                
        # 2. Look right to find where this top-level group ends
        right_v_index = visual_index
        while right_v_index < self.count() - 1:
            next_l_index = self.logicalIndex(right_v_index + 1)
            next_text = str(model.headerData(next_l_index, self.orientation(), Qt.DisplayRole) or "")
            if next_text.startswith(top_text + '\n'):
                right_v_index += 1
            else:
                break

        # 3. Safely calculate coordinates relative to the current cell (fixes scrolling bugs)
        left_offset = sum(self.sectionSize(self.logicalIndex(i)) for i in range(left_v_index, visual_index))
        span_x = rect.left() - left_offset
        span_width = sum(self.sectionSize(self.logicalIndex(i)) for i in range(left_v_index, right_v_index + 1))
        
        half_height = rect.height() // 2

        # Create the bounding boxes for the top and bottom sections
        top_rect = QRect(span_x, rect.top(), span_width, half_height)
        bottom_rect = QRect(rect.left(), rect.top() + half_height, rect.width(), rect.height() - half_height)

        # --- Painting ---
        painter.save()
        
        # Clip the painter to the current column's boundary to create the seamless merge effect
        painter.setClipRect(rect)

        opt = QStyleOptionHeader()
        self.initStyleOption(opt)
        opt.section = logicalIndex

        # Paint the Top Spanning Background (leaving text blank to handle OS quirks)
        opt.rect = top_rect
        opt.text = "" 
        self.style().drawControl(QStyle.CE_HeaderSection, opt, painter, self)

        # Paint the Bottom Individual Background
        opt.rect = bottom_rect
        opt.text = ""
        self.style().drawControl(QStyle.CE_HeaderSection, opt, painter, self)

        # --- Explicitly Draw Text Manually ---
        painter.setPen(self.palette().color(self.foregroundRole()))
        font = self.font()
        # Optional: Set bold to match standard header styles on some OSs
        # font.setBold(True) 
        painter.setFont(font)
        
        # Center the Spectrum Name perfectly across the combined width
        painter.drawText(top_rect, Qt.AlignCenter | Qt.TextShowMnemonic, top_text)
        
        # Center the 'Intensity' or 'SNR' in the bottom row
        painter.drawText(bottom_rect, Qt.AlignCenter | Qt.TextShowMnemonic, bottom_text)

        painter.restore()

def _get_decimal_places(uncertainty: float) -> int:
    """
    Calculates the decimal position of the uncertainty's last significant digit.
    Rule of 20: 
    - If 1st sig digit is 1, keep 2 sig digits.
    - If 1st sig digit is >= 2, keep 1 sig digit.
    """
    if not isinstance(uncertainty, (float, int, np.floating)) or not np.isfinite(uncertainty) or uncertainty <= 0:
        return 2

    # Get the order of magnitude of the first significant digit
    # e.g., 0.005 -> -3 | 150.0 -> 2
    order_of_magnitude = math.floor(math.log10(abs(uncertainty)))
    
    # Scale uncertainty so the first digit is in the ones place
    # e.g., 0.00506 -> 5.06
    first_digits = uncertainty / (10**order_of_magnitude)

    if first_digits >= 2.0:
        # First sig digit is 2-9: Keep 1 sig digit. 
        # Precision is at the 'order_of_magnitude' decimal place.
        return -int(order_of_magnitude)
    else:
        # First sig digit is 1: Keep 2 sig digits.
        # Precision is one place further.
        return -int(order_of_magnitude) + 1
    
class ResultsTableModel(QAbstractTableModel):
    """A specialized model for the results table with custom formatting."""
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = data
        self.bf_dps = {}
        self.tp_dps = {}
        
        for i, row in self.df.iterrows():
            # 1. Calculate Absolute Uncertainty for Branching Fraction
            bf_val = row.get('Branching Fraction', 0)
            bf_unc_pct = row.get('BF Uncertainty (%)', 0)
            bf_abs_unc = abs(bf_val * (bf_unc_pct / 100.0))
            self.bf_dps[i] = _get_decimal_places(bf_abs_unc)

            # 2. Calculate Absolute Uncertainty for Transition Probability
            tp_val = row.get('Trans. Prob. (10^6 s^-1)', 0)
            tp_unc_pct = row.get('Trans. Prob. Unc. (%)', 0)
            tp_abs_unc = abs(tp_val * (tp_unc_pct / 100.0))
            self.tp_dps[i] = _get_decimal_places(tp_abs_unc)

    def rowCount(self, parent=QModelIndex()): return self.df.shape[0]
    def columnCount(self, parent=QModelIndex()): return self.df.shape[1]
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return str(self.df.columns[section])
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        value = self.df.iloc[index.row(), index.column()]
        col_name = str(self.df.columns[index.column()])
        row_idx = self.df.index[index.row()]
        
        if pd.isna(value): return ""

        try:
            # FORMATTING BRANCHING FRACTION
            if col_name == 'Branching Fraction':
                dp = self.bf_dps.get(row_idx, 3)
                if dp < 0: return f"{round(value, dp):.0f}"
                return f"{value:.{max(0, dp)}f}"
            
            # FORMATTING TRANSITION PROBABILITY
            elif col_name == 'Trans. Prob. (10^6 s^-1)':
                dp = self.tp_dps.get(row_idx, 1)
                if dp < 0: return f"{int(round(value, dp))}"
                return f"{value:.{max(0, dp)}f}"
                
            # FORMATTING PERCENTAGE COLUMNS
            # Percentages themselves usually only need 1 decimal place 
            # unless they are very small.
            elif col_name in ['BF Uncertainty (%)', 'Trans. Prob. Unc. (%)']:
                if value >= 10: return f"{value:.0f}"
                return f"{value:.1f}"

            # DEFAULT FOR OTHER FLOATS
            elif isinstance(value, (float, np.floating)):
                if 'Intensity' in col_name: return f"{int(round(value))}"
                return f"{value:.4f}"

        except Exception:
            return f"{value:.3f}"

        return str(value)


class ResultsDisplayDialog(QDialog):
    """A dialog window to display a DataFrame in a QTableView, used for showing results."""
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df  # Store a reference to the dataframe
        self.setWindowTitle("Calculation Results")
        self.setMinimumSize(1000, 500)
        
        layout = QVBoxLayout(self)
        self.table_view = QTableView()
        self.table_view.setModel(ResultsTableModel(df))
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # Extract metadata to show residuals and lifetime
        resid = df.attrs.get('residual_fraction', 0.0) * 100.0
        lifetime = df.attrs.get('lifetime', 0.0)
        
        header_text = f"<b>Branching Fraction Calculation Results</b><br>" \
                      f"Lifetime: {lifetime:.3f} ns | Estimated Residual (Unobserved): {resid:.3f} %"
        
        header_label = QLabel(header_text)
        header_label.setTextFormat(Qt.RichText)
        layout.addWidget(header_label)
        
        layout.addWidget(self.table_view)
        
        # --- NEW: Add the Copy button next to the OK button ---
        button_layout = QHBoxLayout()
        
        copy_btn = QPushButton("Copy to Clipboard (for Excel)")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        button_layout.addWidget(copy_btn)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        button_layout.addWidget(button_box)
        
        layout.addLayout(button_layout)
        # ------------------------------------------------------

    def _copy_to_clipboard(self):
        """Exports the DataFrame to the system clipboard for Excel/Spreadsheets."""
        try:
            # Convert DataFrame to a tab-separated string (perfect for Excel)
            text = self.df.to_csv(sep='\t', index=False)
            
            # Send it to the system clipboard
            QApplication.clipboard().setText(text)
            
            QMessageBox.information(self, "Success", "Results copied to clipboard!\nYou can now paste them directly into Excel or Sheets.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to copy to clipboard:\n{e}")

class LineDetailsDialog(QDialog):
    """A dialog window to display all raw parameters of a line across all spectra."""
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

class PlotPopupDialog(QDialog):
    """A generic dialog for displaying a Matplotlib plot in a separate window."""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title); self.setMinimumSize(600, 400); layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4), dpi=100); self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self); layout.addWidget(self.toolbar); layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

class AnalysisWindow(QMainWindow):
    """
    The main window for the interactive branching fraction analysis workflow.

    This window provides controls for selecting atomic data, experimental spectra,
    and running calculations. It displays an aggregated table of all relevant data
    and allows for interactive plotting of spectral lines.
    """
    def __init__(self, h5_filepath, parent=None):
        """Initializes the analysis window and its components."""
        super().__init__(parent)
        self.setWindowTitle("Interactive Branching Fraction Analysis"); self.setGeometry(100, 100, 1400, 800)

        # --- Data Members ---
        self.h5_filepath = h5_filepath
        self.h5_manager = h5_manager
        self.analysis_module = analysis
        self.current_energy_levels_df = pd.DataFrame()  # Master energy level data
        self.current_previous_ids_df = pd.DataFrame()   # Master list of transitions for an upper level
        self.filtered_levels_df = pd.DataFrame()        # Levels with lifetimes > 0
        self.master_line_data_df = pd.DataFrame()       # The main aggregated data table shown in the GUI
        self.highlight_df = pd.DataFrame()              # NEW: Stores boolean flags for outlier highlighting
        self.result_df = pd.DataFrame()                 # The final calculated branching fractions
        self.current_upper_level_key = ""               # Tracks the currently selected level key

        # Mapping of GUI labels to HDF5 group names for data sources.
        self.DATA_SOURCE_COLUMNS = {"Cal. Linelists": "Calibrated_Linelists", "Raw Spectrum": "Raw_Data"}

        # --- UI Setup ---
        self._create_menu_bar()
        self._create_main_layout()
        self._populate_initial_comboboxes()
        self._populate_data_source_table()
        self._clear_plot()
        
        # Set initial sizes of the splitters
        self.main_splitter.setSizes([350, 1050])
        self.side_panel_splitter.setSizes([self.height() // 2, self.height() // 2])

    def _create_menu_bar(self):
        """Creates the main menu bar (File, Debug, Help)."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        open_action = QAction("Open Saved Analysis...", self) # NEW
        open_action.triggered.connect(self._on_open_analysis_triggered)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        debug_menu = menubar.addMenu("&Debug"); run_diagnostics_action = QAction("Run Diagnostics...", self); run_diagnostics_action.triggered.connect(self._run_debug_diagnostics); debug_menu.addAction(run_diagnostics_action)
        help_menu = menubar.addMenu("&Help"); help_action = QAction("About", self); help_action.triggered.connect(lambda: QMessageBox.information(self, "About", "SAAS")); help_menu.addAction(help_action)

    def _create_main_layout(self):
        """Creates the main horizontal splitter that divides the controls from the data view."""
        self.main_splitter = QSplitter(Qt.Horizontal)
        side_panel_widget = self._create_side_panel()
        central_content_widget = self._create_central_content_widget()
        self.main_splitter.addWidget(side_panel_widget)
        self.main_splitter.addWidget(central_content_widget)
        self.setCentralWidget(self.main_splitter)

    def _create_side_panel(self):
        """Creates the left-hand panel containing all user controls, split vertically."""
        self.side_panel_splitter = QSplitter(Qt.Vertical)
        
        # --- Top Section: Level Selection ---
        level_selector_container = QWidget(); level_selector_layout = QVBoxLayout(level_selector_container)
        self.level_file_combo = QComboBox(); self.level_file_combo.addItem("Select Energy Level File..."); self.level_file_combo.currentIndexChanged.connect(self._on_level_file_selected)
        level_selector_layout.addWidget(QLabel("Master Energy Level File:")); level_selector_layout.addWidget(self.level_file_combo)

        self.level_table = QTableView()
        self.level_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.level_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.level_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.level_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.level_table.clicked.connect(self._on_level_selected_in_table)
        level_selector_layout.addWidget(QLabel("Available Upper Levels:"))
        level_selector_layout.addWidget(self.level_table)

        
        # --- Bottom Section: Data Sources and Actions ---
        data_source_container = QWidget(); data_source_layout = QVBoxLayout(data_source_container)
        self.prev_id_combo = QComboBox(); self.prev_id_combo.addItem("Select Previous IDs File..."); self.prev_id_combo.currentIndexChanged.connect(self._on_prev_id_file_selected)
        data_source_layout.addWidget(QLabel("Master Previous IDs File:")); data_source_layout.addWidget(self.prev_id_combo)

# Create a mini-toolbar for the Data Source section
        ds_header_layout = QHBoxLayout()
        ds_label = QLabel("Data Sources:")
        ds_label.setStyleSheet("font-weight: bold;")
        ds_header_layout.addWidget(ds_label)
        
        self.edit_bands_btn = QPushButton("Edit Band Limits")
        self.edit_bands_btn.setToolTip("Click to configure the wavenumber range for any spectrum")
        self.edit_bands_btn.setMaximumWidth(120)
        self.edit_bands_btn.clicked.connect(self._on_edit_bands_btn_clicked)
        ds_header_layout.addWidget(self.edit_bands_btn)
        
        data_source_layout.addLayout(ds_header_layout)
        
        self.data_source_table = QTableWidget()
        self.data_source_table.itemChanged.connect(self._on_data_source_table_item_changed)

        self.data_source_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_source_table.customContextMenuRequested.connect(self._show_data_source_context_menu)
        data_source_layout.addWidget(self.data_source_table)
 
        self.analysis_controls_group = QWidget()
        analysis_controls_layout = QVBoxLayout(self.analysis_controls_group)
        self.separate_plots_checkbox = QCheckBox("Plot Spectra in Separate Windows")
        analysis_controls_layout.addWidget(self.separate_plots_checkbox)
        self.tolerance_edit = QLineEdit("0.1")
        self.tolerance_edit.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        analysis_controls_layout.addWidget(QLabel("Wavenumber Matching Tolerance (cm⁻¹):"))
        analysis_controls_layout.addWidget(self.tolerance_edit)
        
        data_source_layout.addWidget(self.analysis_controls_group)
        
        self.side_panel_splitter.addWidget(level_selector_container)
        self.side_panel_splitter.addWidget(data_source_container)
        return self.side_panel_splitter

    def _create_central_content_widget(self):
        """Creates the right-hand panel containing the main data table and the plot view."""
        self.central_splitter = QSplitter(Qt.Vertical)
        
        # --- NEW: Create a container for the buttons and the table ---
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a horizontal row for the action buttons
        button_layout = QHBoxLayout()
        self.run_analysis_btn = QPushButton("Calculate Branching Fractions")
        self.run_analysis_btn.clicked.connect(self._calculate_clicked)
        
        self.save_results_btn = QPushButton("Save Results to HDF5")
        self.save_results_btn.clicked.connect(self._save_results_clicked)
        self.save_results_btn.setEnabled(False)
        
        self.copy_table_btn = QPushButton("Copy Table to Clipboard")
        self.copy_table_btn.clicked.connect(self._copy_table_to_clipboard)
        
        # Add the buttons to the horizontal layout
        button_layout.addWidget(self.run_analysis_btn)
        button_layout.addWidget(self.save_results_btn)
        button_layout.addWidget(self.copy_table_btn)
        button_layout.addStretch() # Pushes the buttons to the left
        
        top_layout.addLayout(button_layout)
        
        # The main table for displaying aggregated line data
        self.line_data_table = QTableView()
        self.line_data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.line_data_table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Apply the custom multi-level header
        self.custom_header = MultiLevelHeaderView(Qt.Horizontal, self.line_data_table)
        self.custom_header.setSectionResizeMode(QHeaderView.Interactive)
        
        self.line_data_table.setHorizontalHeader(self.custom_header)
        self.line_data_table.setAlternatingRowColors(True)
        self.line_data_table.clicked.connect(self._on_line_selected)

        self.line_data_table.setEditTriggers(QAbstractItemView.NoEditTriggers) # Table is read-only
        self.line_data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.line_data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.line_data_table.customContextMenuRequested.connect(self._show_line_table_context_menu)
        
        # Add the table to the layout, and the whole container to the splitter
        top_layout.addWidget(self.line_data_table)
        self.central_splitter.addWidget(top_container)
        
        # The Matplotlib widget for plotting spectra
        main_plot_widget = QWidget(); plot_layout = QVBoxLayout(main_plot_widget); plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 4), dpi=100); self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self); self.ax = self.figure.add_subplot(111)
        plot_layout.addWidget(self.toolbar); plot_layout.addWidget(self.canvas)
        self.central_splitter.addWidget(main_plot_widget)
        
        return self.central_splitter    
        
    def _show_line_table_context_menu(self, position):
        """Creates and shows a context menu when the line data table is right-clicked."""
        index = self.line_data_table.indexAt(position)
        if not index.isValid(): return
        
        menu = QMenu()
        normalize_action = menu.addAction("Set as Intensity Reference (Normalize to 1000)")
        
        # --- NEW: Show Details Action ---
        menu.addSeparator()
        details_action = menu.addAction("Show All Line Parameters")
        
        # Get only actual spectrum names (ignoring "Mean" columns)
        spectrum_names = sorted(list(set([col.split('\n')[0] for col in self.master_line_data_df.columns if '\nSNR' in col])))
        
        # --- Transfer Calibration Submenu ---
        menu.addSeparator()
        transfer_menu = menu.addMenu("Transfer Calibration To...")
        transfer_actions = {}
        for spec in spectrum_names:
            action = transfer_menu.addAction(f"Spectrum: {spec}")
            transfer_actions[action] = spec

        # --- Toggle Exclusion Submenu ---
        menu.addSeparator()
        exclude_menu = menu.addMenu("Toggle Line Exclusion In...")
        exclude_actions = {}
        for spec in spectrum_names:
            is_excluded = False
            excluded_col = f"{spec}\nExcluded"
            if excluded_col in self.master_line_data_df.columns:
                is_excluded = bool(self.master_line_data_df.iloc[index.row()].get(excluded_col, False))
                
            status_text = " (Currently Excluded)" if is_excluded else ""
            action = exclude_menu.addAction(f"Spectrum: {spec}{status_text}")
            exclude_actions[action] = spec

        action = menu.exec_(self.line_data_table.viewport().mapToGlobal(position))
        
        if action == normalize_action: 
            self._normalize_intensities(index.row())
        elif action == details_action:
            self._show_line_details(index.row())
        elif action in transfer_actions:
            self._transfer_calibration(index.row(), transfer_actions[action])
        elif action in exclude_actions:
            self._toggle_exclusion(index.row(), exclude_actions[action])

    def _normalize_intensities(self, reference_line_row: int):
        """
        Handles the normalization action from the context menu.

        This function calls the analysis module to perform the normalization,
        re-calculates the weighted averages with the new intensities, and then
        updates the table view with the rescaled data.
        """
        if self.master_line_data_df.empty: QMessageBox.warning(self, "Normalization Error", "No data loaded to normalize."); return
        try:
            # Call the analysis function to rescale the intensity values.
            normalized_df = self.analysis_module.normalize_intensities_by_reference_line(self.master_line_data_df, reference_line_row)
            self.master_line_data_df = normalized_df
            
            # After normalizing, the mean intensities must be recalculated.
            self.master_line_data_df = self.analysis_module.add_weighted_averages(self.master_line_data_df, self.h5_filepath)

            # Calculate outliers based on the new means ---
            self.highlight_df = self.analysis_module.calculate_outliers(self.master_line_data_df, self.h5_filepath)     
           
            # Update the table view with the new data.
            model = LineDataTableModel(self.master_line_data_df, self.highlight_df)
            self.line_data_table.setModel(model)
            self._format_table_columns() 
            
            # Restore the user's selection and update the plot for a smooth workflow.
            new_index_to_select = model.index(reference_line_row, 0)
            if new_index_to_select.isValid():
                self.line_data_table.setCurrentIndex(new_index_to_select)
                selection_model = self.line_data_table.selectionModel()
                selection_model.select(new_index_to_select, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                self._on_line_selected(new_index_to_select)

            ref_level_key = self.master_line_data_df.iloc[reference_line_row].get('lower_level_key', 'Unknown')
            QMessageBox.information(self, "Success", f"Intensities have been normalized using '{ref_level_key}' as the reference.")
        except Exception as e: QMessageBox.critical(self, "Normalization Error", f"An error occurred during normalization:\n{e}")

    def _transfer_calibration(self, transfer_line_row: int, target_spectrum: str):
        """
        Handles the transfer calibration action from the context menu.
        """
        if self.master_line_data_df.empty: 
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        try:
            # Call the analysis function to re-normalize the target spectrum
            updated_df = self.analysis_module.transfer_calibration(self.master_line_data_df, transfer_line_row, target_spectrum, self.h5_filepath)
            self.master_line_data_df = updated_df
            self.master_line_data_df = self.analysis_module.add_weighted_averages(self.master_line_data_df, self.h5_filepath)

            # Recalculate highlights as the mean has changed
            self.highlight_df = self.analysis_module.calculate_outliers(self.master_line_data_df, self.h5_filepath)
            
            # Update the table view
            model = LineDataTableModel(self.master_line_data_df, self.highlight_df)
            self.line_data_table.setModel(model)
            self._format_table_columns()
            
            # Restore the user's selection
            new_index_to_select = model.index(transfer_line_row, 0)
            if new_index_to_select.isValid():
                self.line_data_table.setCurrentIndex(new_index_to_select)
                selection_model = self.line_data_table.selectionModel()
                selection_model.select(new_index_to_select, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                self._on_line_selected(new_index_to_select)

            ref_level_key = self.master_line_data_df.iloc[transfer_line_row].get('lower_level_key', 'Unknown')
            QMessageBox.information(self, "Success", f"Calibration transferred to '{target_spectrum}' using '{ref_level_key}' as the transfer line.")
            
        except Exception as e: 
            QMessageBox.critical(self, "Transfer Error", f"An error occurred during calibration transfer:\n{e}")

    def _toggle_exclusion(self, row_index: int, target_spectrum: str):
        """Toggles the excluded state of a line in a specific spectrum."""
        excluded_col = f"{target_spectrum}\nExcluded"
        
        # Create the hidden tracking column if it doesn't exist yet
        if excluded_col not in self.master_line_data_df.columns:
            self.master_line_data_df[excluded_col] = False
            
        # Flip the boolean state
        current_status = bool(self.master_line_data_df.iloc[row_index].get(excluded_col, False))
        self.master_line_data_df.at[self.master_line_data_df.index[row_index], excluded_col] = not current_status
        
        # Recalculate means (this will now force the excluded weight to 0)
        self.master_line_data_df = self.analysis_module.add_weighted_averages(self.master_line_data_df, self.h5_filepath)

        # Recalculate highlights as the mean has changed
        self.highlight_df = self.analysis_module.calculate_outliers(self.master_line_data_df, self.h5_filepath)
        
        # Refresh the table and plot
        model = LineDataTableModel(self.master_line_data_df, self.highlight_df)
        self.line_data_table.setModel(model)
        self._format_table_columns()
        
        new_index_to_select = model.index(row_index, 0)
        if new_index_to_select.isValid():
            self.line_data_table.setCurrentIndex(new_index_to_select)
            self.line_data_table.selectionModel().select(new_index_to_select, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            self._on_line_selected(new_index_to_select)

    def _show_line_details(self, row_index: int):
        """Fetches the raw parameters from the HDF5 file and displays them in a popup."""
        if self.master_line_data_df.empty: return
        
        line_data = self.master_line_data_df.iloc[row_index]
        target_wavenumber = pd.to_numeric(line_data.get('wavenumber'), errors='coerce')
        
        if pd.isna(target_wavenumber):
            QMessageBox.warning(self, "Error", "Invalid wavenumber for selected line.")
            return
            
        try:
            tolerance = float(self.tolerance_edit.text())
        except ValueError:
            tolerance = 0.1
            
        all_checked_paths = self._get_checked_data_paths()
        linelist_paths =[p for p in all_checked_paths if ('Identified_Lines' in p or 'Calibrated_Linelists' in p)]
        
        details =[]
        for path in linelist_paths:
            try:
                spectrum_name = path.split('/')[2]
                df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
                if 'wavenumber' in df.columns:
                    df['wavenumber_num'] = pd.to_numeric(df['wavenumber'], errors='coerce')
                    diffs = np.abs(df['wavenumber_num'] - target_wavenumber)
                    
                    if not diffs.empty and np.min(diffs) <= tolerance:
                        best_idx = np.argmin(diffs)
                        row_dict = df.iloc[best_idx].copy().to_dict()
                        
                        # Cleanup the internal tracking columns
                        row_dict.pop('wavenumber_num', None)
                        row_dict.pop('index', None)
                        
                        # Put spectrum name first
                        final_dict = {'Spectrum': spectrum_name}
                        final_dict.update(row_dict)
                        details.append(final_dict)
            except Exception as e:
                print(f"Could not load details from {path}: {e}")
                
        if details:
            details_df = pd.DataFrame(details)
            # Reorder columns to ensure Spectrum and wavenumber are the first two columns
            cols = ['Spectrum', 'wavenumber'] +[c for c in details_df.columns if c not in ['Spectrum', 'wavenumber']]
            details_df = details_df[cols]
            
            # Show modeless dialog
            self.details_dialog = LineDetailsDialog(details_df, target_wavenumber, self)
            self.details_dialog.setModal(False)
            self.details_dialog.show()
        else:
            QMessageBox.information(self, "No Details", "No matching lines found in the raw linelists.")

    def _populate_data_source_table(self):
        """Scans the HDF5 file and populates the data source table with available items."""
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Spectra' not in f: 
                    self.data_source_table.clear()
                    return
                
                spectra_names = sorted(list(f['/Spectra'].keys()))
                column_labels = list(self.DATA_SOURCE_COLUMNS.keys())
                
                self.data_source_table.setRowCount(len(spectra_names))
                self.data_source_table.setColumnCount(len(column_labels))
                self.data_source_table.setVerticalHeaderLabels(spectra_names)
                self.data_source_table.setHorizontalHeaderLabels(column_labels)
                
                for r, spectrum_name in enumerate(spectra_names):
                    for c, col_label in enumerate(column_labels):
                        hdf5_group_name = self.DATA_SOURCE_COLUMNS[col_label]
                        base_path = f"/Spectra/{spectrum_name}/{hdf5_group_name}"
                        
                        item = QTableWidgetItem()
                        item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                        item.setBackground(QBrush(QColor('lightGray')))
                        
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
                            
                            if item_text and dset_path in f:
                                # --- FIX: Extract band metadata for the tooltip ---
                                attrs = f[dset_path].attrs
                                # Use wstart as fallback for bandlo, and wend for bandhi
                                b_lo = attrs.get('bandlo', attrs.get('wstart', 0.0))
                                b_hi = attrs.get('bandhi', attrs.get('wend', b_lo + 30000.0))
                                
                                band_tooltip = f"Path: {dset_path}\nBands: {float(b_lo):.1f} to {float(b_hi):.1f} cm⁻¹"
                                
                                item.setText("")
                                item.setToolTip(band_tooltip)
                                item.setData(Qt.UserRole, dset_path)
                                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                                item.setBackground(QBrush(QColor('white')))
                                item.setCheckState(Qt.Unchecked)
                                
                        self.data_source_table.setItem(r, c, item)
            
            self.data_source_table.resizeColumnsToContents()
            self.data_source_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        except Exception as e: 
            QMessageBox.critical(self, "HDF5 Scan Error", f"Failed to populate data source table: {e}")

    def _get_checked_data_paths(self):
        """Retrieves the HDF5 paths of all data sources checked by the user."""
        checked_paths = [];
        for r in range(self.data_source_table.rowCount()):
            for c in range(self.data_source_table.columnCount()):
                item = self.data_source_table.item(r, c)
                if item and item.checkState() == Qt.Checked:
                    path = item.data(Qt.UserRole)
                    if path: checked_paths.append(path)
        return checked_paths
        
    def _populate_initial_comboboxes(self):
        """Populates the dropdown menus with available Level and ID files from the HDF5 file."""
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Levels' in f: self.level_file_combo.addItems([name for name in f['/Levels'].keys() if isinstance(f['/Levels'][name], h5py.Group)])
                if '/Previous_Identifications' in f: self.prev_id_combo.addItems([name for name in f['/Previous_Identifications'].keys() if isinstance(f['/Previous_Identifications'][name], h5py.Group)])
        except Exception as e: QMessageBox.critical(self, "HDF5 Error", f"Failed to read HDF5 structure: {e}")

    def _on_level_file_selected(self):
        """Handles the event when the user selects a master energy level file."""
        selected_file = self.level_file_combo.currentText()
        if selected_file == "Select Energy Level File...": self.level_table.setModel(None); self._clear_level_details(); self.current_energy_levels_df, self.filtered_levels_df = pd.DataFrame(), pd.DataFrame(); return
        path = f"/Levels/{selected_file}/table"
        try:
            self.current_energy_levels_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
            # Sanitize the key column for consistent matching.
            if not self.current_energy_levels_df.empty and 'key' in self.current_energy_levels_df.columns:
                self.current_energy_levels_df['key'] = self.current_energy_levels_df['key'].astype(str).str.replace('*', '', regex=False).str.strip()
            # Filter for levels with a measured lifetime, which are the only valid upper levels.
            if not self.current_energy_levels_df.empty and 'lifetime' in self.current_energy_levels_df.columns:
                self.filtered_levels_df = self.current_energy_levels_df[self.current_energy_levels_df['lifetime'] > 0].copy()
                # Dynamically fetch the columns so it doesn't crash if an older file is missing uncertainty
                cols_to_show = [col for col in['key', 'energy', 'j_value', 'parity', 'lifetime', 'lifetime_unc_frac'] if col in self.filtered_levels_df.columns]
                self.level_table.setModel(PandasTableModel(self.filtered_levels_df[cols_to_show]))
                self.level_table.resizeColumnsToContents()

            else: self.level_table.setModel(None); QMessageBox.warning(self, "Data Error", f"Table at {path} is empty or missing required columns.")
        except Exception as e:
            self.level_table.setModel(None); self.current_energy_levels_df, self.filtered_levels_df = pd.DataFrame(), pd.DataFrame()
            QMessageBox.critical(self, "HDF5 Read Error", f"Could not read energy levels from {path}:\n{e}")
        finally: self._clear_level_details()

    def _on_prev_id_file_selected(self):
        """Handles the event when the user selects a Previous Identifications file."""
        selected_file = self.prev_id_combo.currentText()
        if selected_file == "Select Previous IDs File...": self.current_previous_ids_df = pd.DataFrame(); self.line_data_table.setModel(None); self._clear_plot(); return
        path = f"/Previous_Identifications/{selected_file}/table"
        try:
            self.current_previous_ids_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
            # Create a sanitized 'normalized_key' for consistent matching with the levels file.
            if not self.current_previous_ids_df.empty and 'upper_level_key' in self.current_previous_ids_df.columns:
                self.current_previous_ids_df['normalized_key'] = self.current_previous_ids_df['upper_level_key'].astype(str).str.replace('*', '', regex=False).str.strip()
            # If a level is already selected, refresh the main table.
            if self.level_table.selectionModel() and self.level_table.selectionModel().hasSelection(): self._on_level_selected_in_table()
        except Exception as e:
            self.current_previous_ids_df = pd.DataFrame(); QMessageBox.critical(self, "HDF5 Read Error", f"Could not read Previous IDs from {path}:\n{e}")
            self.line_data_table.setModel(None); self._clear_plot()
            
    def _on_level_selected_in_table(self):
        """Handles the event when a user clicks on an upper level in the level table."""
        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes or self.filtered_levels_df.empty: 
            self._clear_level_details()
            self.line_data_table.setModel(None)
            self._clear_plot()
            return
            
        row = selected_indexes[0].row()
        selected_level_data = self.filtered_levels_df.iloc[row]
        
        # Save the key internally for saving/calculations later
        self.current_upper_level_key = str(selected_level_data.get('key', ''))
        
        # This is the primary trigger to build the main data table.
        self._populate_line_data_table(selected_level_data['key'])
        
    def _format_table_columns(self):
        """Helper method to explicitly resize the width of all columns and hide unwanted ones."""
        model = self.line_data_table.model()
        if not model: 
            return
            
        for col in range(model.columnCount()):
            header_text = str(model.headerData(col, Qt.Horizontal, Qt.DisplayRole))
            col_name = str(model.df.columns[col])
            
            # --- NEW: Hide the math-only tracking columns from the user ---
            if "Width" in col_name or "Excluded" in col_name:
                self.line_data_table.setColumnHidden(col, True)
                continue
            else:
                self.line_data_table.setColumnHidden(col, False)
            # --------------------------------------------------------------
                
            if col in [0, 1, 2]:  # wavenumber, lower_level_key, intensity
                self.line_data_table.resizeColumnToContents(col)
            elif "Mean" in header_text: 
                # Catch both 'Mean Intensity' and 'Mean Uncertainty'
                self.line_data_table.resizeColumnToContents(col)
            else:
                # Everything else (the Spectrum Intensity & SNR columns)
                self.line_data_table.setColumnWidth(col, 55)

    def _populate_line_data_table(self, upper_level_key: str):
        """
        The core data aggregation and display function.

        This function is called when a valid upper level is selected. It:
        1. Filters the 'Previous IDs' for lines originating from the selected upper level.
        2. Gathers the paths of all user-checked experimental linelists.
        3. Calls `analysis.aggregate_observed_data_for_display` to merge all data.
        4. Calls `analysis.add_weighted_averages` to calculate mean values.
        5. Displays the resulting final DataFrame in the main table view.
        """
        if self.current_previous_ids_df.empty: self.line_data_table.setModel(None); self._clear_plot(); return
        if 'normalized_key' not in self.current_previous_ids_df.columns: return
        
        lines_from_level = self.current_previous_ids_df[self.current_previous_ids_df['normalized_key'] == upper_level_key]
        if lines_from_level.empty: self.line_data_table.setModel(None); self._clear_plot(); return
        
        all_checked_paths = self._get_checked_data_paths()
        linelist_paths_to_merge = [p for p in all_checked_paths if ('Identified_Lines' in p or 'Calibrated_Linelists' in p)]
        
        try:
            df_to_pass = lines_from_level.drop(columns=['normalized_key'], errors='ignore')
            # 1. Aggregate experimental data.
            self.master_line_data_df = self.analysis_module.aggregate_observed_data_for_display(h5_filepath=self.h5_filepath, previous_ids_df=df_to_pass, linelist_paths=linelist_paths_to_merge, tolerance=float(self.tolerance_edit.text()))
            
            # 2. Calculate weighted averages.
            if not self.master_line_data_df.empty: 
                self.master_line_data_df = self.analysis_module.add_weighted_averages(self.master_line_data_df, self.h5_filepath)
            if self.master_line_data_df.empty: self.line_data_table.setModel(None); self._clear_plot(); return

            # Clear any previous highlighting ---
            self.highlight_df = pd.DataFrame()
            
            # 3. Display the final table.
            model = LineDataTableModel(self.master_line_data_df, self.highlight_df)
            self.line_data_table.setModel(model)
            self._format_table_columns() 
            self._clear_plot()
            current_height = self.central_splitter.height()
            self.central_splitter.setSizes([current_height // 2, current_height // 2])

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error in _populate_line_data_table: {e}")
            self.line_data_table.setModel(None); self._clear_plot()

    def _clear_level_details(self):
        """Clears the internal state when no level is selected."""
        self.current_upper_level_key = ""
        
    def _on_data_source_table_item_changed(self, item):
        """Handles the event when a user checks/unchecks a data source, triggering a table refresh."""
        
        # Block signals temporarily to prevent infinite recursion when we programmatically check another box
        self.data_source_table.blockSignals(True)
        try:
            col = item.column()
            row = item.row()
            header_label = self.data_source_table.horizontalHeaderItem(col).text()
            
            # Auto-check the "Raw Spectrum" box if "Cal. Linelists" is checked
            if header_label == "Cal. Linelists" and item.checkState() == Qt.Checked:
                raw_col = -1
                # Dynamically find the column index for "Raw Spectrum"
                for c in range(self.data_source_table.columnCount()):
                    if self.data_source_table.horizontalHeaderItem(c).text() == "Raw Spectrum":
                        raw_col = c
                        break
                        
                if raw_col != -1:
                    raw_item = self.data_source_table.item(row, raw_col)
                    # Ensure the item exists and is user-checkable
                    if raw_item and (raw_item.flags() & Qt.ItemIsUserCheckable):
                        raw_item.setCheckState(Qt.Checked)
        finally:
            # Always restore signals no matter what
            self.data_source_table.blockSignals(False)

        # Trigger the main table and plot update as usual
        if self.level_table.selectionModel() and self.level_table.selectionModel().hasSelection():
            selected_indexes = self.level_table.selectionModel().selectedRows()
            selected_row = selected_indexes[0].row()
            selected_level_data = self.filtered_levels_df.iloc[selected_row]
            self._populate_line_data_table(selected_level_data['key'])
        else: 
            self.line_data_table.setModel(None)

    def _show_data_source_context_menu(self, position):
        """Creates a context menu for the data source table to edit metadata."""
        index = self.data_source_table.indexAt(position)
        if not index.isValid(): return
        
        row = index.row()
        # The vertical header stores the spectrum names
        spectrum_name = self.data_source_table.verticalHeaderItem(row).text()
        
        menu = QMenu()
        edit_bands_action = menu.addAction(f"Edit Band Limits (bandlo, bandhi) for {spectrum_name}")
        
        action = menu.exec_(self.data_source_table.viewport().mapToGlobal(position))
        
        if action == edit_bands_action:
            self._edit_spectrum_bands(spectrum_name)

    def _edit_spectrum_bands(self, spectrum_name: str):
        """Pops up a dialog to let the user override bandlo and bandhi."""
        spec_path = f"/Spectra/{spectrum_name}/Raw_Data/spectrum"
        
        # 1. Fetch current limits from HDF5
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if spec_path in f:
                    attrs = f[spec_path].attrs
                    current_bandlo = attrs.get('bandlo', attrs.get('wstart', 0.0))
                    current_bandhi = attrs.get('bandhi', attrs.get('wend', current_bandlo + 30000.0))
                else:
                    QMessageBox.warning(self, "Error", f"Raw data not found for {spectrum_name}")
                    return
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not read metadata: {e}")
            return
            
        # 2. Build the Dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Band Limits: {spectrum_name}")
        layout = QFormLayout(dialog)
        
        lo_edit = QLineEdit(str(current_bandlo))
        hi_edit = QLineEdit(str(current_bandhi))
        
        layout.addRow("bandlo (cm⁻¹):", lo_edit)
        layout.addRow("bandhi (cm⁻¹):", hi_edit)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        # 3. Save and Refresh if Accepted
        if dialog.exec_() == QDialog.Accepted:
            try:
                new_lo = float(lo_edit.text())
                new_hi = float(hi_edit.text())
                
                # Save straight to the HDF5 attributes
                with h5py.File(self.h5_filepath, 'a') as f:
                    f[spec_path].attrs['bandlo'] = new_lo
                    f[spec_path].attrs['bandhi'] = new_hi
                    
                QMessageBox.information(self, "Success", f"Band limits for {spectrum_name} updated successfully.")
                
                # Force a recalculation with the new values by refreshing the table
                if self.level_table.selectionModel() and self.level_table.selectionModel().hasSelection():
                    selected_indexes = self.level_table.selectionModel().selectedRows()
                    row = selected_indexes[0].row()
                    selected_level_data = self.filtered_levels_df.iloc[row]
                    self._populate_line_data_table(selected_level_data['key'])
                    
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save metadata: {e}")
            
    def _on_line_selected(self, index: QModelIndex):
        """Handles the event when a user clicks a line in the main table, triggering a plot update."""
        if not index.isValid() or self.master_line_data_df.empty: self._clear_plot(); return
        row = index.row(); line_data = self.master_line_data_df.iloc[row]
        wavenumber = line_data.get('wavenumber')
        if wavenumber is not None:
            try:
                # Wavenumber must be converted to float for plotting.
                wavenumber_float = float(wavenumber)
                # CHANGE: Pass line_data so the plotter knows if it's excluded
                self._update_plot(wavenumber_float, self._get_checked_data_paths(), line_data)
            except (ValueError, TypeError): self._clear_plot()
        else: self._clear_plot()
            
    def _update_plot(self, target_wavenumber: float, all_checked_paths: list, line_data=None):
        """
        Draws the spectral data for a selected line in the Matplotlib canvas.

        This function supports two modes based on the checkbox state:
        1.  Overlay Mode: All selected spectra are plotted on a single axis.
        2.  Separate Mode: Each spectrum is plotted on its own subplot with an independent Y-axis.
        """
        self.figure.clear()
        plot_in_separate_windows = self.separate_plots_checkbox.isChecked()
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        spectrum_data_paths = [p for p in all_checked_paths if 'Raw_Data' in p]
        
        # Determine plot range by finding the widest line width among matching lines.
        linelist_paths =[p for p in all_checked_paths if 'Calibrated_Linelists' in p or 'Identified_Lines' in p]
        max_fwhm = 0.0
        tolerance = float(self.tolerance_edit.text())
        for path in linelist_paths:
            try:
                linelist_df = h5_manager.read_hdf_table_robustly(self.h5_filepath, path)
                if 'wavenumber' not in linelist_df.columns or 'width' not in linelist_df.columns: continue
                linelist_df['wavenumber'] = pd.to_numeric(linelist_df['wavenumber'], errors='coerce')
                differences = np.abs(linelist_df['wavenumber'] - target_wavenumber)
                best_match_index = differences.idxmin()
                if differences[best_match_index] <= tolerance: max_fwhm = max(max_fwhm, linelist_df.loc[best_match_index, 'width'])
            except Exception as e: print(f"Could not read FWHM for line {target_wavenumber} in {path}: {e}")
            
        if max_fwhm > 0: max_fwhm /= 1000.0 # Convert from mK to cm-1
        plot_range = (5.0 * max_fwhm) if max_fwhm > 0 else 5.0
        
        spectrum_data_loaded = False
        num_plots = len(spectrum_data_paths)
        axes =[]
        
        # --- Create subplot layout based on user choice ---
        if plot_in_separate_windows and num_plots > 0:
            # Independent scaling (no sharey)
            for i in range(num_plots):
                axes.append(self.figure.add_subplot(1, num_plots, i + 1))
        else:
            axes = [self.figure.add_subplot(1, 1, 1)]

        # --- Loop through and plot each selected spectrum ---
        for i, spec_path in enumerate(spectrum_data_paths):
            try:
                plot_axis = axes[i] if plot_in_separate_windows and num_plots > 0 else axes[0]
                spectrum_name = spec_path.split('/')[2]
                
                # --- Check if this spectrum is excluded for this line ---
                is_excluded = False
                if line_data is not None:
                    excluded_col = f"{spectrum_name}\nExcluded"
                    if excluded_col in line_data.index and line_data[excluded_col] == True:
                        is_excluded = True
                        
                # --- Set plot aesthetics based on exclusion status ---
                if is_excluded:
                    line_color = 'lightgray'
                    alpha = 0.5
                    plot_label = f"{spectrum_name} (Excluded)"
                else:
                    line_color = color_cycle[i % len(color_cycle)]
                    alpha = 0.7
                    plot_label = spectrum_name

                # Load and plot the actual data
                with h5py.File(self.h5_filepath, 'r') as f:
                    h5_dataset = f[spec_path]; attrs = h5_dataset.attrs
                    wavcorr, wstart, delw, rdsclfct = attrs.get('wavcorr', 0.0), attrs.get('wstart', 0.0), attrs.get('delw', 1.0), attrs.get('rdsclfct', 1.0)
                    data = h5_dataset[:] 
                    y = data * rdsclfct
                    indices = np.arange(len(data))
                    x = wstart + indices * delw
                    x_corrected = x * (1.0 + wavcorr)
                    mask = (x_corrected >= target_wavenumber - plot_range) & (x_corrected <= target_wavenumber + plot_range)
                    
                    if np.any(mask):
                        plot_axis.plot(x_corrected[mask], y[mask], color=line_color, alpha=alpha, label=plot_label)
                        # Make the vertical red dashed line semi-transparent if excluded
                        plot_axis.axvline(target_wavenumber, color='red', linestyle='--', alpha=0.3 if is_excluded else 1.0)
                        plot_axis.grid(True)
                        if plot_in_separate_windows:
                            plot_axis.set_title(plot_label, fontsize=10)
                        spectrum_data_loaded = True
            except Exception as e: print(f"Error loading spectrum data for plot from {spec_path}: {e}")
        
        # --- Add labels and titles appropriate for the plot mode ---
        if spectrum_data_loaded:
            if plot_in_separate_windows and num_plots > 0:
                self.figure.suptitle(f"Spectra around {target_wavenumber:.3f} cm⁻¹")
                self.figure.supxlabel(r'$\sigma$ (cm$^{-1}$)')
                axes[0].set_ylabel('Intensity')
            else:
                main_ax = axes[0]
                main_ax.set_title(f"Spectra around {target_wavenumber:.3f} cm⁻¹")
                main_ax.set_xlabel(r'$\sigma$ (cm$^{-1}$)')
                main_ax.set_ylabel('Intensity')
                main_ax.legend()
            self.figure.tight_layout()
        else:
            ax = self.figure.add_subplot(1,1,1)
            ax.text(0.5, 0.5, "No Spectrum Data Selected or Loaded", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='darkred')
        
        self.canvas.draw()       

    def _clear_plot(self):
        """Clears the plot canvas and shows a placeholder message."""
        if self.figure.get_axes():
            ax = self.figure.get_axes()[0]; ax.clear()
            ax.text(0.5, 0.5, "Select an upper level and a line to view spectrum", ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw()

    def _calculate_clicked(self):   
        """Handles the 'Calculate' button click, runs the analysis, and shows the results dialog."""
        if self.master_line_data_df.empty: QMessageBox.warning(self, "Calculation Error", "No lines loaded."); return
        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes: QMessageBox.warning(self, "Calculation Error", "Please select an upper level."); return
        row = selected_indexes[0].row()
        selected_level_data = self.filtered_levels_df.iloc[row]
        upper_level_key = selected_level_data['key']
        
        # --- NEW: Fetch calculations table to compute unobserved residuals ---
        calcs_df = pd.DataFrame()
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if '/Calculations' in f:
                    calc_groups = list(f['/Calculations'].keys())
                    if calc_groups:
                        calc_path = f"/Calculations/{calc_groups[0]}/table"
                        calcs_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, calc_path)
        except Exception as e:
            print(f"Warning: Could not load calculations table for residuals: {e}")
            
        try:
            tolerance = float(self.tolerance_edit.text())
        except ValueError:
            tolerance = 0.1
            
        try:
            self.result_df = self.analysis_module.calculate_branching_fractions(
                self.master_line_data_df, 
                upper_level_key=upper_level_key, 
                energy_levels_df=self.current_energy_levels_df,
                calculations_df=calcs_df,
                wavenumber_tolerance=tolerance
            )
            
            if not self.result_df.empty:
                self.save_results_btn.setEnabled(True)
                
                # Show modeless dialog
                self.results_dialog = ResultsDisplayDialog(self.result_df, self)
                self.results_dialog.setModal(False)
                self.results_dialog.show()
                
            else:
                QMessageBox.warning(self, "Calculation Error", "Calculation returned no results."); self.save_results_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"An error occurred: {e}"); self.result_df = pd.DataFrame(); self.save_results_btn.setEnabled(False)
            
    def _save_results_clicked(self):
        """
        Handles the 'Save Results' button click, creating a structured group
        in the HDF5 file for reproducibility.
        """
        if self.result_df.empty or self.master_line_data_df.empty:
            QMessageBox.warning(self, "Save Error", "No results to save."); return

        default_name = f"BF_analysis_{self.current_upper_level_key}_{date.today().strftime('%Y%m%d')}"

        analysis_name, ok = QInputDialog.getText(self, "Save Analysis", "Enter a unique name for this analysis:", text=default_name)
        
        if ok and analysis_name:
            base_group = "/Branching_Fraction_Analyses"
            analysis_group_path = f"{base_group}/{analysis_name}"
            
            try:
                # Check for and handle overwriting existing data.
                with h5py.File(self.h5_filepath, 'a') as f:
                    if analysis_group_path in f:
                        reply = QMessageBox.question(self, "Overwrite Confirmation", f"An analysis named '{analysis_name}' already exists. Do you want to overwrite it?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                        if reply == QMessageBox.No:
                            QMessageBox.information(self, "Save Cancelled", "Save operation cancelled."); return
                        else:
                            del f[analysis_group_path]

                # Create groups and save both the input data and the final results for reproducibility.
                self.h5_manager.create_group_if_not_exists(self.h5_filepath, base_group)
                self.h5_manager.create_group_if_not_exists(self.h5_filepath, analysis_group_path)
                self.h5_manager.add_pandas_table(self.h5_filepath, analysis_group_path, "calculation_input_data", self.master_line_data_df)
                self.h5_manager.add_pandas_table(self.h5_filepath, analysis_group_path, "branching_fraction_results", self.result_df)
                
                # Attach all relevant parameters as metadata to the analysis group.
                metadata_to_save = {'analysis_date': date.today().isoformat(), 'source_level_file': self.level_file_combo.currentText(),'source_previous_ids_file': self.prev_id_combo.currentText(), 'source_linelists': str(self._get_checked_data_paths()), 'wavenumber_tolerance': float(self.tolerance_edit.text()), 'upper_level_key':  self.current_upper_level_key}
                self.h5_manager.attach_metadata_to_group(self.h5_filepath, analysis_group_path, metadata_to_save)
                
                QMessageBox.information(self, "Save Complete", f"Analysis saved to HDF5 at:\n{analysis_group_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "HDF5 Save Error", f"Failed to save results:\n{e}")
        else:
            QMessageBox.information(self, "Save Cancelled", "Save operation cancelled.")

    def _copy_table_to_clipboard(self):
        """
        Exports the main interactive data table to the system clipboard, 
        formatting it specifically for easy pasting into Excel or Google Sheets.
        """
        if self.master_line_data_df.empty:
            QMessageBox.warning(self, "Copy Error", "No data to copy. Please select an upper level first.")
            return
            
        try:
            # Create a copy so we don't accidentally modify the actual working dataframe
            export_df = self.master_line_data_df.copy()
            
            # Replace the '\n' in the column headers with a space. 
            # (Otherwise, Excel will split the headers across multiple rows!)
            export_df.columns =[str(col).replace('\n', ' ') for col in export_df.columns]
            
            # Convert the dataframe to a tab-separated string
            text = export_df.to_csv(sep='\t', index=False)
            
            # Send it to the system clipboard
            QApplication.clipboard().setText(text)
            
            QMessageBox.information(self, "Success", "Main table copied to clipboard!\nYou can now paste it directly into Excel or Sheets.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy to clipboard:\n{e}")
            
    def _run_debug_diagnostics(self):
        """Runs a series of checks on the current data state and shows a report."""
        report = []
        report.append("--- 1. Master DataFrames ---")
        if self.current_energy_levels_df.empty: report.append("WARNING: Energy Levels DataFrame is EMPTY.")
        else: report.append(f"OK: Energy Levels DataFrame loaded ({len(self.current_energy_levels_df)} rows).")
        report.append("\n")
        if self.current_previous_ids_df.empty: report.append("WARNING: Previous IDs DataFrame is EMPTY.")
        else:
            report.append(f"OK: Previous IDs DataFrame loaded ({len(self.current_previous_ids_df)} rows).")
            if 'normalized_key' in self.current_previous_ids_df.columns: report.append("OK: 'normalized_key' column was successfully created.")
            else: report.append("ERROR: 'normalized_key' column was NOT created.")
        report.append("\n--- 2. Level Selection & Filtering ---")
        selected_indexes = self.level_table.selectionModel().selectedRows()
        if not selected_indexes: report.append("INFO: No level is currently selected in the table."); self._show_debug_report(report); return
        row = selected_indexes[0].row()
        selected_level_data = self.filtered_levels_df.iloc[row]
        upper_level_key = selected_level_data.get('key')
        if not upper_level_key: report.append("ERROR: A level is selected, but could not get its 'key' value!"); self._show_debug_report(report); return
        report.append(f"OK: A level is selected. The key being used for filtering is: '{upper_level_key}'")
        report.append("\n--- 3. Filtering Previous IDs ---")
        if 'normalized_key' not in self.current_previous_ids_df.columns:
            report.append(f"FATAL ERROR: The Previous IDs DataFrame does NOT have the 'normalized_key' column."); self._show_debug_report(report); return
        lines_from_level = self.current_previous_ids_df[self.current_previous_ids_df['normalized_key'] == upper_level_key]
        if lines_from_level.empty: report.append(f"\nRESULT: CRITICAL FAILURE! Found 0 matching lines for key '{upper_level_key}'.")
        else: report.append(f"\nRESULT: SUCCESS! Found {len(lines_from_level)} matching lines for key '{upper_level_key}'.")
        self._show_debug_report(report)
        
    def _show_debug_report(self, report_lines):
        """Displays the debug report in a simple dialog."""
        dialog = QDialog(self); dialog.setWindowTitle("Debug Diagnostics Report"); dialog.setMinimumSize(700, 500)
        layout = QVBoxLayout(dialog); report_text = QTextEdit(); report_text.setReadOnly(True)
        report_text.setFont(QFont("Monospace", 10)); report_text.setText("\n".join(report_lines))
        layout.addWidget(report_text); button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept); layout.addWidget(button_box)
        dialog.exec_()

    def _on_edit_bands_btn_clicked(self):
        """Logic for the 'Edit Bands' button."""
        # Check if a row is currently selected in the source table
        selected_ranges = self.data_source_table.selectedRanges()
        
        if selected_ranges:
            # Use the spectrum from the first selected row
            row = selected_ranges[0].topRow()
            spectrum_name = self.data_source_table.verticalHeaderItem(row).text()
            self._edit_spectrum_bands(spectrum_name)
        else:
            # If nothing is selected, let the user pick from a list
            spectra = []
            for r in range(self.data_source_table.rowCount()):
                spectra.append(self.data_source_table.verticalHeaderItem(r).text())
            
            if not spectra:
                QMessageBox.warning(self, "Error", "No spectra available to configure.")
                return

            item, ok = QInputDialog.getItem(self, "Select Spectrum", 
                                            "Which spectrum do you want to configure?", 
                                            spectra, 0, False)
            if ok and item:
                self._edit_spectrum_bands(item)

    def _on_open_analysis_triggered(self):
        """Displays a dialog to select and load a previously saved analysis."""
        base_group = "/Branching_Fraction_Analyses"
        analyses = []
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                if base_group in f:
                    analyses = sorted(list(f[base_group].keys()))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read saved analyses: {e}")
            return

        if not analyses:
            QMessageBox.information(self, "No Saved Analyses", "No saved analyses were found in this project.")
            return

        item, ok = QInputDialog.getItem(self, "Open Analysis", "Select an analysis to load:", analyses, 0, False)
        if ok and item:
            self._load_saved_analysis(item)

    def _load_saved_analysis(self, analysis_name: str):
        """Restores the UI state and data from a saved analysis group."""
        path = f"/Branching_Fraction_Analyses/{analysis_name}"
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                group = f[path]
                attrs = group.attrs
                
                # 1. Restore the simple settings
                self.tolerance_edit.setText(str(attrs.get('wavenumber_tolerance', "0.1")))
                
                # 2. Update the Comboboxes
                # Note: We block signals so we don't trigger intermediate auto-refreshes
                self.level_file_combo.blockSignals(True)
                self.prev_id_combo.blockSignals(True)
                
                lvl_file = attrs.get('source_level_file')
                if lvl_file: self.level_file_combo.setCurrentText(lvl_file)
                self._on_level_file_selected() # Populate the levels table
                
                ids_file = attrs.get('source_previous_ids_file')
                if ids_file: self.prev_id_combo.setCurrentText(ids_file)
                self._on_prev_id_file_selected() # Load the ID data
                
                self.level_file_combo.blockSignals(False)
                self.prev_id_combo.blockSignals(False)

                # 3. Find and select the correct Upper Level in the table
                target_key = attrs.get('upper_level_key')
                if target_key:
                    model = self.level_table.model()
                    for r in range(model.rowCount()):
                        if str(model.index(r, 0).data()) == target_key:
                            self.level_table.selectRow(r)
                            self.current_upper_level_key = target_key
                            break

                # 4. Check the correct Spectrum Data Sources
                # source_linelists is saved as a string representation of a list
                raw_paths = attrs.get('source_linelists', "[]")
                # Simple cleanup to turn the string back into a real list of paths
                checked_paths = raw_paths.strip("[]").replace("'", "").split(", ")
                
                self.data_source_table.blockSignals(True)
                for r in range(self.data_source_table.rowCount()):
                    for c in range(self.data_source_table.columnCount()):
                        item = self.data_source_table.item(r, c)
                        if item:
                            path_in_table = item.data(Qt.UserRole)
                            item.setCheckState(Qt.Checked if path_in_table in checked_paths else Qt.Unchecked)
                self.data_source_table.blockSignals(False)

                # 5. LOAD THE ACTUAL DATA (The most important part)
                # We load the saved 'calculation_input_data' table directly.
                # This preserves normalization and exclusion states exactly as they were.
                input_data_path = f"{path}/calculation_input_data/table"
                self.master_line_data_df = self.h5_manager.read_hdf_table_robustly(self.h5_filepath, input_data_path)
                
                # Recalculate highlighting for the loaded data
                self.highlight_df = self.analysis_module.calculate_outliers(self.master_line_data_df, self.h5_filepath)
                
                # Update the display
                model = LineDataTableModel(self.master_line_data_df, self.highlight_df)
                self.line_data_table.setModel(model)
                self._format_table_columns()
                
                QMessageBox.information(self, "Load Successful", f"Analysis '{analysis_name}' has been restored.")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load analysis: {e}")