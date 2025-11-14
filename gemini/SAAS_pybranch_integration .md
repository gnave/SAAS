# User

It's now time to start adding the first analysis step. This will be to derive branching fractions and transition probabilities from the experimental spectra. We can use code from the following Github repository:

https://github.com/gnave/pybranch/tree/Gemini

We will also need to add a function to match the wavenumbers in the experimental linelist to the previous identifications. The new analysis unit will probably need a new analysis window to avoid confusion. Tell me how you would approach this problem

# Google AI studio

This is an excellent and logical progression for the project. Moving from data ingest and browsing to the first real analysis step is the primary goal. Your proposal to use the `pybranch` repository and create a new, dedicated analysis window is the correct approach. It maintains a clean separation between data management and scientific computation.

Here is a breakdown of how I would approach this problem, outlining the necessary changes and the new components we will build.

### Core Challenge: Adapting File-Based Code

The `pybranch/data_io.py` code you've provided is written in a common, file-centric style: it expects to read data directly from individual `.dat`, `.hdr`, and `.lin` files on the disk.

Our application, however, has already consolidated this data into a structured HDF5 file. The central task is to create a "bridge" between our HDF5 data store and the scientific logic contained within the `pybranch` code.

Directly using the `data_io.py` file is not feasible. Instead, we will **extract its core scientific logic** and adapt it to work with data that is already in memory (as Pandas DataFrames and NumPy arrays), which we will read from our HDF5 file.

---

### A Step-by-Step Implementation Plan

#### Step 1: Create a New `analysis.py` Module

To maintain separation of concerns, all new scientific code will go into a new file, `analysis.py`. This module will contain the adapted logic from `pybranch` and the new wavenumber matching function.

**A. Wavenumber Matching Function:**
First, we'll implement the function you requested. This is a classic data-merging task, and Pandas is the ideal tool.

```python
# analysis.py (New File)
import pandas as pd
import numpy as np

def match_wavenumbers(experimental_linelist: pd.DataFrame, 
                      previous_ids: pd.DataFrame, 
                      tolerance: float = 0.02) -> pd.DataFrame:
    """
    Matches lines between an experimental linelist and a list of previous identifications.

    Args:
        experimental_linelist: DataFrame from the HDF5 file (e.g., from a .lin file).
                               Must contain a 'wavenumber' column.
        previous_ids: DataFrame of previous identifications from the HDF5 file.
                      Must also contain a 'wavenumber' column.
        tolerance: The matching tolerance in wavenumbers (cm⁻¹).

    Returns:
        A new DataFrame containing the merged data for all matched lines.
    """
    # Sort both dataframes by wavenumber, which is required for merge_asof
    exp_sorted = experimental_linelist.sort_values('wavenumber').copy()
    ids_sorted = previous_ids.sort_values('wavenumber').copy()

    # Use pandas.merge_asof for a fast, nearest-neighbor merge
    # This finds the closest previous_id for each experimental line.
    merged_df = pd.merge_asof(
        exp_sorted,
        ids_sorted,
        on='wavenumber',
        direction='nearest',
        tolerance=tolerance,
        suffixes=('_exp', '_id') # Suffixes to distinguish columns like 'intensity'
    )
    
    # Filter out any rows where no match was found within the tolerance
    matched_lines = merged_df.dropna(subset=['intensity_id']).reset_index(drop=True)
    
    return matched_lines
```

**B. Adapting `pybranch` Logic:**
We will review the other files in the `pybranch` repository (which I can now access) and extract the core functions that calculate branching fractions. We will rewrite them to accept our HDF5 data structures (DataFrames, arrays) as direct inputs, rather than file paths. For example, a function might look like this conceptually:

```python
# analysis.py (Conceptual)

def calculate_branching_fractions(matched_lines: pd.DataFrame, 
                                  energy_levels: pd.DataFrame, 
                                  calculations: pd.DataFrame) -> pd.DataFrame:
    """
    Derives branching fractions and transition probabilities from matched lines.
    
    (This function will contain the core scientific logic adapted from pybranch)
    """
    # ... scientific calculations happen here ...
    
    # The function returns a new DataFrame with the results
    results_df = pd.DataFrame() # Placeholder for the results
    # ... populate results_df ...
    
    return results_df
```

#### Step 2: Design the New Analysis Window (`gui.py`)

As you suggested, a new window is the best approach. We'll create a new `AnalysisWindow(QDialog)` class.

**Key Components of the `AnalysisWindow`:**

1.  **Spectrum Selector (Dropdown):** The first and most important field. This will be populated with the names of all imported spectra (e.g., `cr042416_005_r`).
2.  **Input Data Selectors (Dropdowns):** Once a spectrum is selected, these dropdowns will be dynamically populated with the available datasets *within that spectrum's group*.
    *   Dropdown for "Experimental Linelist" (listing tables from `/Spectra/{name}/Raw_Linelists` and `/Spectra/{name}/Calibrated_Linelists`).
    *   Dropdown for "Previous Identifications" (listing tables from the top-level `/Previous_Identifications` group).
    *   Dropdown for "Energy Levels" (from `/Levels`).
    *   Dropdown for "Calculations" (from `/Calculations`).
3.  **Parameters (Line Edits):** A field for the user to enter the `tolerance` for wavenumber matching.
4.  **"Run Analysis" Button:** This will trigger the backend logic.
5.  **Output Log (Text Area):** A read-only text box to display progress, results, and any warnings or errors.

#### Step 3: Update `h5_manager.py` and `importers.py`

To store the analysis results, we need a place for them. Following your file description, we will create an `Analysis` group within each spectrum.

1.  In `importers.py`, within the `import_spectrum_pair` function, we will add one line to create this new group when a spectrum is first imported:
    ```python
    # importers.py snippet
    ...
    analysis_group = f"{base_spectrum_group}/Analysis"
    ...
    with h5py.File(h5_filepath, 'a') as f:
        ...
        f.create_group(analysis_group)
    ```

2.  The `analysis.py` module will then call `h5_manager.add_pandas_table` to save its results DataFrame into this new group (e.g., `/Spectra/cr042416_005_r/Analysis/branching_fractions_v1`).

#### Step 4: Integrate into the `MainWindow` (`gui.py`)

1.  Add a new button to the main toolbar: `"Run Analysis..."`.
2.  Connect this button to a new method, `_show_analysis_window`, which will create and show an instance of our new `AnalysisWindow`.

### The User Workflow

This new design will result in the following user workflow:

1.  The user clicks the "Run Analysis..." button on the main window.
2.  The Analysis Window appears.
3.  The user selects the target spectrum (e.g., `cr042416_005_r`) from the first dropdown.
4.  The other dropdowns automatically populate with the relevant linelists, level lists, etc. The user selects the specific datasets they wish to use for this run.
5.  The user enters the matching tolerance.
6.  The user clicks "Run Analysis".
7.  In the background, the GUI controller:
    *   Reads the selected DataFrames and arrays from the HDF5 file.
    *   Calls `analysis.match_wavenumbers()` with the linelist and previous IDs.
    *   Calls `analysis.calculate_branching_fractions()` with the output of the matching and the other selected tables.
    *   Saves the resulting DataFrame back to the appropriate `/Analysis` group in the HDF5 file.
    *   Prints a summary of the results to the output log.
8.  The user can then close the Analysis Window and see the new results dataset appear in the HDF5 browser tree in the main window.

This approach cleanly separates the UI from the analysis logic, reuses our existing HDF5 infrastructure, and provides a powerful, extensible framework for future analysis modules.

Does this approach align with your vision for how the analysis unit should function?

# User

The analysis needs to be interactive because the input data might be affected by blends, inaccurate measurements and all sorts of other things. The analysis window will need two large sections. The first would display a table for each upper energy level, with the rows being the wavenumbers and IDs of all the lines from the upper level. The columns would contain  the intensity and its uncertainty in the linelists of each spectrum for each upper level. The second would provide plots of the region around each wavenumber in each spectrum. It should be possible to select or deselect lines in both the table and the plot window for inclusion in the analysis. 

Two side panels would be needed. The first would enable the user to select the upper energy level of interest and see its details. The second would provide a list of all the spectra and its datasets

Does this make snes. If not, ask me for any extra information you need

# Google AI studio

Yes, this makes perfect sense. Thank you for the detailed and clear description. This is a significant leap in functionality, moving from a simple "run" dialog to a full-fledged, interactive analysis dashboard. Your design is excellent and mirrors the workflow of professional scientific analysis tools.

My previous, simpler proposal is inadequate. Your interactive design is the correct approach.

I understand the requirements. The core challenge is building a complex GUI that links multiple, synchronized views (a side panel for level selection, another for data selection, a main table for line data, and a plot view) and allows user interaction (selecting/deselecting lines) to flow between them.

Here is a detailed plan for how I would architect and build this interactive analysis window.

---

### Architectural Plan for the Interactive Analysis Window

We will create a new, complex `AnalysisWindow` class in `gui.py`. It will be a `QMainWindow` itself, rather than a simple `QDialog`, to accommodate toolbars, status bars, and dockable side panels for a richer user experience.

#### 1. Main Window Layout (`AnalysisWindow` class)

The window will be built using a central widget and two dockable side panels (`QDockWidget`).

*   **Left Dock Widget (`Level Selector`):**
    *   **Purpose:** To choose the upper energy level to analyze.
    *   **Components:**
        *   A `QTableView` to list all available upper energy levels from the selected "Energy Levels" file. We will filter this to only show levels with a defined `lifetime` and possibly parity = 0, as per the science.
        *   Several read-only `QLineEdit` fields below the table to display the full details of the currently selected level (e.g., Key, Energy, J, Designation).
    *   **Interaction:** Clicking a level in this table will be the primary trigger that updates all other views.

*   **Right Dock Widget (`Data Source Selector`):**
    *   **Purpose:** To select which datasets are used in the analysis.
    *   **Components:**
        *   A `QTreeView` that mirrors the structure of the main application's browser. It will show the entire HDF5 file structure.
        *   Checkboxes next to relevant items (calibrated linelists, spectra) will allow the user to select which datasets to include in the main table and plots. For example, you could check three different `Calibrated_Linelists` to compare their intensities side-by-side.

*   **Central Widget (`QSplitter`):**
    *   This will be a vertical splitter containing the two main sections you described.
    *   **Top Pane (`Line Data Table`):**
        *   A `QTableView` to display the detailed line information.
        *   **Rows:** Each row will represent a single transition *from* the currently selected upper level.
        *   **Columns:**
            *   A checkbox column for `Include in Fit`. This is the core interactive element.
            *   `Wavenumber` (from Previous IDs).
            *   `Lower Level Key`.
            *   `Intensity (Spectrum A)`, `Uncertainty (Spectrum A)`.
            *   `Intensity (Spectrum B)`, `Uncertainty (Spectrum B)`.
            *   ... additional columns for each spectrum's linelist selected in the right-hand panel.
    *   **Bottom Pane (`Plot View`):**
        *   A `matplotlib` canvas.
        *   When a row is selected in the Line Data Table, this plot will automatically update to show the region of the spectrum around that line's wavenumber. If multiple spectra are selected, it can show them as overlaid plots.

#### 2. The Data Model and Workflow (The "Controller")

This is the most critical part. We need a controller class or a set of methods within `AnalysisWindow` to manage the flow of data.

1.  **Initialization:** The window is launched. The right-hand `Data Source Selector` is populated with the HDF5 tree. The user selects the primary "Energy Levels" and "Previous IDs" files.
2.  **Level Selection (Trigger):** The user clicks an upper level in the Left Dock Widget.
    *   The controller queries the "Previous IDs" table for all lines originating from this upper level's key.
    *   This forms the basis of the `Line Data Table`.
    *   For each of these lines, and for each *checked* calibrated linelist in the right-hand panel, the controller finds the corresponding line's intensity and uncertainty.
    *   The `Line Data Table` is populated with this rich, merged dataset. All checkboxes are initially checked.
3.  **Line Selection (Interaction 1):** The user clicks a row in the `Line Data Table`.
    *   The `Plot View` immediately updates. It reads the raw spectral data for each *checked* spectrum and plots the small region around the selected line's wavenumber.
4.  **Inclusion/Exclusion (Interaction 2):** The user unchecks a box for a line in the `Line Data Table`.
    *   The model behind the table is updated to mark this line as "excluded".
    *   This might visually grey out the line in the table and/or the plot.
5.  **"Calculate" Button:** The user clicks a "Calculate Branching Fractions" button (likely in a toolbar).
    *   The controller takes the current state of the `Line Data Table`.
    *   It filters for only the rows where `Include in Fit` is checked.
    *   It passes this filtered DataFrame to a function in `analysis.py` (which we will adapt from `pybranch`).
    *   The results are returned.
6.  **Saving Results:** A "Save Results" button will open a dialog asking for a name for this analysis run. The resulting table of branching fractions will be saved to the `/Analysis` group of the relevant spectrum, with user-provided notes and a timestamp as metadata.

### Questions for You

This design is detailed, but a few points need clarification to ensure it matches your scientific needs perfectly:

1.  **Uncertainty Source:** You mentioned "intensity and its uncertainty". Where does the uncertainty value come from? Is it a column in the calibrated linelist file, or is it derived from the fit parameters (like `epstot`, `epsran`) in the raw `.lin` file?
2.  **Plotting Interaction:** When a line is unchecked in the table, how should it be represented in the plot? Should it disappear, be greyed out, or be marked with an 'x'?
3.  **Initial Data Loading:** Is it acceptable for the user to first select a "master" Energy Level file and a "master" Previous IDs file that will be used to populate the initial views, before they start selecting individual linelists to compare?

This interactive dashboard is a significant but achievable step. It moves the application from a simple data viewer into a powerful, purpose-built analysis tool.

# User

1. The uncertainty is calculated from the snr and calubration uncertainty in the function aggregate_observed_data in the file analysis.py in the github repository I linked to before. 
2. The line should be greyed out in both the plot window and the table.
3. Yes, a master energy level file and master previous IDs file should be used for the analysis. 
The results should be saved to a "Calculated Branching Fractions" top-level group in the hdf5 file, as they are calculated from several different spectra.

# Google AI studio

Thank you. Your answers provide the perfect clarity needed to finalize the architectural plan.

Your reasoning for saving the results to a top-level `Calculated Branching Fractions` group is absolutely correct. Since the analysis combines data from multiple spectra, the results are a project-level asset, not something that belongs to a single spectrum. This is an excellent refinement.

Here is the definitive, step-by-step plan to build the interactive analysis window, incorporating all your requirements.

---

### Definitive Architectural and Implementation Plan

#### Step 1: Backend Setup (`h5_manager.py` and `analysis.py`)

1.  **HDF5 Structure (`h5_manager.py`):**
    *   Add the new top-level group to the `HDF5_STRUCTURE` dictionary:
        ```python
        HDF5_STRUCTURE = {
            'Calculations': [],
            'Levels': [],
            'Standard_Lamp_Calibrations': [],
            'Previous_Identifications': [],
            'Spectra': [],
            'Calculated_Branching_Fractions': [] # NEW
        }
        ```
    *   This group will now be automatically created with every new project file.

2.  **Analysis Logic (`analysis.py`):**
    *   I will integrate the core logic from the `aggregate_observed_data` function in the `pybranch` repository.
    *   This new function in our `analysis.py` will be designed to accept our in-memory Pandas DataFrames as input, rather than file paths.
    *   It will perform the necessary joins and calculations to produce the single, large DataFrame that will serve as the "model" for our main analysis table. This DataFrame will contain the merged data, including the calculated uncertainties.
    *   A second function will be adapted to take this merged DataFrame (potentially filtered by the user), calculate the final branching fractions, and return a results DataFrame.

#### Step 2: GUI Implementation (`gui.py`)

We will create a new file, `analysis_window.py`, to contain the code for this complex new window, keeping the main `gui.py` clean. The main window will simply be responsible for launching it.

**`analysis_window.py` - The Core Components:**

1.  **`AnalysisWindow(QMainWindow)` Class:**
    *   **Initialization (`__init__`):**
        *   It will accept the path to the current HDF5 file.
        *   It will build the main layout: two dock widgets (left and right) and a central splitter.
        *   It will contain placeholder widgets for the views (tables, plots).
        *   It will have a toolbar with "Load Data", "Calculate", and "Save Results" buttons.

2.  **Left Dock: Level Selector:**
    *   A `QComboBox` to select the "master" Energy Levels table from the `/Levels` group.
    *   A `QTableView` (`level_view`) to display the contents of the selected table.
    *   Read-only `QLineEdit`s to show details of the selected level.
    *   **Signal:** `level_view.selectionModel().selectionChanged.connect(self._on_level_selected)`

3.  **Right Dock: Data Source Selector:**
    *   A `QComboBox` to select the "master" Previous IDs table from `/Previous_Identifications`.
    *   A `QTreeView` (`data_source_tree`) that displays the HDF5 structure, but with checkboxes enabled for specific items (e.g., datasets within `/Calibrated_Linelists` groups).
    *   **Signal:** `data_source_tree.model().itemChanged.connect(self._on_data_source_changed)`

4.  **Central Widget: Main View:**
    *   A vertical `QSplitter`.
    *   **Top Pane (`line_data_table`):** A `QTableView`. We will create a custom `QAbstractTableModel` for this table that can handle checkboxes and cell coloring (for greying out rows).
    *   **Bottom Pane (`plot_widget`):** A `matplotlib` canvas.
    *   **Signal:** `line_data_table.selectionModel().selectionChanged.connect(self._on_line_selected)`

#### Step 3: Controller Logic (Methods within `AnalysisWindow`)

This is the heart of the interactivity.

1.  **`_on_level_selected()` (Primary Trigger):**
    *   **Get Inputs:** Get the selected upper level key from the left dock, the path to the master IDs file from the right dock, and the list of all *checked* calibrated linelist paths from the tree view.
    *   **Call Backend:** Pass these inputs to the `analysis.py` function that aggregates the data.
    *   **Receive DataFrame:** This function returns a single, comprehensive DataFrame containing all the merged data and calculated uncertainties. Store this as `self.master_df`.
    *   **Update View:** Set the custom model for the `line_data_table` with this new `self.master_df`. The table will populate with all the lines for the selected upper level.

2.  **`_on_line_selected()` (Plot Trigger):**
    *   Get the selected row from `line_data_table`.
    *   Extract the `wavenumber` for that row.
    *   For each checked spectrum in the right-hand tree, read the raw spectrum data.
    *   Update the `matplotlib` plot to show the region around the selected wavenumber for all selected spectra.
    *   If the selected row is marked as "excluded", draw its plot line in grey.

3.  **Handling the Checkbox:**
    *   The custom table model's `setData` method will be implemented to handle checkbox state changes. When a box is unchecked:
    *   It will add a boolean `_is_excluded` column to `self.master_df` and set the value for that row to `True`.
    *   It will emit a `dataChanged` signal. The table view will automatically re-render, and we'll use this signal to trigger a re-draw of the plot, allowing the line color to be updated to grey.

4.  **`_calculate_clicked()`:**
    *   Create a filtered DataFrame: `active_lines_df = self.master_df[self.master_df['_is_excluded'] == False]`.
    *   Pass `active_lines_df` to the branching fraction calculation function in `analysis.py`.
    *   Display the results in a simple message box or a new results table. Store the result DataFrame in `self.result_df`.

5.  **`_save_results_clicked()`:**
    *   If `self.result_df` exists:
    *   Open a dialog to ask the user for a name for the results dataset (e.g., `bf_run_20251111_CrII_level_X`).
    *   Call `h5_manager.add_pandas_table`, saving `self.result_df` to the `/Calculated_Branching_Fractions` group with appropriate metadata (date, source datasets used, etc.).
    *   Refresh the main window's browser tree to show the new result.

This design provides a clear and robust separation of concerns: the backend `analysis.py` knows nothing about the GUI, and the `AnalysisWindow` acts as a sophisticated controller orchestrating the flow of data from the HDF5 file, through the analysis functions, and back to the user views.