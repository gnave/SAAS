# main.py (FULLY DOCUMENTED)

import click
import h5_manager
import importers
import versioning
import sys

# The 'click' library is used to create a clean and composable command-line interface (CLI).
# The `@click.group()` decorator turns the `cli` function into a container for other commands.
@click.group()
def cli():
    """
    SAAS: A Command-Line Tool for Managing Spectroscopy Data and Analysis.

    This tool provides both a command-line interface for scripting and a graphical
    user interface for interactive work. Use `[COMMAND] --help` for more
    information on a specific command.
    """
    pass

@cli.command()
def gui():
    """Launches the graphical user interface (GUI)."""
    # We import the GUI components here, inside the command function, rather than
    # at the top of the file. This is a crucial optimization. It means that users
    # running CLI-only commands (like 'create' or 'import-spectrum') do not have to
    # wait for the large PyQt5 libraries to be loaded into memory, making the
    # CLI much faster and more responsive.
    from gui import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

@cli.command()
@click.argument('filepath')
@click.option('--author', prompt='Enter author name', help='The author of the experiment.')
def create(filepath, author):
    """
    Creates a new, empty HDF5 project file with the standard group structure.

    FILEPATH: The full path for the new .h5 file to be created.
    """
    metadata = {'author': author}
    h5_manager.create_experiment_file(filepath, metadata)
    click.echo(f"Successfully created HDF5 project file at: {filepath}")

@cli.command()
@click.argument('h5_file', type=click.Path(exists=True))
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('header_file', type=click.Path(exists=True))
def import_spectrum(h5_file, data_file, header_file):
    """
    Imports a spectrum data file (.raw) and its corresponding header file (.hdr).

    H5_FILE: Path to the existing HDF5 project file.
    DATA_FILE: Path to the raw binary spectrum data file.
    HEADER_FILE: Path to the corresponding .hdr metadata file.
    """
    try:
        importers.import_spectrum_pair(h5_file, data_file, header_file)
        click.echo("Spectrum imported successfully.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

# Note: The 'import_levels' command was present in the original main.py but the
# corresponding function 'importers.import_energy_levels' does not exist in
# the provided importers.py file. It is left here as a placeholder for completeness.
@cli.command()
@click.argument('h5_file')
@click.argument('data_file')
def import_levels(h5_file, data_file):
    """(Placeholder) Imports energy level data into the HDF5 file."""
    click.echo("Note: 'import_levels' function is not yet implemented in importers.py.")
    # importers.import_energy_levels(h5_file, data_file)


# --- Version Control Commands ---
# These commands act as simple wrappers around the functions in versioning.py,
# making version control accessible directly from the main application CLI.

@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
def dvc_track(filepath):
    """
    Starts tracking a large data file (like the HDF5 project file) with DVC.

    This command should be run once for each large file you want to version.
    It creates a small .dvc pointer file that should be committed to Git.
    """
    try:
        versioning.track_data_file(filepath)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.option('-m', '--message', prompt='Enter a description of your changes', help='A concise message describing the changes being saved.')
def save(message):
    """
    Saves the current state of the project.

    This is a high-level command that performs two main actions:
    1. Commits any changes to DVC-tracked data files.
    2. Commits all code changes and DVC pointer files to Git.
    """
    try:
        versioning.save_progress(message)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
def sync():
    """
    Pulls the latest data and code from remote storage.

    This command runs 'git pull' to get the latest code and pointer files,
    then runs 'dvc pull' to download any updated data files from DVC remote storage.
    """
    try:
        versioning.sync_project()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
def push():
    """
    Pushes saved changes to remote storage.

    This command runs 'dvc push' to upload changed data to DVC remote storage,
    then runs 'git push' to upload code and pointer files to the Git remote.
    """
    try:
        versioning.push_changes()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

# This standard Python construct ensures that the `cli` function is called
# when the script is executed directly from the command line.
if __name__ == '__main__':
    cli()