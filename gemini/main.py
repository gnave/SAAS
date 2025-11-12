# main.py
import click
import h5_manager
import importers
import versioning
import h5py
import sys
from PyQt5.QtWidgets import QApplication

@click.group()
def cli():
    """A tool for managing spectroscopy data and analysis."""
    pass

@cli.command()
def gui():
    """Launches the graphical user interface."""
    # We import here to avoid loading heavy GUI libraries for CLI commands
    from gui import MainWindow
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

@cli.command()
@click.argument('filepath')
@click.option('--author', prompt='Enter author name', help='The author of the experiment.')
def create(filepath, author):
    """Creates a new HDF5 experiment file."""
    metadata = {'author': author}
    h5_manager.create_experiment_file(filepath, metadata)
    click.echo(f"Don't forget to track this file with 'dvc-track {filepath}'")

@cli.command()
@click.argument('h5_file')
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('header_file', type=click.Path(exists=True))
def import_spectrum(h5_file, data_file, header_file):
    """
    Imports a spectrometer data file and its corresponding .hdr metadata file together.
    """
    importers.import_spectrum_pair(h5_file, data_file, header_file)


@cli.command()
@click.argument('h5_file')
@click.argument('data_file')
def import_levels(h5_file, data_file):
    """Imports spectrometer data into the HDF5 file."""
    importers.import_energy_levels(h5_file, data_file)


# ... add more import commands for each file type ...

@cli.command()
@click.argument('filepath')
def dvc_track(filepath):
    """Start tracking a data file with DVC and Git."""
    versioning.track_data_file(filepath)

@cli.command()
@click.option('-m', '--message', prompt='Enter a description of your changes', help='Version commit message.')
def save(message):
    """Saves the current state of your data and code."""
    versioning.save_progress(message)

@cli.command()
def sync():
    """Gets the latest changes from collaborators."""
    versioning.sync_project()

@cli.command()
def push():
    """Shares your saved changes with collaborators."""
    versioning.push_changes()


if __name__ == '__main__':
    cli()