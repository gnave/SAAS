# main.py (COMPLETE)
import sys
import os

# --- STANDALONE BUNDLE & GRAPHICS FIX ---
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    os.environ['LD_LIBRARY_PATH'] = bundle_dir + ":" + os.environ.get('LD_LIBRARY_PATH', '')
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

# NEW: Force X11 mode to prevent Wayland colored noise/artifacts
# This is a standard fix for Matplotlib + PyQt5 apps on modern Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Disable buggy hardware acceleration that causes the "colored pattern"
os.environ["QT_XCB_GL_INTEGRATION"] = "none"
# -----------------------------------------

import click
import h5_manager
import importers
import versioning
from gui import MainWindow
from PyQt5.QtWidgets import QApplication

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """SAAS: Atomic Spectra Analysis Tool. Default: Launch GUI."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(gui)

@cli.command()
def gui():
    """Launches the graphical user interface."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

@cli.command()
@click.argument('filepath')
@click.option('--author', prompt='Author Name')
def create(filepath, author):
    """Creates a new HDF5 project file."""
    h5_manager.create_experiment_file(filepath, {'author': author})
    click.echo(f"Created: {filepath}")

@cli.command()
@click.argument('h5_file', type=click.Path(exists=True))
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('header_file', type=click.Path(exists=True))
def import_spectrum(h5_file, data_file, header_file):
    """Imports spectrum and header pair."""
    try:
        importers.import_spectrum_pair(h5_file, data_file, header_file)
        click.echo("Success.")
    except Exception as e: click.echo(f"Error: {e}", err=True)

@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
def dvc_track(filepath):
    """Tracks a large file with DVC."""
    versioning.track_data_file(filepath)

@cli.command()
@click.option('-m', '--message', prompt='Commit Message')
def save(message):
    """Saves progress via DVC and Git."""
    versioning.save_progress(message)

@cli.command()
def sync():
    """Syncs data and code from remote."""
    versioning.sync_project()

@cli.command()
def push():
    """Pushes local changes to remote."""
    versioning.push_changes()

if __name__ == '__main__':
    cli()