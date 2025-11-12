### 3. Versioning Made Easy (`versioning.py`)
###  This is the key to making the system user-friendly. This module wraps Git and DVC commands into simple, intention-driven functions.

###  python

# versioning.py
import subprocess
import sys

def _run_command(command):
    """Helper function to run a command and stream its output."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command '{' '.join(command)}' failed.")

def initialize_project():
    """Initializes Git and DVC in the current directory."""
    print("Initializing Git repository...")
    _run_command(['git', 'init'])
    print("\nInitializing DVC project...")
    _run_command(['dvc', 'init'])
    print("\nProject initialized. Configure your DVC remote storage next.")

def track_data_file(filepath):
    """Tells DVC to start tracking a data file."""
    print(f"Adding {filepath} to DVC tracking...")
    _run_command(['dvc', 'add', filepath])
    print(f"Important: Now run 'git commit' to save this change.")

def save_progress(message):
    """A user-friendly command to version all current changes."""
    print("--- Saving project state ---")
    print("Step 1: Committing data changes to DVC...")
    # The '-a -M' flags commit all changed DVC-tracked files with a message
    _run_command(['dvc', 'commit', '-a', '-M', message])
    
    print("\nStep 2: Committing code and data pointers to Git...")
    _run_command(['git', 'add', '.'])
    _run_command(['git', 'commit', '-m', message])
    print("\n--- Progress Saved Successfully ---")
    print("Run 'push_changes()' to share with collaborators.")


def push_changes():
    """Pushes both data (DVC) and code (Git) to remotes."""
    print("--- Pushing changes to remote storage ---")
    _run_command(['dvc', 'push'])
    _run_command(['git', 'push']) # Assumes a remote like 'origin' is configured
    print("\n--- Push Complete ---")

def sync_project():
    """Pulls the latest code and data from remotes."""
    print("--- Syncing project with remote storage ---")
    _run_command(['git', 'pull'])
    _run_command(['dvc', 'pull'])
    print("\n--- Sync Complete. Your project is up-to-date. ---")