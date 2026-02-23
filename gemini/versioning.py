# versioning.py (FULLY DOCUMENTED)

"""
versioning.py

This module provides a simplified, intention-driven Python API for managing
project versions using both Git (for source code) and DVC (Data Version Control,
for large binary datasets like HDF5 files).

By wrapping complex command-line operations into clear Python functions, this
module allows researchers to easily sync, track, and share both their analysis
scripts and their heavy spectroscopic data without needing to be experts in
version control systems.
"""

import subprocess
import sys

def _run_command(command: list):
    """
    Executes a system command and streams its output to the console in real-time.

    This helper function is used by all other functions in this module to run
    Git and DVC commands. By streaming the output, it ensures the user isn't
    left waiting with a blank screen during long data transfers or operations.

    Args:
        command (list): A list of strings representing the command and its
                        arguments (e.g., ['git', 'commit', '-m', 'message']).

    Raises:
        RuntimeError: If the executed command returns a non-zero exit code,
                      indicating a failure.
    """
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    
    # Read the output line by line as it is generated and print it to the console.
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        
    process.wait()
    
    # Check for execution failure.
    if process.returncode != 0:
        raise RuntimeError(f"Command '{' '.join(command)}' failed with exit code {process.returncode}.")

def initialize_project():
    """
    Initializes both Git and DVC in the current working directory.

    This function should be run once when starting a completely new project
    folder. It sets up the hidden '.git' and '.dvc' directories required for
    tracking code and data, respectively.
    """
    print("Initializing Git repository...")
    _run_command(['git', 'init'])
    
    print("\nInitializing DVC project...")
    _run_command(['dvc', 'init'])
    
    print("\nProject initialized. Configure your DVC remote storage next.")

def track_data_file(filepath: str):
    """
    Tells DVC to start tracking a specific large data file.

    When you track a file with DVC, the actual large file is added to the DVC
    cache (and ignored by Git), while a tiny `.dvc` pointer file is created.
    This pointer file is what Git tracks to know which version of the data
    corresponds to the current version of the code.

    Args:
        filepath (str): The path to the large data file (e.g., the HDF5 project file)
                        that needs to be tracked.
    """
    print(f"Adding {filepath} to DVC tracking...")
    _run_command(['dvc', 'add', filepath])
    print(f"Important: Now run 'git commit' to save this change.")

def save_progress(message: str):
    """
    A unified command to safely version all current changes in the project.

    This function coordinates saving the project state across both version
    control systems. It first commits any modified large datasets to DVC,
    then stages all code changes and updated DVC pointer files, and commits
    them to Git with the provided message.

    Args:
        message (str): A clear description of the progress or changes made,
                       which will be used as the commit message for both DVC and Git.
    """
    print("--- Saving project state ---")
    print("Step 1: Committing data changes to DVC...")
    # The '-a' flag commits all modified tracked files, and '-M' provides the message.
    _run_command(['dvc', 'commit', '-a', '-M', message])
    
    print("\nStep 2: Committing code and data pointers to Git...")
    # Add all changes (code and .dvc files) to the Git staging area.
    _run_command(['git', 'add', '.'])
    # Commit the staged changes to the local Git repository.
    _run_command(['git', 'commit', '-m', message])
    
    print("\n--- Progress Saved Successfully ---")
    print("Run 'push_changes()' to share with collaborators.")

def push_changes():
    """
    Uploads locally saved changes to remote storage for backup and collaboration.

    This function pushes the heavy data files to the configured DVC remote
    (like an S3 bucket or a remote server) and pushes the code/pointers to
    the configured Git remote (like GitHub or GitLab).
    """
    print("--- Pushing changes to remote storage ---")
    print("Pushing data to DVC remote...")
    _run_command(['dvc', 'push'])
    
    print("\nPushing code to Git remote...")
    # Assumes a default remote (e.g., 'origin') and upstream tracking is configured.
    _run_command(['git', 'push']) 
    print("\n--- Push Complete ---")

def sync_project():
    """
    Downloads the latest updates from remote storage to the local machine.

    This function is crucial for collaboration. It first pulls the latest code
    and `.dvc` pointer files from Git. Then, it uses DVC to pull the exact
    versions of the heavy data files that correspond to those new pointers.
    """
    print("--- Syncing project with remote storage ---")
    print("Pulling latest code and pointers from Git...")
    _run_command(['git', 'pull'])
    
    print("\nPulling corresponding data files from DVC...")
    _run_command(['dvc', 'pull'])
    print("\n--- Sync Complete. Your project is up-to-date. ---")