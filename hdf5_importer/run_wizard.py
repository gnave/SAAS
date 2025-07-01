"""
Launcher script for the Import Wizard with dependency checking.
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import numpy
        print("✓ numpy available")
    except ImportError:
        missing.append("numpy")
    
    try:
        import pandas
        print("✓ pandas available")
    except ImportError:
        missing.append("pandas")
    
    try:
        import h5py
        print("✓ h5py available")
    except ImportError:
        missing.append("h5py")
    
    try:
        import PyQt6
        print("✓ PyQt6 available")
    except ImportError:
        missing.append("PyQt6")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nTo install missing dependencies, run:")
        print("pip install " + " ".join(missing))
        print("\nOr install all requirements with:")
        print("pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies available!")
    return True

def main():
    """Main launcher."""
    print("HDF5 Spectroscopy Import Wizard")
    print("=" * 40)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\nStarting Import Wizard...")
    
    try:
        from import_wizard import main as wizard_main
        wizard_main()
    except Exception as e:
        print(f"Error starting wizard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()