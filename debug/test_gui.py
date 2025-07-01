"""
Test script to verify GUI components work correctly.
"""

import sys
import tempfile
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from import_wizard import ImportWizard

def test_gui_startup():
    """Test that the GUI starts up correctly."""
    print("Testing GUI startup...")
    
    app = QApplication(sys.argv)
    
    try:
        wizard = ImportWizard()
        print("✓ ImportWizard created successfully")
        
        # Test basic UI elements
        assert wizard.hdf5_path_edit is not None
        assert wizard.file_list is not None
        assert wizard.preview_widget is not None
        print("✓ UI elements initialized")
        
        # Show window briefly
        wizard.show()
        print("✓ Window displayed")
        
        # Process events briefly
        QTimer.singleShot(100, app.quit)
        app.exec()
        
        print("✓ GUI test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_preview():
    """Test file preview functionality."""
    print("\nTesting file preview...")
    
    app = QApplication(sys.argv)
    
    try:
        from import_wizard import FilePreviewWidget
        
        preview = FilePreviewWidget()
        print("✓ FilePreviewWidget created")
        
        # Test CSV preview
        csv_file = "cr_demo_data/CrII_CS.csv"
        if Path(csv_file).exists():
            preview.preview_file(csv_file)
            print("✓ CSV file preview works")
        
        # Test HDR preview
        hdr_file = "cr_demo_data/cr042416.005_r.hdr"
        if Path(hdr_file).exists():
            preview.preview_file(hdr_file)
            print("✓ HDR file preview works")
        
        return True
        
    except Exception as e:
        print(f"✗ File preview test failed: {e}")
        return False

def test_column_mapping():
    """Test column mapping widget."""
    print("\nTesting column mapping...")
    
    app = QApplication(sys.argv)
    
    try:
        from import_wizard import ColumnMappingWidget
        
        mapping = ColumnMappingWidget()
        print("✓ ColumnMappingWidget created")
        
        # Test mapping setup
        test_columns = ['wavenumber', 'lower_energy', 'upper_energy', 'lower_desig', 'upper_desig']
        mapping.setup_mapping(test_columns, 'identifications')
        print("✓ Column mapping setup works")
        
        # Test get mapping
        column_map = mapping.get_mapping()
        print(f"✓ Column mapping retrieved: {column_map}")
        
        return True
        
    except Exception as e:
        print(f"✗ Column mapping test failed: {e}")
        return False

def main():
    """Run all GUI tests."""
    print("=== GUI Tests ===\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    tests = [
        test_gui_startup,
        test_file_preview,
        test_column_mapping
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== GUI Tests Complete: {passed}/{len(tests)} passed ===")
    
    if passed == len(tests):
        print("\n🎉 All GUI tests passed! You can now run: python import_wizard.py")
    else:
        print("\n⚠️  Some GUI tests failed. Check dependencies and PyQt6 installation.")

if __name__ == "__main__":
    main()