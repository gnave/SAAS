#!/bin/bash

# --- SAAS Build & Install Script ---
APP_NAME="SAAS"
LOGO_NAME="SAAS_logo.png"

echo "--- Starting Build for $APP_NAME ---"

# 1. Install Python dependencies
echo "Checking dependencies..."
pip install pyinstaller pandas numpy h5py matplotlib PyQt5 click

# 2. Build the standalone executable
echo "Building executable with PyInstaller..."
pyinstaller --noconsole --onefile \
            --add-data "$LOGO_NAME:." \
            --name "$APP_NAME" main.py

# 3. Create the .desktop file dynamically based on current paths
echo "Generating Linux Desktop Entry..."
cat <<EOF > $APP_NAME.desktop
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Atomic Spectra Analysis Tool
Exec=$(pwd)/dist/$APP_NAME
Icon=$(pwd)/$LOGO_NAME
Terminal=false
Categories=Science;Education;
StartupWMClass=$APP_NAME
EOF

# 4. Finalize
chmod +x $APP_NAME.desktop
echo "------------------------------------------------"
echo "Build Complete!"
echo "1. The standalone executable is in: $(pwd)/dist/"
echo "2. To install the app to your Linux menu, run:"
echo "   cp $APP_NAME.desktop ~/.local/share/applications/"
echo "------------------------------------------------"