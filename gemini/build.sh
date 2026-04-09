#!/bin/bash

APP_NAME="SAAS"
LOGO_NAME="SAAS_logo.png"

echo "--- Starting Robust Build for $APP_NAME ---"

# 1. Clean previous attempts
rm -rf build/ dist/

# 2. Run PyInstaller using the SPEC file
# This is much more reliable than passing flags via CLI
echo "Building executable via SAAS.spec..."
pyinstaller --clean SAAS.spec

# 3. Create/Update the .desktop file
echo "Updating Linux Desktop Entry..."
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

chmod +x $APP_NAME.desktop

echo "------------------------------------------------"
echo "Build Complete!"
echo "To test: ./dist/SAAS"
echo "To install menu icon: cp $APP_NAME.desktop ~/.local/share/applications/"