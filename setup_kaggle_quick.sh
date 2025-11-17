#!/bin/bash
# Quick setup script for Kaggle credentials

echo "Kaggle API Setup"
echo "================"
echo ""
echo "This script will help you set up your Kaggle API credentials."
echo ""

# Check if kaggle.json exists in Downloads
if [ -f ~/Downloads/kaggle.json ]; then
    echo "✓ Found kaggle.json in Downloads folder"
    read -p "Do you want to use this file? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p ~/.kaggle
        cp ~/Downloads/kaggle.json ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
        echo "✓ Credentials set up successfully!"
        echo "✓ Location: ~/.kaggle/kaggle.json"
        exit 0
    fi
fi

echo ""
echo "To get your Kaggle API credentials:"
echo "1. Go to: https://www.kaggle.com/account"
echo "2. Scroll down to 'API' section"
echo "3. Click 'Create New API Token'"
echo "4. This downloads kaggle.json to your Downloads folder"
echo ""
echo "Then run this script again, or manually:"
echo "  mkdir -p ~/.kaggle"
echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
echo "  chmod 600 ~/.kaggle/kaggle.json"
