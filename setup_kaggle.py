#!/usr/bin/env python3
"""
Kaggle API Setup Helper Script

This script helps you set up Kaggle API credentials.
"""

import os
import json
from pathlib import Path

def setup_kaggle_credentials():
    """Guide user through Kaggle API setup."""
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    print("=" * 60)
    print("Kaggle API Setup")
    print("=" * 60)
    
    # Check if already exists
    if kaggle_json.exists():
        print(f"\n✓ Kaggle credentials already exist at: {kaggle_json}")
        response = input("\nDo you want to replace them? (y/n): ").lower()
        if response != 'y':
            print("Keeping existing credentials.")
            return
    
    print("\nTo get your Kaggle API credentials:")
    print("1. Go to: https://www.kaggle.com/account")
    print("2. Scroll down to the 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This will download a file named 'kaggle.json'")
    print("\n" + "=" * 60)
    
    # Option 1: User provides the file path
    print("\nOption 1: Provide path to downloaded kaggle.json file")
    file_path = input("Enter the path to your kaggle.json file (or press Enter to skip): ").strip()
    
    if file_path:
        source_file = Path(file_path)
        if source_file.exists():
            # Create .kaggle directory
            kaggle_dir.mkdir(exist_ok=True)
            
            # Copy file
            import shutil
            shutil.copy2(source_file, kaggle_json)
            
            # Set permissions (Unix/Mac)
            if os.name != 'nt':  # Not Windows
                os.chmod(kaggle_json, 0o600)
            
            print(f"\n✓ Credentials copied to: {kaggle_json}")
            print("✓ Setup complete!")
            return
        else:
            print(f"✗ File not found: {file_path}")
    
    # Option 2: Manual entry
    print("\nOption 2: Enter credentials manually")
    print("(You can find these in your downloaded kaggle.json file)")
    
    username = input("Enter your Kaggle username: ").strip()
    key = input("Enter your Kaggle API key: ").strip()
    
    if username and key:
        # Create credentials dictionary
        credentials = {
            "username": username,
            "key": key
        }
        
        # Create .kaggle directory
        kaggle_dir.mkdir(exist_ok=True)
        
        # Write credentials file
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        # Set permissions (Unix/Mac)
        if os.name != 'nt':  # Not Windows
            os.chmod(kaggle_json, 0o600)
        
        print(f"\n✓ Credentials saved to: {kaggle_json}")
        print("✓ Setup complete!")
        
        # Verify
        try:
            import kaggle
            print("\n✓ Kaggle API is ready to use!")
        except ImportError:
            print("\n⚠️  Kaggle package not installed. Install it with: pip install kaggle")
    else:
        print("\n✗ Setup cancelled. No credentials provided.")
        print("\nYou can run this script again later or manually:")
        print(f"1. Download kaggle.json from https://www.kaggle.com/account")
        print(f"2. Place it at: {kaggle_json}")

if __name__ == '__main__':
    setup_kaggle_credentials()

