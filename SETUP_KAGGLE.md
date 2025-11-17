# Setting Up Kaggle API Credentials

## Quick Setup Guide

### Step 1: Get Your Kaggle API Token

1. Go to **https://www.kaggle.com/account**
2. Scroll down to the **"API"** section
3. Click **"Create New API Token"**
4. This will download a file named `kaggle.json` to your Downloads folder

### Step 2: Place the Credentials File

**On Mac/Linux:**
```bash
# Create the .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions (important!)
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows:**
```powershell
# Create the .kaggle directory
mkdir $env:USERPROFILE\.kaggle

# Move the downloaded kaggle.json file
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
```

### Step 3: Verify Setup

Run this command to verify:
```bash
python -c "import kaggle; print('✓ Kaggle API is ready!')"
```

## Alternative: Manual Setup

If you prefer to create the file manually:

1. Create the directory: `~/.kaggle/` (or `C:\Users\<username>\.kaggle\` on Windows)
2. Create a file named `kaggle.json` in that directory
3. Copy the contents from your downloaded `kaggle.json` file, or use this template:

```json
{
  "username": "your_kaggle_username",
  "key": "your_api_key_here"
}
```

Replace `your_kaggle_username` and `your_api_key_here` with your actual credentials.

## Using the Setup Script

You can also use the provided setup script:

```bash
python setup_kaggle.py
```

This will guide you through the setup process interactively.

## Troubleshooting

- **File not found**: Make sure the file is named exactly `kaggle.json` (lowercase)
- **Permission denied**: On Mac/Linux, make sure to set permissions with `chmod 600 ~/.kaggle/kaggle.json`
- **Invalid credentials**: Make sure you copied the username and key correctly from your downloaded file

