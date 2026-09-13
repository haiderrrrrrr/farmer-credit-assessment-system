#!/usr/bin/env python3
"""
Setup script for Farmer Credit Assessment System

This script helps set up the environment and dependencies for the farmer credit assessment system.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("[PACKAGE] Installing Python dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("[OK] Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error installing dependencies: {e}")
        return False

def install_tech_stack():
    """Install complete tech stack for the project"""
    print("[START] Installing complete tech stack...")
    
    # Complete tech stack packages
    tech_packages = [
        "Flask", "Flask-Login", "psycopg2-binary",
        "scikit-learn", "pandas", "numpy", 
        "matplotlib", "seaborn", "plotly",
        "xgboost", "lightgbm",
        "papermill", "jupyter", "python-dotenv",
        "requests", "beautifulsoup4", "lxml", "openpyxl"
    ]
    
    print(f"[PACKAGE] Installing {len(tech_packages)} packages...")
    
    for package in tech_packages:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"   [OK] {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"   [ERROR] Failed to install {package}: {e}")
            print(f"   [WARN]  Continuing with other packages...")
    
    print("[OK] Tech stack installation completed!")
    return True

def check_python_version():
    """Check if Python version is compatible"""
    print("[PYTHON] Checking Python version...")
    
    if sys.version_info < (3, 7):
        print("[ERROR] Python 3.7 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"[OK] Python version OK: {sys.version}")
        return True

def check_dataset_structure():
    """Check if dataset folder exists and has expected structure"""
    print("[DATA] Checking dataset structure...")
    
    dataset_path = Path("Dataset")
    if not dataset_path.exists():
        print("[WARN]  Dataset folder not found!")
        print("Please create a 'Dataset' folder and place your agricultural datasets there.")
        print("Expected structure:")
        print("  Dataset/")
        print("    |---- Agricultural Crop Yield in Indian States Dataset/")
        print("    |---- Crop Recommendation Dataset/")
        print("    |---- Indian Agriculture and Climate Dataset/")
        print("    |---- Rainfall in Pakistan/")
        print("    `---- ... (other datasets)")
        return False
    
    # Check for some key files
    key_files = [
        "Agricultural Crop Yield in Indian States Dataset/crop_yield.csv",
        "Crop Recommendation Dataset/Crop_recommendation.csv"
    ]
    
    missing_files = []
    for file_path in key_files:
        if not (dataset_path / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("[WARN]  Some expected dataset files are missing:")
        for file_path in missing_files:
            print(f"    - {file_path}")
        print("The system will work with available datasets.")
    else:
        print("[OK] Key dataset files found!")
    
    return True

def create_directory_structure():
    """Create necessary directories"""
    print("[DIR] Creating directory structure...")
    
    directories = [
        "logs",
        "models", 
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
    print("[OK] Directory structure created!")

def display_next_steps():
    """Display next steps for the user"""
    print("\n[TARGET] Setup Complete! Next Steps:")
    print()
    print("[DOCS] For detailed instructions, see README.md")

def main():
    """Main setup function"""
    print("[AGRI] Farmer Credit Assessment System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("[WARN]  Continuing despite installation issues...")
    
    # Install tech stack
    install_tech_stack()

    # Check dataset structure
    check_dataset_structure()
    
    # Create directories
    create_directory_structure()
    
    # Show next steps
    display_next_steps()
    
    print("\n[OK] Setup completed successfully!")

if __name__ == "__main__":
    main()


