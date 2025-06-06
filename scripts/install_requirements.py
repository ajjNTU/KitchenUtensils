#!/usr/bin/env python3
"""
Install required packages for multi-object dataset generation.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install required packages."""
    print("🔧 Installing required packages for multi-object dataset generation...\n")
    
    required_packages = [
        "albumentations",
        "opencv-python",
        "pillow",
        "numpy",
        "pyyaml"
    ]
    
    success_count = 0
    for package in required_packages:
        if install_package(package):
            success_count += 1
        print()  # Add spacing
    
    print(f"📊 Installation complete: {success_count}/{len(required_packages)} packages installed")
    
    if success_count == len(required_packages):
        print("🎉 All packages installed successfully!")
        print("\nYou can now run the multi-object dataset generator:")
        print("   python scripts/create_multi_object_dataset.py")
    else:
        print("⚠️  Some packages failed to install. Please install them manually.")

if __name__ == "__main__":
    main() 