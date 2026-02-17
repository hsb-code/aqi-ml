"""
Setup Verification Script
Checks if all dependencies are installed correctly
"""

import sys
from importlib import import_module

def check_import(module_name, display_name=None):
    """Try to import a module and report status"""
    if display_name is None:
        display_name = module_name
    
    try:
        import_module(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name} - {str(e)}")
        return False

def main():
    print("=" * 50)
    print("AQI Project - Setup Verification")
    print("=" * 50)
    print()
    
    print("Checking core dependencies...")
    print("-" * 50)
    
    dependencies = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("xarray", "Xarray"),
        ("rasterio", "Rasterio"),
        ("netCDF4", "netCDF4"),
        ("sklearn", "scikit-learn"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("dotenv", "python-dotenv"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
    ]
    
    results = []
    for module, name in dependencies:
        results.append(check_import(module, name))
    
    print()
    print("-" * 50)
    print("Checking deep learning framework...")
    print("-" * 50)
    
    pytorch_available = check_import("torch", "PyTorch")
    tf_available = check_import("tensorflow", "TensorFlow")
    
    if not pytorch_available and not tf_available:
        print("\n⚠️  WARNING: No deep learning framework found!")
        print("Please install either PyTorch or TensorFlow")
    
    print()
    print("-" * 50)
    print("Checking satellite data libraries...")
    print("-" * 50)
    
    check_import("sentinelsat", "sentinelsat")
    # Note: earthaccess might not be available yet
    
    print()
    print("=" * 50)
    
    success_rate = sum(results) / len(results) * 100
    print(f"Core dependencies: {sum(results)}/{len(results)} ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print("✓ Setup looks good! You're ready to start.")
    else:
        print("⚠️  Some dependencies are missing. Please install them:")
        print("   pip install -r requirements.txt")
    
    print("=" * 50)
    print()
    
    # Check Python version
    print("Python version:", sys.version)
    if sys.version_info < (3, 9):
        print("⚠️  WARNING: Python 3.9+ recommended")
    
    print()
    print("Next steps:")
    print("1. Register for satellite data access (see docs/getting_started.md)")
    print("2. Configure .env file with your credentials")
    print("3. Start with Phase 1: Research (see task.md)")

if __name__ == "__main__":
    main()
