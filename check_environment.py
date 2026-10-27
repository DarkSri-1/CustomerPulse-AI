"""
Stage 1 Environment Verification Script for CustomerPulse AI.
Run this script to confirm the workspace, virtual environment, and core packages are ready.
"""
import sys
import importlib
from pathlib import Path

# Import project paths from src.config
try:
    from src.config import (
        BASE_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    )
except ImportError as exc:
    print(f"[FAIL] Could not import src.config: {exc}")
    print("Ensure you are running this script from the CustomerPulse-AI root folder.")
    sys.exit(1)


def check_python_version() -> bool:
    """Verify Python version is 3.9 or higher."""
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"Python Version      : {major}.{minor}.{sys.version_info.micro}")
    print(f"Python Executable   : {sys.executable}")
    if (major, minor) < (3, 9):
        print("[FAIL] Python 3.9 or higher is required.")
        return False
    return True


def check_virtual_env() -> bool:
    """Verify the script is running inside a virtual environment (.venv)."""
    in_venv = sys.prefix != sys.base_prefix
    venv_path = Path(sys.prefix)
    print(f"Virtual Environment : {'Active (' + venv_path.name + ')' if in_venv else 'NOT ACTIVE'}")
    if not in_venv:
        print("[WARN] You are running global Python, not the .venv virtual environment.")
        print("       Use: .\\.venv\\Scripts\\python.exe check_environment.py")
        return False
    return True


def check_directories() -> bool:
    """Verify required project directories exist."""
    required_dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        BASE_DIR / "notebooks",
        BASE_DIR / "src",
        BASE_DIR / "api",
        BASE_DIR / "app",
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        BASE_DIR / "tests",
    ]
    all_exist = True
    print("\n--- Checking Project Directories ---")
    for directory in required_dirs:
        exists = directory.exists() and directory.is_dir()
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status:10} {directory.relative_to(BASE_DIR)}")
        if not exists:
            all_exist = False
    return all_exist


def check_packages() -> bool:
    """Verify core Stage 1 & 2 Python packages are installed and importable."""
    required_packages = [
        "pandas",
        "numpy",
        "openpyxl",
        "dotenv",
        "jupyter",
        "pytest",
    ]
    all_installed = True
    print("\n--- Checking Core Packages ---")
    for pkg in required_packages:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, "__version__", "installed")
            print(f"[OK]       {pkg:15} (version: {version})")
        except ImportError:
            print(f"[MISSING]  {pkg:15} -> Run: pip install {pkg}")
            all_installed = False
    return all_installed


def main() -> None:
    print("=" * 60)
    print("  CustomerPulse AI - Stage 1 Environment Check")
    print("=" * 60)

    py_ok = check_python_version()
    venv_ok = check_virtual_env()
    dirs_ok = check_directories()
    pkgs_ok = check_packages()

    print("\n" + "=" * 60)
    if all([py_ok, venv_ok, dirs_ok, pkgs_ok]):
        print("[SUCCESS] Stage 1 Environment Setup is complete and verified!")
        print("You are ready to proceed to Stage 2: Dataset Acquisition.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("[ERROR] Environment verification failed. Please review the items above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
