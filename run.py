"""
run.py

Simple launcher to run the Streamlit app.
Usage:
    python run.py
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    # Get the directory where this script is located
    base_dir = Path(__file__).resolve().parent
    
    # Activate venv if it exists
    venv_python = base_dir / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python_exe = str(venv_python)
    else:
        python_exe = sys.executable

    # Check if required packages are installed
    try:
        import streamlit
        print("✅ Streamlit is installed.")
    except ImportError:
        print("❌ Streamlit is not installed. Installing dependencies...")
        subprocess.run(
            [python_exe, "-m", "pip", "install", "-r", str(base_dir / "requirements.txt")],
            check=True,
        )
        print("✅ Dependencies installed!")

    # Check if model exists
    model_path = base_dir / "model.pkl"
    
    if not model_path.exists():
        print("⚠️  Model file not found. Training model first...")
        subprocess.run([python_exe, str(base_dir / "train_model.py")], check=True)
        print("✅ Model trained successfully!")

    # Run Streamlit app
    print("\n" + "=" * 60)
    print("🚀 Starting Vietnam Housing Price Predictor...")
    print("📊 Opening Streamlit app in your browser...")
    print("=" * 60 + "\n")
    
    subprocess.run(
        [python_exe, "-m", "streamlit", "run", str(base_dir / "app.py")]
    )


if __name__ == "__main__":
    main()

