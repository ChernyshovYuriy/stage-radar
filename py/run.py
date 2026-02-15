#!/usr/bin/env python3
"""
run.py — one-command launcher for the Market Stage Dashboard

Usage:
    python run.py

Then open http://localhost:8000 in your browser.
"""
import subprocess, sys, os

# Check deps
required = ["fastapi", "uvicorn", "yfinance", "pandas", "numpy", "tabulate"]
missing  = []
for pkg in required:
    try: __import__(pkg.replace("-","_"))
    except ImportError: missing.append(pkg)

if missing:
    print(f"Installing missing packages: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

print("\n" + "="*55)
print("  Market Stage Dashboard")
print("  Open → http://localhost:8000")
print("  Stop → Ctrl+C")
print("="*55 + "\n")

os.execlp(sys.executable, sys.executable, "server.py")
