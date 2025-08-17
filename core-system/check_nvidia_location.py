#!/usr/bin/env python3
"""
Check specific NVIDIA file location that was found in test
"""
import os
import glob

# Check the specific file the test claimed to find
nvidia_file = r"C:\Users\rrose\OneDrive\Desktop\NVIDIA_Bull_Momentum_Report_20250808_154249.txt"
print(f"File exists check: {nvidia_file}")
print(f"Result: {os.path.exists(nvidia_file)}")

# Search for all NVIDIA files in OneDrive/Desktop area
search_patterns = [
    r"C:\Users\rrose\OneDrive\Desktop\*NVIDIA*.txt",
    r"C:\Users\rrose\OneDrive\Desktop\Trading System\*NVIDIA*.txt",
    r"C:\Users\rrose\*NVIDIA*154249*",
]

for pattern in search_patterns:
    files = glob.glob(pattern)
    print(f"\nPattern: {pattern}")
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f}")

# Also check what the find_report_files function is actually finding
from trading_reports_parsers import find_report_files
import os

directories = [
    os.path.expanduser("~/OneDrive/Desktop"),
    os.path.expanduser("~/Desktop")
]

print(f"\n=== Using find_report_files function ===")
all_files = find_report_files(directories)

nvidia_files = all_files.get('nvidia', [])
print(f"NVIDIA files found by parser: {len(nvidia_files)}")
for f in nvidia_files:
    print(f"  {f}")
    print(f"    Exists: {os.path.exists(f)}")
    print(f"    Size: {os.path.getsize(f) if os.path.exists(f) else 'N/A'}")