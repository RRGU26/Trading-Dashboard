#!/usr/bin/env python3
"""
Debug the exact search paths being used by the parser
"""
import os
from trading_reports_parsers import find_report_files

# Replicate the exact directory logic from parse_all_reports
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# UPDATED: Prioritize GitHub repo over Desktop locations
directories = [
    os.path.join(script_dir, "reports"),  # GitHub repo reports directory (PRIORITY)
    script_dir,  # Core system directory itself
    os.path.expanduser("~/OneDrive/Desktop"),  # Legacy location
    os.path.expanduser("~/Desktop")  # Legacy location
]

print("\nAll directories (before filtering):")
for i, d in enumerate(directories, 1):
    exists = os.path.exists(d)
    print(f"  {i}. {d} - EXISTS: {exists}")

# Remove duplicates and non-existent directories
directories = [d for d in set(directories) if os.path.exists(d)]
print("\nFiltered directories (exists only):")
for i, d in enumerate(directories, 1):
    print(f"  {i}. {d}")

print(f"\n=== NVIDIA Search Results ===")
all_files = find_report_files(directories)

nvidia_files = all_files.get('nvidia', [])
print(f"NVIDIA files found: {len(nvidia_files)}")

for f in nvidia_files:
    print(f"\nFile: {f}")
    print(f"  Directory: {os.path.dirname(f)}")
    print(f"  Filename: {os.path.basename(f)}")
    print(f"  Exists: {os.path.exists(f)}")
    if os.path.exists(f):
        print(f"  Size: {os.path.getsize(f)} bytes")
        print(f"  Modified: {os.path.getmtime(f)}")

# Also test the specific glob pattern for NVIDIA
import glob
nvidia_pattern = 'NVIDIA_Bull_Momentum_Report_*.txt'

print(f"\n=== Testing glob pattern: {nvidia_pattern} ===")
for directory in directories:
    search_path = os.path.join(directory, nvidia_pattern)
    files = glob.glob(search_path)
    print(f"\nDirectory: {directory}")
    print(f"Pattern: {search_path}")
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f}")
        
# Also check if there's a path difference
print(f"\n=== Path Expansions ===")
print(f"~/OneDrive/Desktop expands to: {os.path.expanduser('~/OneDrive/Desktop')}")
print(f"~/Desktop expands to: {os.path.expanduser('~/Desktop')}")

# Check what happens with forward vs backward slashes
alt_paths = [
    "C:/Users/rrose/OneDrive/Desktop",
    "C:\\Users\\rrose\\OneDrive\\Desktop",
]

for alt_path in alt_paths:
    exists = os.path.exists(alt_path)
    print(f"{alt_path} exists: {exists}")
    if exists:
        search_path = os.path.join(alt_path, nvidia_pattern)
        files = glob.glob(search_path)
        print(f"  NVIDIA files: {len(files)}")