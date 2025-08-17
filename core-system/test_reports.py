#!/usr/bin/env python3
"""
Test script for trading reports parsing
"""
import sys
import os
from trading_reports_parsers import parse_all_reports, find_report_files

def test_report_discovery():
    """Test what reports are being found"""
    print("TESTING REPORT DISCOVERY")
    print("=" * 50)
    
    # Get the script directory (core-system)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define search directories in priority order
    directories = [
        os.path.join(script_dir, "reports"),  # GitHub repo reports directory (PRIORITY)
        script_dir,  # Core system directory itself
        os.path.expanduser("~/OneDrive/Desktop"),  # Legacy location
        os.path.expanduser("~/Desktop")  # Legacy location
    ]
    
    print("Search directories:")
    for i, directory in enumerate(directories, 1):
        exists = os.path.exists(directory)
        print(f"  {i}. {directory} {'(EXISTS)' if exists else '(MISSING)'}")
    
    print("\nFinding report files...")
    all_files = find_report_files(directories)
    
    print("\nREPORT FILES FOUND:")
    total_files = 0
    for report_type, files in all_files.items():
        print(f"\n{report_type.upper()}:")
        if files:
            for i, filepath in enumerate(files, 1):
                file_dir = os.path.dirname(filepath)
                filename = os.path.basename(filepath)
                print(f"  {i}. {filename} in {file_dir}")
                total_files += 1
        else:
            print("  No files found")
    
    print(f"\nTOTAL FILES FOUND: {total_files}")
    return all_files

def test_report_parsing():
    """Test parsing of found reports"""
    print("\n" + "=" * 50)
    print("TESTING REPORT PARSING")
    print("=" * 50)
    
    results = parse_all_reports(latest_only=True)
    
    print(f"\nPARSING RESULTS:")
    successful_parses = 0
    
    for report_type, data in results.items():
        if data:
            meaningful_fields = len([v for k,v in data.items() if v is not None and v != ""])
            print(f"\n{report_type.upper()}: SUCCESS")
            print(f"  Fields extracted: {meaningful_fields}")
            
            # Show key fields
            key_fields = ['current_price', 'signal', 'confidence', 'suggested_action']
            for field in key_fields:
                if field in data and data[field] is not None:
                    print(f"  {field}: {data[field]}")
            successful_parses += 1
        else:
            print(f"\n{report_type.upper()}: FAILED - No data extracted")
    
    print(f"\nSUMMARY:")
    print(f"  Successful parses: {successful_parses}")
    print(f"  Total report types: {len(results)}")
    
    return results

if __name__ == "__main__":
    # Run tests
    file_results = test_report_discovery()
    parse_results = test_report_parsing()