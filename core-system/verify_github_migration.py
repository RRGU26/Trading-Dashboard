#!/usr/bin/env python3
"""
Verify the GitHub migration status and identify missing reports
"""
import os
import glob
from trading_reports_parsers import find_report_files, parse_all_reports

def check_github_structure():
    """Check the GitHub repo structure"""
    print("=== GITHUB REPOSITORY STRUCTURE CHECK ===")
    
    # Check both potential GitHub locations
    github_locations = [
        r"C:\GitHub\Trading-Models",
        r"C:\Users\rrose\trading-models-system\core-system\reports",  # Current repo location
        r"C:\Users\rrose\trading-models-system\reports",
    ]
    
    for location in github_locations:
        print(f"\nLocation: {location}")
        if os.path.exists(location):
            print("  EXISTS")
            files = []
            for pattern in ["*.txt", "*Report*"]:
                files.extend(glob.glob(os.path.join(location, pattern)))
            print(f"  Report files: {len(files)}")
            if files:
                for f in files[:5]:  # Show first 5
                    print(f"    {os.path.basename(f)}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")
        else:
            print("  DOES NOT EXIST")

def check_desktop_reports():
    """Check what reports are still on Desktop"""
    print("\n=== DESKTOP REPORTS CHECK ===")
    
    desktop_locations = [
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/OneDrive/Desktop"),
        os.path.expanduser("~/OneDrive/Desktop/Trading System"),
    ]
    
    total_desktop_reports = 0
    
    for location in desktop_locations:
        if os.path.exists(location):
            report_files = glob.glob(os.path.join(location, "*Report*.txt"))
            if report_files:
                print(f"\n{location}: {len(report_files)} reports")
                total_desktop_reports += len(report_files)
                
                # Group by type
                report_types = {}
                for f in report_files:
                    filename = os.path.basename(f)
                    if "Bitcoin" in filename:
                        report_types.setdefault("Bitcoin", []).append(filename)
                    elif "NVIDIA" in filename:
                        report_types.setdefault("NVIDIA", []).append(filename)
                    elif "Algorand" in filename:
                        report_types.setdefault("Algorand", []).append(filename)
                    elif "QQQ" in filename:
                        report_types.setdefault("QQQ", []).append(filename)
                    else:
                        report_types.setdefault("Other", []).append(filename)
                
                for report_type, files in report_types.items():
                    print(f"    {report_type}: {len(files)} files")
                    # Show most recent
                    files.sort(reverse=True)
                    if files:
                        print(f"      Latest: {files[0]}")
    
    print(f"\nTOTAL DESKTOP REPORTS: {total_desktop_reports}")
    return total_desktop_reports

def check_current_parser_results():
    """Check what the current parser is finding"""
    print("\n=== CURRENT PARSER RESULTS ===")
    
    results = parse_all_reports(latest_only=True)
    
    print(f"Parser found {len(results)} report types:")
    
    for report_type, data in results.items():
        if data and 'file_path' in data:
            filepath = data['file_path']
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            
            # Determine location type
            if "trading-models-system" in directory:
                location_type = "GITHUB REPO"
            elif "OneDrive" in directory:
                location_type = "ONEDRIVE DESKTOP"
            elif "Desktop" in directory:
                location_type = "DESKTOP"
            else:
                location_type = "OTHER"
                
            print(f"  {report_type.upper()}: {location_type}")
            print(f"    File: {filename}")
            print(f"    Path: {directory}")
        else:
            print(f"  {report_type.upper()}: NO DATA")

def check_missing_models():
    """Check for missing model types"""
    print("\n=== MISSING MODELS CHECK ===")
    
    expected_models = [
        "algorand",
        "nvidia", 
        "bitcoin",
        "wishing_wealth",  # QQQ Trading Signal
        "trading_signal",
        "longhorn"
    ]
    
    results = parse_all_reports(latest_only=True)
    found_models = list(results.keys())
    missing_models = [m for m in expected_models if m not in found_models]
    
    print(f"Expected models: {len(expected_models)}")
    print(f"Found models: {len(found_models)}")
    print(f"Missing models: {missing_models}")
    
    if missing_models:
        print("\nSearching for missing model reports...")
        for model in missing_models:
            if model == "algorand":
                pattern = "*Algorand*Report*.txt"
            elif model == "nvidia":
                pattern = "*NVIDIA*Report*.txt"
            
            # Search all locations
            all_locations = [
                r"C:\Users\rrose\trading-models-system\core-system\reports",
                r"C:\Users\rrose\OneDrive\Desktop\Trading System",
                r"C:\Users\rrose\Desktop",
                r"C:\GitHub\Trading-Models",
            ]
            
            for location in all_locations:
                if os.path.exists(location):
                    files = glob.glob(os.path.join(location, pattern))
                    if files:
                        print(f"  {model}: Found {len(files)} files in {location}")
                        # Show latest
                        files.sort(reverse=True)
                        if files:
                            print(f"    Latest: {os.path.basename(files[0])}")

if __name__ == "__main__":
    check_github_structure()
    desktop_count = check_desktop_reports()
    check_current_parser_results()
    check_missing_models()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"- Desktop reports still exist: {desktop_count > 0}")
    print(f"- GitHub repo prioritization: {'Working' if desktop_count == 0 else 'NEEDS ATTENTION'}")
    print("- Check above for missing models and their locations")