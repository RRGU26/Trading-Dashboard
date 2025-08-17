# Daily Health Check Runner
# This script runs the health report and saves output with timestamp

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$reportPath = "C:\Users\rrose\trading-models-system\core-system\reports\health_report_$timestamp.log"
$scriptPath = "C:\Users\rrose\trading-models-system\core-system\daily_health_report.py"

Write-Host "Running Trading System Health Report..."
Write-Host "Report will be saved to: $reportPath"

# Run the health report and capture output
try {
    Set-Location "C:\Users\rrose\trading-models-system\core-system"
    python daily_health_report.py | Tee-Object -FilePath $reportPath
    
    Write-Host ""
    Write-Host "Health report completed successfully!"
    Write-Host "Report saved to: $reportPath"
} catch {
    Write-Host "Error running health report: $_"
}

# Keep window open if running manually
if ($Host.Name -eq "ConsoleHost") {
    Write-Host ""
    Write-Host "Press any key to continue..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}