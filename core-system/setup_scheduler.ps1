# PowerShell script to create scheduled task for Trading Models Wrapper

$taskName = "TradingModelsWrapper"
$taskPath = "\Trading\"
$scriptPath = "C:\Users\rrose\trading-models-system\core-system\run_wrapper.bat"
$time = "03:40"
$description = "Run Trading Models Wrapper daily at 3:40 AM"

# Remove existing task if it exists
try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed existing task: $taskName"
} catch {
    Write-Host "No existing task found"
}

# Create the action
$action = New-ScheduledTaskAction -Execute $scriptPath

# Create the trigger (daily at 3:40 AM)
$trigger = New-ScheduledTaskTrigger -Daily -At $time

# Create the principal (run with highest privileges)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest

# Create the settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Register the task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description $description -TaskPath $taskPath

Write-Host "Successfully created scheduled task: $taskName"
Write-Host "The task will run daily at $time"
Write-Host ""
Write-Host "To verify the task, run: Get-ScheduledTask -TaskName $taskName"
Write-Host "To run the task immediately for testing: Start-ScheduledTask -TaskName $taskName"