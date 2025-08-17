# Create Trading Models Wrapper scheduled task for 3:40 PM
$taskName = "TradingModelsWrapper"
$scriptPath = "C:\Users\rrose\trading-models-system\core-system\run_wrapper.bat"
$time = "15:40"  # 3:40 PM in 24-hour format
$description = "Run Trading Models Wrapper daily at 3:40 PM"

# Create the action
$action = New-ScheduledTaskAction -Execute $scriptPath

# Create the trigger (daily at 3:40 PM)
$trigger = New-ScheduledTaskTrigger -Daily -At $time

# Create the settings (allow long running tasks)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3)

# Register the task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description $description

Write-Host "Successfully created scheduled task: $taskName"
Write-Host "The task will run daily at 3:40 PM"
Write-Host "Maximum execution time: 3 hours"