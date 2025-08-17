# Create Daily Health Check Email scheduled task for 9:00 AM
$taskName = "DailyHealthCheckEmail"
$scriptPath = "C:\Users\rrose\trading-models-system\core-system\run_health_check_email.bat"
$time = "09:00"  # 9:00 AM
$description = "Send daily health check email at 9:00 AM"

# Create the action
$action = New-ScheduledTaskAction -Execute $scriptPath

# Create the trigger (daily at 9:00 AM)
$trigger = New-ScheduledTaskTrigger -Daily -At $time

# Create the settings (allow long running tasks)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

# Register the task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description $description

Write-Host "Successfully created scheduled task: $taskName"
Write-Host "The health check email will be sent daily at 9:00 AM"
Write-Host "Maximum execution time: 30 minutes"