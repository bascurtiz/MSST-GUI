# setup_task.ps1 - Creates a scheduled task to auto-deploy the site
# Run this once as Administrator, or manually create the task in Task Scheduler

$TaskName = "GDocSiteDeploy"
$ScriptPath = "D:\Downloads\msst_gui_neo\doc-site\deploy.bat"
$Description = "Auto-regenerate and deploy Google Doc mirror site"

# Remove existing task if any
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Create the task action
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$ScriptPath`""

# Create trigger - every 6 hours (adjust as needed)
# Options: Once, Daily, Weekly, AtLogOn, AtStartup
$Trigger = New-ScheduledTaskTrigger -Daily -At "3:00AM"
# For every 6 hours, use this instead:
# $Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6)

# Create task settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register the task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description $Description

Write-Host "Task '$TaskName' created successfully!"
Write-Host "Schedule: Daily at 3:00 AM"
Write-Host ""
Write-Host "To modify:"
Write-Host "  - Open Task Scheduler (taskschd.msc)"
Write-Host "  - Find '$TaskName' in Task Scheduler Library"
Write-Host "  - Right-click > Properties to edit schedule"
Write-Host ""
Write-Host "To run manually:"
Write-Host "  schtasks /run /tn `"$TaskName`""
Write-Host "  or double-click deploy.bat"
