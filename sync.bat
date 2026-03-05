@echo off
git add .
set /p msg="Enter Commit Message: "
git commit -S -m "%msg%"
echo.
echo 🚀 Syncing to Mirza-sufyan-baig and shehbaz0101...
git push all main
pause
