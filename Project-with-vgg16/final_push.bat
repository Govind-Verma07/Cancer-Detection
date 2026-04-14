@echo off
echo 🚀 RESOLVING GIT CONFLICTS...
del "C:\Users\Govind Verma\.git\index.lock" /F /Q 2>nul
git init
git remote add origin https://github.com/Govind-Verma07/Cancer-Detection.git 2>nul
git remote set-url origin https://github.com/Govind-Verma07/Cancer-Detection.git
echo 📦 STAGING...
git add .
echo 💾 COMMITTING...
git commit -m "Integrated optimized tumor detection, refined contours (1px), staging heuristics, and online learning modules."
echo 🚀 PUSHING TO MAIN...
git branch -M main
git push -u origin main -f
echo ✅ DONE.
