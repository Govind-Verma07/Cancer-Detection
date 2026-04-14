@echo off
echo 🚀 STARTING PUSH... > push_log.txt
del "C:\Users\Govind Verma\.git\index.lock" /F /Q 2>> push_log.txt
git init >> push_log.txt 2>&1
git remote add origin https://github.com/Govind-Verma07/Cancer-Detection.git 2>> push_log.txt
git remote set-url origin https://github.com/Govind-Verma07/Cancer-Detection.git >> push_log.txt 2>&1
git add . >> push_log.txt 2>&1
git commit -m "Final submission: Optimized tumor analysis & refinement pipeline" >> push_log.txt 2>&1
git branch -M main >> push_log.txt 2>&1
git push -u origin main -f >> push_log.txt 2>&1
echo ✅ FINISHED >> push_log.txt
