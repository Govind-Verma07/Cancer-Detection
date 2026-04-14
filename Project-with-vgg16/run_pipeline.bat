@echo off
echo --- 1. Verifying User Data ---
python check_user_data.py
if %ERRORLEVEL% NEQ 0 (
    echo Error verifying user data. Check Config.IMAGE_DIR and Config.MASK_DIR.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo --- 2. Running Verification Script (VGG16) ---
python test_vgg16_model.py
if %ERRORLEVEL% NEQ 0 (
    echo Model verification failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo --- 3. Starting Training Pipeline ---
echo This will run for 20 epochs...
python src/train.py
if %ERRORLEVEL% NEQ 0 (
    echo Training failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo --- Pipeline Completed Successfully! ---
pause
