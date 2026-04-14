@echo off
echo --- Installing Cancer-Detection Environment Dependencies ---
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo --- Verifying Installation ---
python -c "import torch; print(f'Torch version: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
echo.
echo Installation complete! You can now run the training or inference scripts.
pause
