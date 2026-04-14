# Procedure: Breast Cancer AI Diagnostic Suite

This document provides step-by-step instructions for running the frontend, training the models, and a high-level overview of the system architecture.

## 1. How to Run the Frontend

The frontend is a premium Streamlit-based dashboard designed for medical image analysis.

### Option A: Using the Batch File (Fastest)
Double-click the `run_streamlit.bat` file in the project root.

### Option B: Using the Terminal
1. Open a terminal in the project root directory.
2. Run the following command:
   ```bash
   python -m streamlit run ui/app.py
   ```
3. The dashboard will automatically open in your default browser (usually at `http://localhost:8501`).

---

## 2. How to Train the Model

The training pipeline optimizes both the segmentation (U-Net) and classification (VGG16) models.

1. Ensure your dataset is correctly placed (default path is `D:\dataset for dl\Dataset DL\TIFF Images\TIFF Images`).
2. Open a terminal in the project root.
3. Run the training script:
   ```bash
   python src/train.py
   ```
4. The models will be trained for 20 epochs (as currently configured) and saved to the `models/` directory.

---

## 3. How to Apply/Test on a Foreign Image

1. Start the **Frontend** using the instructions in Section 1.
2. On the dashboard, use the **File Uploader** to select a "foreign" image (TIFF, PNG, or JPG format).
3. The system will automatically:
   - **Initialize Neural Networks**: Load the U-Net and VGG16 models.
   - **Process Patches**: Split the high-resolution image into manageable patches.
   - **Generate Findings**: Perform tumor localization and classification.
4. **Review Results**:
   - Check the **Automated Contour Detection** image (Red = Malignant, Green = Benign).
   - Review **Diagnostic Parameters** for tumor presence, staging, and coverage ratio.
5. **Online Learning**: You can provide feedback ("Correct", "False Positive", etc.) to perform a micro-training step and refine the AI's accuracy for future scans.

---

## 4. Architecture & Workflow

### Simplest Architecture Description
The system uses a **Dual-Stage Deep Learning Pipeline**:
1. **Segmentation Layer (U-Net)**: Identifies the exact pixels where a tumor is located.
2. **Classification Layer (VGG16)**: A multi-head VGG16 model that takes the identified regions and predicts:
   - **Binary Status**: Malignant vs. Benign.
   - **Clinical Stage**: Estimates the cancer stage (1-4).

### Automated Workflow
1. **Input**: High-resolution medical image.
2. **Patching**: Image is split into 512x512 patches to maintain high-detail analysis.
3. **Segmentation**: U-Net generates a tumor probability mask for each patch.
4. **Classification**: VGG16 analyzes detected regions for malignancy and staging.
5. **Reconstruction**: Patches are combined back into a full-size diagnostic map.
6. **Output**: Interactive report with highlighted tumor contours and clinical metrics.
