# Breast Cancer Tumor Analysis Pipeline

This project implements a production-ready deep learning pipeline for breast cancer tumor analysis using high-resolution medical images.

## 🚀 Features
- **Patch-based Processing**: Handles high-resolution images without quality loss.
- **Tumor Segmentation**: U-Net model for precise tumor boundary detection.
- **Multi-head Classification**: Predicts both malignancy (Benign/Malignant) and Cancer Stage (1-4).
- **Metric Extraction**: calculates tumor area, breast-to-tumor ratio, and relative coordinates.
- **Visual Overlay**: Generates images with contour overlays (Red for Malignant, Green for Benign).
- **Streamlit UI**: A user-friendly interface for image uploads and analysis results.

## 📂 Structure
- `src/`: Core logic (patching, preprocessing, models, metrics, visualization).
- `ui/`: Streamlit web application.
- `utils/`: Configuration and helper functions.
- `data/`: Raw and processed image data.
- `models/`: Trained model weights (.pth files).

## 🛠️ Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Generate dummy weights for testing the pipeline:
   ```bash
   python save_dummy_weights.py
   ```
3. Generate a mock medical image for testing:
   ```bash
   python create_mock_data.py
   ```

## 🖥️ Usage

### Inference Script
Run analysis on a single image via CLI:
```bash
python src/inference.py --image data/raw/test_sample.jpg
```

### Streamlit UI
Launch the interactive dashboard:
```bash
streamlit run ui/app.py
```

## 🧪 Technical Details
- **Segmentation**: U-Net architecture with Sigmoid activation.
- **Classification**: VGG16 backbone with two heads (Binary & 4-Class).
- **Contour Detection**: OpenCV `findContours` on segmentation masks.
- **Normalization**: ImageNet statistics for consistent model performance.


Run :- python -m streamlit run ui/app.py