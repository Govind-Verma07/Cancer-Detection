from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import shutil
import math

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ensemble_learning import predict_ensemble
from utils.config import Config
import pandas as pd

app = FastAPI(title="Unified Cancer Diagnostics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize(records):
    """Replace NaN/Inf values with None so they JSON-serialize cleanly."""
    cleaned = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            else:
                clean_row[k] = v
        cleaned.append(clean_row)
    return cleaned

# API Routes
@app.post("/api/predict")
async def predict(file: UploadFile = File(...), ground_truth: int = Form(None)):
    os.makedirs("results", exist_ok=True)
    temp_path = os.path.join("results", "api_upload.jpg")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run Ensemble Prediction
        result = predict_ensemble(temp_path, ground_truth=ground_truth)
        
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/run_accuracy_test")
async def run_accuracy_test(num_images: int = 100):
    try:
        # Import here to avoid circular imports
        from accuracy_test import run_accuracy_test as run_test
        run_test(num_images=num_images)
        return JSONResponse(status_code=200, content={"message": "Accuracy test completed"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/history")
async def history():
    if os.path.exists(Config.COMPARISON_LOG):
        df = pd.read_csv(Config.COMPARISON_LOG)
        records = df.tail(10).to_dict(orient="records")
        return sanitize(records)
    return []

@app.get("/api/analytics")
async def analytics():
    log_path = os.path.join(Config.RESULTS_DIR, "accuracy_log.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        records = df.to_dict(orient="records")
        return sanitize(records)
    return []

# Mount static files for the frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/media", StaticFiles(directory="media"), name="media")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
