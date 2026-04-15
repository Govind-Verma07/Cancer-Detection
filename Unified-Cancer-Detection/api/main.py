from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import shutil
import math
import threading

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

accuracy_test_job = {
    "running": False,
    "processed": 0,
    "total": 0,
    "current_file": None,
    "message": "Idle",
    "error": None
}
accuracy_test_lock = threading.Lock()


def update_accuracy_test_job(processed=None, total=None, current_file=None, message=None, error=None, running=None):
    with accuracy_test_lock:
        if processed is not None:
            accuracy_test_job["processed"] = processed
        if total is not None:
            accuracy_test_job["total"] = total
        if current_file is not None:
            accuracy_test_job["current_file"] = current_file
        if message is not None:
            accuracy_test_job["message"] = message
        if error is not None:
            accuracy_test_job["error"] = error
        if running is not None:
            accuracy_test_job["running"] = running


def run_accuracy_job(num_images: int):
    try:
        from accuracy_test import run_accuracy_test as run_test

        def status_callback(processed, total, current_file, message=None):
            update_accuracy_test_job(
                processed=processed,
                total=total,
                current_file=current_file,
                message=message or f"Processed {processed}/{total}",
                running=True
            )

        update_accuracy_test_job(processed=0, total=num_images, current_file=None, message="Starting accuracy test...", running=True, error=None)
        run_test(num_images=num_images, status_callback=status_callback)
        update_accuracy_test_job(message="Accuracy test completed.", running=False)
    except Exception as e:
        update_accuracy_test_job(message="Accuracy test failed.", running=False, error=str(e))


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
async def run_accuracy_test_endpoint(num_images: int = 100):
    with accuracy_test_lock:
        if accuracy_test_job["running"]:
            return JSONResponse(status_code=409, content={
                "message": "Accuracy test already running",
                "processed": accuracy_test_job["processed"],
                "total": accuracy_test_job["total"]
            })

        update_accuracy_test_job(processed=0, total=num_images, current_file=None, message="Starting accuracy test...", running=True, error=None)
        accuracy_thread = threading.Thread(target=run_accuracy_job, args=(num_images,), daemon=True)
        accuracy_thread.start()

    return JSONResponse(status_code=202, content={
        "message": "Accuracy test started",
        "processed": 0,
        "total": num_images
    })

@app.get("/api/accuracy_test_status")
async def accuracy_test_status():
    with accuracy_test_lock:
        return sanitize([accuracy_test_job])[0]

@app.get("/api/accuracy_test_history")
async def accuracy_test_history():
    history_path = os.path.join(Config.RESULTS_DIR, "accuracy_test_results.csv")
    try:
        if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
            df = pd.read_csv(history_path)
            records = df.to_dict(orient="records")
            return sanitize(records)
    except Exception as e:
        print(f"Error reading accuracy test history: {e}")
    return []

@app.get("/api/history")
async def history():
    try:
        if os.path.exists(Config.COMPARISON_LOG) and os.path.getsize(Config.COMPARISON_LOG) > 0:
            df = pd.read_csv(Config.COMPARISON_LOG)
            records = df.tail(10).to_dict(orient="records")
            return sanitize(records)
    except Exception as e:
        print(f"Error reading history: {e}")
    return []

@app.get("/api/analytics")
async def analytics():
    results_path = os.path.join(Config.RESULTS_DIR, "accuracy_test_results.csv")
    try:
        if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
            df = pd.read_csv(results_path)
            records = []
            for _, row in df.iterrows():
                records.append({
                    "timestamp": row.get("filename", ""),
                    "filename": row.get("filename", ""),
                    "resnet_accuracy": row.get("resnet_iou", 0),
                    "vgg_accuracy": row.get("vgg_iou", 0),
                    "resnet_dice": row.get("resnet_dice", 0),
                    "vgg_dice": row.get("vgg_dice", 0),
                    "resnet_precision": row.get("resnet_precision", 0),
                    "vgg_precision": row.get("vgg_precision", 0),
                    "resnet_recall": row.get("resnet_recall", 0),
                    "vgg_recall": row.get("vgg_recall", 0)
                })
            return sanitize(records)
    except Exception as e:
        print(f"Error reading analytics from accuracy results: {e}")

    # Fallback to accuracy_log.csv
    log_path = os.path.join(Config.RESULTS_DIR, "accuracy_log.csv")
    try:
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            df = pd.read_csv(log_path)
            records = df.to_dict(orient="records")
            return sanitize(records)
    except Exception as e:
        print(f"Error reading accuracy log: {e}")
    return []

# Mount static files for the frontend
# Order matters: more specific routes first
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/media", StaticFiles(directory="media"), name="media")
app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
