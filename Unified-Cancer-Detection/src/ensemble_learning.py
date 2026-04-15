import os
import time
import pandas as pd
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resnet50.inference import run_resnet50_inference
from vgg16.inference import run_vgg16_inference
from utils.config import Config

def generate_conclusion_report(resnet_result, vgg_result, ensemble_score, status):
    """
    Generate a detailed conclusion report based on model predictions.
    """
    report = []
    report.append("=== Breast Cancer Diagnostic Report ===")
    report.append("")
    
    # Overall Status
    report.append(f"Overall Diagnosis: {status}")
    report.append(f"Ensemble Tumor Burden Score: {ensemble_score:.4f}")
    report.append("")
    
    # Model Details
    if resnet_result:
        res_findings = resnet_result.get('findings', [])
        res_ratio = resnet_result.get('overall_ratio', 0)
        report.append("ResNet50 Model Results:")
        report.append(f"  - Tumor Burden: {res_ratio:.4f}")
        report.append(f"  - Regions Detected: {len(res_findings)}")
        for i, f in enumerate(res_findings):
            report.append(f"    Region {i+1}: {f['classification']} at {f['location_pct']}")
        report.append("")
    
    if vgg_result:
        vgg_findings = vgg_result.get('findings', [])
        vgg_ratio = vgg_result.get('overall_ratio', 0)
        report.append("VGG16 Model Results:")
        report.append(f"  - Tumor Burden: {vgg_ratio:.4f}")
        report.append(f"  - Regions Detected: {len(vgg_findings)}")
        for i, f in enumerate(vgg_findings):
            report.append(f"    Region {i+1}: {f['classification']} at {f['location_pct']}")
        report.append("")
    
    # Recommendations
    if status == "Malignant":
        report.append("Recommendations:")
        report.append("  - Immediate consultation with oncologist recommended.")
        report.append("  - Further imaging (MRI/CT) advised.")
        report.append("  - Biopsy may be required for confirmation.")
    elif status == "Benign":
        report.append("Recommendations:")
        report.append("  - Regular monitoring advised.")
        report.append("  - Follow-up in 6-12 months.")
    else:
        report.append("Recommendations:")
        report.append("  - Additional testing required.")
        report.append("  - Clinical correlation needed.")
    
    report.append("")
    report.append("Note: This is an AI-assisted analysis. Final diagnosis requires clinical expertise.")
    
    return "\n".join(report)

def init_logs():
    # Ensure results dir exists
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Initialize COMPARISON_LOG
    if not os.path.exists(Config.COMPARISON_LOG):
        df = pd.DataFrame(columns=[
            "timestamp", "filename", "resnet_score", "vgg_score", 
            "ensemble_score", "status", "ground_truth"
        ])
        df.to_csv(Config.COMPARISON_LOG, index=False)
        
    # Initialize accuracy_log.csv
    acc_log_path = os.path.join(Config.RESULTS_DIR, "accuracy_log.csv")
    if not os.path.exists(acc_log_path):
        df = pd.DataFrame(columns=[
            "timestamp", "resnet_accuracy", "vgg_accuracy"
        ])
        df.to_csv(acc_log_path, index=False)

def predict_ensemble(image_path, ground_truth=None):
    init_logs()
    
    # 1. Run ResNet
    resnet_result = run_resnet50_inference(image_path, output_dir="results")
    # 2. Run VGG
    vgg_result = run_vgg16_inference(image_path, output_dir="results")
    
    # Defaults in case of failure
    res_burden = resnet_result.get('overall_ratio', 0) if resnet_result else 0
    res_regions = len(resnet_result.get('findings', [])) if resnet_result else 0
    res_img = "/results/prediction_resnet50.jpg" if resnet_result else ""
    res_findings = resnet_result.get('findings', []) if resnet_result else []
    
    vgg_burden = vgg_result.get('overall_ratio', 0) if vgg_result else 0
    vgg_regions = len(vgg_result.get('findings', [])) if vgg_result else 0
    vgg_img = "/results/prediction_vgg16.jpg" if vgg_result else ""
    vgg_findings = vgg_result.get('findings', []) if vgg_result else []
    
    # 3. Calculate Ensemble Score (Average burden)
    ensemble_score = (res_burden + vgg_burden) / 2.0
    
    # Determine Status
    if ensemble_score > 0.05: # > 5% burden
        status = "Malignant"
    elif ensemble_score <= 0.01:
        status = "Benign"
    else:
        status = "Needs Review"
        
    # 4. Generate Conclusion Report
    conclusion = generate_conclusion_report(resnet_result, vgg_result, ensemble_score, status)
    
    # 5. Log to History
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.basename(image_path)
    
    hist_entry = {
        "timestamp": [timestamp],
        "filename": [filename],
        "resnet_score": [res_burden],
        "vgg_score": [vgg_burden],
        "ensemble_score": [ensemble_score],
        "status": [status],
        "ground_truth": [ground_truth if ground_truth is not None else ""]
    }
    pd.DataFrame(hist_entry).to_csv(Config.COMPARISON_LOG, mode='a', header=False, index=False)
    
    # 6. Accuracy Tracking (Ad-Hoc)
    if ground_truth is not None and ground_truth != "":
        acc_log_path = os.path.join(Config.RESULTS_DIR, "accuracy_log.csv")
        try:
            gt_val = int(ground_truth)
            # Binary mock correctness logic for demonstration: 
            # If gt is 1 (Malignant), the prediction is correct if burden > 0.05
            # Accuracy cumulative calculation requires a bit of logic, but we'll do smoothed moving averages here
            df_acc = pd.read_csv(acc_log_path)
            
            # Did they guess correctly?
            res_correct = 1 if (res_burden > 0.05) == gt_val else 0
            vgg_correct = 1 if (vgg_burden > 0.05) == gt_val else 0
            
            # Calculate new cumulative accuracy (mocking historical progression based on previous runs if small)
            total_runs = len(df_acc) + 1
            if total_runs == 1:
                new_res_acc = res_correct
                new_vgg_acc = vgg_correct
            else:
                prev_res_acc = df_acc.iloc[-1]['resnet_accuracy']
                prev_vgg_acc = df_acc.iloc[-1]['vgg_accuracy']
                
                new_res_acc = ((prev_res_acc * len(df_acc)) + res_correct) / total_runs
                new_vgg_acc = ((prev_vgg_acc * len(df_acc)) + vgg_correct) / total_runs
                
            acc_entry = {
                "timestamp": [timestamp],
                "resnet_accuracy": [new_res_acc],
                "vgg_accuracy": [new_vgg_acc]
            }
            pd.DataFrame(acc_entry).to_csv(acc_log_path, mode='a', header=False, index=False)
        except ValueError:
            pass # Invalid ground truth format

    return {
        "resnet": {
            "image_url": f"{res_img}?t={int(time.time())}", # Cache busing for dynamic UI updates
            "tumor_burden": res_burden,
            "regions_detected": res_regions,
            "findings": res_findings
        },
        "vgg": {
            "image_url": f"{vgg_img}?t={int(time.time())}",
            "tumor_burden": vgg_burden,
            "regions_detected": vgg_regions,
            "findings": vgg_findings
        },
        "ensemble": {
            "score": ensemble_score,
            "status": status,
            "conclusion_report": conclusion
        }
    }
