import os
import pandas as pd
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ensemble_learning import generate_conclusion_report

def compare_models(image_path):
    """
    Runs both ResNet50 and VGG16 on the same image and returns a comparison.
    """
    # 1. Run ResNet50
    resnet_result = run_unified_inference(image_path, model_type="resnet50")
    
    # 2. Run VGG16
    vgg_result = run_unified_inference(image_path, model_type="vgg16")
    
    # Calculate Insights
    resnet_findings = resnet_result.get('findings', [])
    vgg_findings = vgg_result.get('findings', [])
    
    # Simple consensus: Do they agree on tumor presence?
    resnet_present = resnet_result.get('tumor_present', False)
    vgg_present = vgg_result.get('tumor_present', False)
    
    consensus = "Agreement" if resnet_present == vgg_present else "Discrepancy"
    
    # Calculate ensemble score
    resnet_ratio = resnet_result.get('overall_ratio', 0)
    vgg_ratio = vgg_result.get('overall_ratio', 0)
    ensemble_score = (resnet_ratio + vgg_ratio) / 2.0
    
    # Determine status
    if ensemble_score > 0.05:
        status = "Malignant"
    elif ensemble_score <= 0.01:
        status = "Benign"
    else:
        status = "Needs Review"
    
    # Generate conclusion report
    conclusion = generate_conclusion_report(resnet_result, vgg_result, ensemble_score, status)
    
    # Log to CSV
    log_data = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Image': os.path.basename(image_path),
        'ResNet_Tumors': len(resnet_findings),
        'VGG_Tumors': len(vgg_findings),
        'Consensus': consensus,
        'ResNet_Ratio': f"{resnet_ratio:.4f}",
        'VGG_Ratio': f"{vgg_ratio:.4f}"
    }
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame([log_data])
    
    if not os.path.isfile(Config.COMPARISON_LOG):
        df.to_csv(Config.COMPARISON_LOG, index=False)
    else:
        df.to_csv(Config.COMPARISON_LOG, mode='a', header=False, index=False)
        
    return {
        'resnet': resnet_result,
        'vgg': vgg_result,
        'consensus': consensus,
        'ensemble': {
            'score': ensemble_score,
            'status': status,
            'conclusion_report': conclusion
        },
        'log_path': Config.COMPARISON_LOG
    }

if __name__ == "__main__":
    print("Comparison Engine ready. Usage: compare_models(path_to_image)")
