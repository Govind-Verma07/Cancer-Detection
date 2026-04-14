import os
import pandas as pd
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference_unified import run_unified_inference
from utils.config import Config

def compare_models(image_path):
    """
    Runs both ResNet50 and VGG16 on the same image and returns a comparison.
    """
    # 1. Run ResNet50
    resnet_result = run_unified_inference(image_path, model_type="resnet50")
    
    # 2. Run VGG16
    vgg_result = run_unified_inference(image_path, model_type="vgg16")
    
    # 3. Calculate Insights
    resnet_findings = resnet_result.get('findings', [])
    vgg_findings = vgg_result.get('findings', [])
    
    # Simple consensus: Do they agree on tumor presence?
    resnet_present = resnet_result.get('tumor_present', False)
    vgg_present = vgg_result.get('tumor_present', False)
    
    consensus = "Agreement" if resnet_present == vgg_present else "Discrepancy"
    
    # Log to CSV
    log_data = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Image': os.path.basename(image_path),
        'ResNet_Tumors': len(resnet_findings),
        'VGG_Tumors': len(vgg_findings),
        'Consensus': consensus,
        'ResNet_Ratio': f"{resnet_result.get('overall_ratio', 0):.4f}",
        'VGG_Ratio': f"{vgg_result.get('overall_ratio', 0):.4f}"
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
        'log_path': Config.COMPARISON_LOG
    }

if __name__ == "__main__":
    print("Comparison Engine ready. Usage: compare_models(path_to_image)")
