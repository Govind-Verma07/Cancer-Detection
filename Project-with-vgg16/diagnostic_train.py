import sys
import os

# Add project root
sys.path.append(os.getcwd())

try:
    from src.train import train_pipeline
    print("Import successful. Starting pipeline...")
    train_pipeline(num_train=15, num_test=5, epochs=20)
    print("Pipeline finished successfully.")
except Exception as e:
    print(f"FAILED with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except BaseException as e:
    print(f"FAILED with base error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
