import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_pipeline

# Redirect stdout to both console and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

log_file = os.path.join(os.getcwd(), "train_log_v2.txt")
sys.stdout = Logger(log_file)
sys.stderr = Logger(log_file)

try:
    print("Log started for train_pipeline")
    train_pipeline(num_train=10, num_test=10, epochs=2)
    print("Train pipeline finished successfully.")
except Exception as e:
    print(f"FAILED with error: {e}")
finally:
    sys.stdout.log.close()
