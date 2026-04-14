import subprocess
import os

def check():
    print(f"Current Directory: {os.getcwd()}")
    print(f"Files: {os.listdir('.')}")
    
    try:
        with open("write_test.txt", "w") as f:
            f.write("I can write!")
        print("Write test: SUCCESS")
    except Exception as e:
        print(f"Write test: FAILURE ({e})")
        
    try:
        res = subprocess.run(["git", "--version"], capture_output=True, text=True)
        print(f"Git Version: {res.stdout.strip()} (Err: {res.stderr.strip()})")
    except Exception as e:
        print(f"Git test: FAILURE ({e})")
        
if __name__ == "__main__":
    check()
