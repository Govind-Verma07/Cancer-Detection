import subprocess
import sys

def run():
    print("Running refine_and_test...")
    result = subprocess.run(
        [sys.executable, "refine_and_test.py"],
        capture_output=True,
        text=True
    )
    with open("python_output.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    print("Done. Wrote python_output.txt")

if __name__ == "__main__":
    run()
