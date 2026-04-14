from pathlib import Path
import runpy


if __name__ == "__main__":
    ui_app = Path(__file__).resolve().parent / "ui" / "app.py"
    runpy.run_path(str(ui_app), run_name="__main__")
