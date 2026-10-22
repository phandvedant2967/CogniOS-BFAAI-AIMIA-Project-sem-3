import importlib.util
import os
import sys

from app import create_app

app = create_app()

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print("[CogniOS] DEMO LOGIN: usernames 'clinician' and 'manufacturer'; both use the fixed demo password set in config.py. Not for public deployment.")
        # Say at start-up which Python runs the app and whether the model packages are importable there (model predictions need them).
        missing = [m for m in ("xgboost", "torch", "sklearn", "joblib") if importlib.util.find_spec(m) is None]
        print(f"[CogniOS] Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor})")
        if missing:
            print(f"[CogniOS] WARNING: missing in this environment: {', '.join(missing)}. Model predictions will fail until you run: "
                  f"\"{sys.executable}\" -m pip install -r requirements.txt  (Windows: setup_windows.bat, then start_windows.bat).")
    # The Werkzeug debugger is a code-execution console; enable it only on purpose (FLASK_DEBUG=1).
    app.run(host="127.0.0.1", port=int(os.environ.get("COGNIOS_PORT", "5000")), debug=os.environ.get("FLASK_DEBUG") == "1")
