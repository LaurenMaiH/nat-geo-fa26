# ─────────────────────────────────────────────────────────────────────────────
# Paste and run this entire file as a single Colab cell.
# Make sure your runtime is set to T4 GPU before running.
# ─────────────────────────────────────────────────────────────────────────────

# 1. Mount Google Drive (so the app can read/write videos)
from google.colab import drive
drive.mount('/content/drive')

# 2. Copy the app file from Drive to Colab local disk
import shutil, os

DRIVE_APP_FOLDER = '/content/drive/My Drive/nat-geo/seaanimals_proto'
shutil.copy(f'{DRIVE_APP_FOLDER}/model_runner_colab.py', '/content/model_runner_colab.py')

# Optional: copy the banner image too
banner_src = f'{DRIVE_APP_FOLDER}/natgeobanner.png'
if os.path.exists(banner_src):
    shutil.copy(banner_src, '/content/natgeobanner.png')

# 3. Install dependencies
os.system('pip install -q streamlit pyngrok ultralytics supervision lapjv opencv-python-headless')
os.system('yolo settings sync=False')

# 4. Launch Streamlit in background
import threading, subprocess
def run_streamlit():
    subprocess.run([
        'streamlit', 'run', '/content/model_runner_colab.py',
        '--server.port', '8501',
        '--server.headless', 'true',
        '--server.maxUploadSize', '2000',   # allow up to 2 GB uploads
    ])

threading.Thread(target=run_streamlit, daemon=True).start()

# 5. Expose via ngrok and print the public URL
from pyngrok import ngrok
import time
time.sleep(3)  # wait for streamlit to start

public_url = ngrok.connect(8501)
print('=' * 60)
print(f'  Streamlit is live at: {public_url}')
print('=' * 60)
