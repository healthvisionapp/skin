import os
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string
import requests

# TensorFlow/Keras
from tensorflow.keras.models import load_model

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# ---------------- Firebase Setup (via env secret) ----------------
# Paste your serviceAccountKey.json contents into a Koyeb Secret named SA_JSON
db = None
sa_json = os.getenv("SA_JSON")
if sa_json:
    try:
        cred = credentials.Certificate(json.loads(sa_json))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized from SA_JSON")
    except Exception as e:
        print("❌ Firebase init error:", e)
else:
    print("⚠️ SA_JSON not found. Firebase not initialized.")

# ---------------- Model Setup (Dropbox download once, then reuse) ----------------
MODEL_URL = (
    "https://www.dropbox.com/scl/fi/zq3rd08qztt52sad61m30/"
    "skin_type_classifier.h5?rlkey=wagg56ok83eu8d1ay25o1g3pr&st=7tzintp0&dl=1"
)
MODEL_PATH = "skin_type_classifier.h5"
labels = ['Dry', 'Normal', 'Oily']
IMG_SIZE = (224, 224)
_model = None  # lazy-load container

def ensure_model_file(path=MODEL_PATH, url=MODEL_URL):
    """Download the .h5 once if missing."""
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return path
    print("📥 Downloading model from Dropbox...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    print("✅ Model downloaded:", path)
    return path

def get_model():
    """Load the Keras model on first use (keeps startup light)."""
    global _model
    if _model is not None:
        return _model
    ensure_model_file()
    _model = load_model(MODEL_PATH)
    print("✅ Model loaded into memory")
    return _model

# ---------------- HTML (same style you wrote) ----------------
HTML_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title> Skin Type Detector</title>
  <style>
    body {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(135deg, #e6f0f2, #f8f8f8);
      color: #333;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
      margin: 0;
      padding: 40px 15px;
      overflow: hidden;
    }
    .container {
      background: #fff;
      border-radius: 20px;
      box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
      text-align: center;
      padding: 30px 25px;
      width: 100%;
      max-width: 400px;
      border: 1px solid #7dacb5;
      animation: fadeIn 0.6s ease-in-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(30px); }
      to { opacity: 1; transform: translateY(0); }
    }
    h1 { color: #5A827E; font-size: 24px; margin-bottom: 20px; font-weight: 600; }
    input[type="file"] {
      margin: 20px auto;
      display: block;
      width: calc(100% - 10px);
      border: 2px dashed #7dacb5;
      border-radius: 10px;
      padding: 12px;
      background-color: #f1f5f6;
      color: #333;
      cursor: pointer;
      font-size: 15px;
      box-sizing: border-box;
    }
    input[type="file"]:hover { border-color: #5A827E; background: #fff; }
    button {
      margin-top: 15px; padding: 12px 20px; background: #5A827E; color: #fff;
      border: none; border-radius: 10px; font-weight: bold; cursor: pointer;
      width: 100%; transition: 0.3s; font-size: 16px;
    }
    button:hover { background: #4e6f6b; transform: scale(1.03); }
    .footer { margin-top: 20px; font-size: 14px; color: #777; }
    @media (max-width: 480px) {
      .container { padding: 20px 15px; }
      h1 { font-size: 20px; } button { font-size: 15px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Skin Type Detector ✨</h1>
    <form action="/predict?uid={{uid}}" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Analyze</button>
    </form>
    <p class="footer">Upload a clear face image to reveal your skin type.</p>
  </div>
</body>
</html>
'''

# ---------------- Preprocess (kept same logic) ----------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- Routes ----------------
@app.route('/', methods=['GET'])
def index():
    uid = request.args.get("uid", "")
    return render_template_string(HTML_FORM.replace("{{uid}}", uid))

# Health endpoint for Koyeb
@app.route('/health', methods=['GET'])
def health():
    return "ok", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    user_id = request.args.get("uid")

    try:
        img = preprocess_image(image_bytes)
        model = get_model()
        preds = model.predict(img)
        predicted_label = labels[int(np.argmax(preds))]

        if db and user_id:
            db.collection("users").document(user_id).collection("skin_records").add({
                "skin_type": predicted_label,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

        return f'''
        <!doctype html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>GlowCheck Result</title>
          <style>
            body {{
              background: linear-gradient(135deg, #e6f0f2, #f8f8f8);
              font-family: 'Poppins', sans-serif;
              display: flex; justify-content: center; align-items: center;
              height: 100vh; margin: 0;
            }}
            .result-card {{
              position: fixed; top: 40px; left: 50%; transform: translateX(-50%);
              background: white; padding: 25px 20px; border-radius: 15px;
              box-shadow: 0 6px 18px rgba(0,0,0,0.1);
              text-align: center; width: 90%; max-width: 360px; border: 1px solid #7dacb5;
            }}
            h3 {{ color: #5A827E; font-size: 22px; font-weight: 600; margin-bottom: 25px; }}
            a {{ text-decoration: none; color: white; background: #5A827E; padding: 12px 20px;
                border-radius: 10px; display: inline-block; transition: 0.3s; }}
            a:hover {{ transform: scale(1.05); background: #4e6f6b; }}
          </style>
        </head>
        <body>
          <div class="result-card">
            <h3>Your Skin Type: {predicted_label}</h3>
            <a href="/?uid={user_id or ''}">Try Another</a>
          </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><br><a href='/'>Back</a>"

if __name__ == '__main__':
    # Local run only. In Koyeb we use Gunicorn.
    app.run(debug=True, host="0.0.0.0", port=8000)
