import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template_string
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# ---------------- Firebase Setup ----------------
cred = credentials.Certificate("serviceAccountkey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------- Model Setup ----------------
model = load_model('skin_type_classifier.h5')
labels = ['Dry', 'Normal', 'Oily']
IMG_SIZE = (224, 224)

# ---------------- HTML Form (Eye Classifier Style) ----------------
HTML_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GlowCheck</title>
  <style>
body {
  font-family: 'Poppins', sans-serif;
  background: linear-gradient(135deg, #e6f0f2, #f8f8f8);
  color: #333;
  display: flex;
  justify-content: center;
  align-items: flex-start; /* moves card toward the top */
  min-height: 100vh;
  margin: 0;
  padding: 40px 15px; /* top padding for spacing */
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

    h1 {
      color: #5A827E;
      font-size: 24px;
      margin-bottom: 20px;
      font-weight: 600;
    }

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


    input[type="file"]:hover {
      border-color: #5A827E;
      background: #fff;
    }

    button {
      margin-top: 15px;
      padding: 12px 20px;
      background: #5A827E;
      color: #fff;
      border: none;
      border-radius: 10px;
      font-weight: bold;
      cursor: pointer;
      width: 100%;
      transition: 0.3s;
      font-size: 16px;
    }

    button:hover {
      background: #4e6f6b;
      transform: scale(1.03);
    }

    .footer {
      margin-top: 20px;
      font-size: 14px;
      color: #777;
    }

    @media (max-width: 480px) {
      .container {
        padding: 20px 15px;
      }
      h1 {
        font-size: 20px;
      }
      button {
        font-size: 15px;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>GlowCheck ✨</h1>
    <form action="/predict?uid={{uid}}" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Analyze</button>
    </form>
    <p class="footer">Upload a clear face image to reveal your skin type.</p>
  </div>
</body>
</html>
'''

# ---------------- Image Preprocessing ----------------
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

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    user_id = request.args.get("uid")

    try:
        img = preprocess_image(image_bytes)
        prediction = model.predict(img)
        predicted_label = labels[np.argmax(prediction)]

        if user_id:
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
              display: flex;
              justify-content: center;
              align-items: center;
              height: 100vh;
              margin: 0;
            }}
          .result-card {{
  position: fixed;
  top: 40px;
  left: 50%;
  transform: translateX(-50%);
  background: white;
  padding: 25px 20px;
  border-radius: 15px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.1);
  text-align: center;
  width: 90%;
  max-width: 360px;
  border: 1px solid #7dacb5;
}}

            h3 {{
              color: #5A827E;
              font-size: 22px;
              font-weight: 600;
              margin-bottom: 25px;
            }}
            a {{
              text-decoration: none;
              color: white;
              background: #5A827E;
              padding: 12px 20px;
              border-radius: 10px;
              display: inline-block;
              transition: 0.3s;
            }}
            a:hover {{
              transform: scale(1.05);
              background: #4e6f6b;
            }}
          </style>
        </head>
        <body>
          <div class="result-card">
            <h3>Your Skin Type: {predicted_label}</h3>
            <a href="/?uid={user_id}">Try Another</a>
          </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><br><a href='/'>Back</a>"

# ---------------- Run Flask ----------------
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5002)
