# Slim Python image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# OS deps for Pillow/OpenCV/TensorFlow CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Model ko image me bake karo (Dropbox direct link) ----
# NOTE: yahan aapka wahi MODEL_URL hardcode kar rahe hain taake startup pe download na karna pade
RUN curl -L --fail "https://www.dropbox.com/scl/fi/zq3rd08qztt52sad61m30/skin_type_classifier.h5?rlkey=wagg56ok83eu8d1ay25o1g3pr&st=jy1w96k8&dl=1" \
    -o skin_type_classifier.h5

# ---- App files ----
COPY app.py ./

# ---- Entry point: SA_JSON ko file me likho, phir gunicorn ----
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

# Optional healthcheck (root pe HTML aata hai)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/ || exit 1

CMD ["/entrypoint.sh"]
