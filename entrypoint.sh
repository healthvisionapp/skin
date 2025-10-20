#!/usr/bin/env bash
set -e

# Write Firebase service key from secret
if [ -n "$SA_JSON" ]; then
  echo "$SA_JSON" > serviceAccountkey.json
  echo "[ok] serviceAccountkey.json written"
fi

# Ensure model exists (download at runtime)
if [ ! -f "skin_type_classifier.h5" ]; then
  if [ -n "$MODEL_URL" ]; then
    echo "[info] downloading model..."
    curl -L --fail "$MODEL_URL" -o skin_type_classifier.h5
    echo "[ok] model downloaded"
  else
    echo "[err] MODEL_URL not set and model file missing" && exit 1
  fi
fi

# Start server
exec gunicorn app:app \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers 1 \
  --threads 1 \
  --timeout 300 \
  --access-logfile - \
  --error-logfile -
