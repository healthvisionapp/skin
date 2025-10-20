#!/usr/bin/env bash
set -e

# Firebase service key (runtime secret)
if [ -n "$SA_JSON" ]; then
  echo "$SA_JSON" > serviceAccountkey.json
  echo "[ok] serviceAccountkey.json written"
fi

exec gunicorn app:app \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers 1 \
  --threads 1 \
  --timeout 300 \
  --access-logfile - \
  --error-logfile -
