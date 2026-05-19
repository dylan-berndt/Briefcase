# =========================
# Build React frontend
# =========================
FROM node:20 AS frontend-build

WORKDIR /frontend

COPY site/frontend/package*.json ./
RUN npm ci

COPY site/frontend/ .
RUN npm run build


# =========================
# Flask backend runtime
# =========================
FROM python:3.11-slim

WORKDIR /app

# Ensure SQLite file path exists at runtime
RUN mkdir -p /data
ENV SQLITE_PATH=/app/backend/fontsearch.db

# System deps (safe baseline for ML/vector DBs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend code
COPY site/backend ./backend
COPY utils ./backend/utils
COPY checkpoints ./backend/checkpoints

# Inject React build into Flask static folder
COPY --from=frontend-build /frontend/build ./backend/static

WORKDIR /app/backend

EXPOSE 8000

# Production server (IMPORTANT for DO deployment)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]