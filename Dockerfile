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

COPY site/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend code, with the search bundle in backend/data (built by site/tools, committed through git-lfs)
COPY site/backend ./backend
# The query parser and its reviewed vocabulary are shared with the research code, which owns them
COPY utils/tagVocabulary.py ./backend/tagVocabulary.py
COPY configs/tagVocabulary.json ./backend/configs/tagVocabulary.json

# Inject React build into Flask static folder
COPY --from=frontend-build /frontend/build ./backend/static

# # Add the font map page to the static folder
# COPY results/fontMap.html ./backend/static/points.html

# # Check for points.html
# RUN find /app -name "*.html"

WORKDIR /app/backend

# Fail the build, rather than the deploy, if the bundle is missing or is an unpulled git-lfs pointer
RUN python -c "from bundle import Bundle; b = Bundle('data'); print('search bundle:', len(b.fonts), 'fonts,', len(b.vocab), 'tags')"

EXPOSE 8000

# Production server (IMPORTANT for DO deployment)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
