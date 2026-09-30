# End-to-end tests

Drive the built site in a real browser against the real Flask app and a fake bundle (`site/tools/fakeBundle.py`).

    cd site/frontend && npm ci && npm run build
    pip install -r site/backend/requirements.txt pytest playwright pillow && playwright install chromium
    pytest site/e2e

`CHROMIUM_PATH` points at a specific Chromium binary if Playwright's own is not installed. `E2E_SHOTS=<dir>` saves screenshots.
