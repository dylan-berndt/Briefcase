import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..", "backend")
sys.path.append(os.path.join(HERE, "..", "tools"))
from fakeBundle import makeFakeBundle  # noqa: E402


def freePort():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def site(tmp_path_factory):
    """The production entry point (gunicorn app:app) serving the built frontend."""
    build = os.path.join(HERE, "..", "frontend", "build")
    if not os.path.exists(os.path.join(build, "index.html")):
        pytest.skip("frontend is not built: cd site/frontend && npm run build")
    work = tmp_path_factory.mktemp("site")
    planted = makeFakeBundle(str(work / "bundle"), numFonts=300, seed=0)
    port = freePort()
    env = {**os.environ, "SECRET_KEY": "e2e-secret-e2e-secret-e2e-secret-0123", "BUNDLE_DIR": str(work / "bundle"),
           "SQLITE_PATH": str(work / "site.db"), "STATIC_DIR": build, "COOKIE_SECURE": "0"}
    process = subprocess.Popen([sys.executable, "-m", "gunicorn", "-w", "1", "-b", f"127.0.0.1:{port}", "app:app"],
                               cwd=BACKEND, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            break
        except Exception:
            if process.poll() is not None:
                raise RuntimeError(process.stdout.read().decode())
            time.sleep(0.1)
    else:
        process.kill()
        raise RuntimeError("server did not start")
    yield {"url": url, "planted": planted}
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright
    path = os.environ.get("CHROMIUM_PATH") or ("/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None)
    with sync_playwright() as p:
        instance = p.chromium.launch(executable_path=path, args=["--no-sandbox"])
        yield instance
        instance.close()


@pytest.fixture
def page(browser, site):
    context = browser.new_context(viewport={"width": 1400, "height": 1000}, base_url=site["url"])
    context.set_default_timeout(8000)
    # The animated WebGL background is software-rendered in a headless browser and starves everything else.
    # Slow its animation loop down; nothing under test uses requestAnimationFrame.
    context.add_init_script("window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 1000);")
    page = context.new_page()
    page.errors = []
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    yield page
    context.close()


def shot(page, name):
    directory = os.environ.get("E2E_SHOTS")
    if directory:
        os.makedirs(directory, exist_ok=True)
        page.screenshot(path=os.path.join(directory, name + ".png"), full_page=True)
