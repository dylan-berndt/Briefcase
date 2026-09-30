import os

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-test-secret-test-secret-0123")

from fakeBundle import makeFakeBundle  # noqa: E402
from app import createApp  # noqa: E402


@pytest.fixture(scope="session")
def fake(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bundle")
    planted = makeFakeBundle(str(directory), numFonts=300, seed=0)
    return str(directory), planted


@pytest.fixture
def makeApp(tmp_path, fake):
    def make(**overrides):
        static = tmp_path / "static"
        static.mkdir(exist_ok=True)
        (static / "index.html").write_text("<html>index</html>")
        (static / "hello.txt").write_text("hi")
        config = {"DATABASE": str(tmp_path / "test.db"), "BUNDLE_DIR": fake[0], "STATIC_DIR": str(static),
                  "COOKIE_SECURE": False, "RATELIMIT_ENABLED": False, "TESTING": True}
        config.update(overrides)
        return createApp(config)
    return make


@pytest.fixture
def app(makeApp):
    return makeApp()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def planted(fake):
    return fake[1]


def register(client, username="alice", password="correct horse"):
    return client.post("/api/font/register", data={"username": username, "password": password})


def login(client, username="alice", password="correct horse"):
    return client.post("/api/font/login", data={"username": username, "password": password})


@pytest.fixture
def user(client):
    """A client logged in as alice."""
    assert register(client).status_code == 200
    assert login(client).status_code == 200
    return client
