"""serve.py's optional API key guards every route except /health."""
import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

spec = importlib.util.spec_from_file_location("serve", Path(__file__).parents[1] / "scripts/serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)
KEY = "k" * 32


class Engine:
    def __init__(self):
        self.config = {"name": "trutina-4b"}


@pytest.fixture
def url(monkeypatch):
    monkeypatch.setitem(serve.ROUTES, "/v1/echo", lambda body, engine: {"echo": body})
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(Engine(), 1 << 20, KEY))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def post(url, headers):
    req = Request(url + "/v1/echo", data=b'{"a": 1}', headers={"Content-Type": "application/json", **headers})
    with urlopen(req) as r:
        return json.loads(r.read())


def test_health_is_open(url):
    with urlopen(url + "/health") as r:
        assert json.loads(r.read())["status"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"X-API-Key": KEY[:-1]}])
def test_rejects_missing_or_wrong_key(url, headers):
    with pytest.raises(HTTPError) as e:
        post(url, headers)
    assert e.value.code == 401


@pytest.mark.parametrize("headers", [{"Authorization": f"Bearer {KEY}"}, {"X-API-Key": KEY}])
def test_accepts_key(url, headers):
    assert post(url, headers) == {"echo": {"a": 1}}
