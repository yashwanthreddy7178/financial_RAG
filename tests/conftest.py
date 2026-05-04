import pytest
from fastapi.testclient import TestClient

# We import the FastAPI 'app' object from our main.py file.
# We need this to spin up a mock test server.
from main import app

# @pytest.fixture is a decorator. It tells pytest that this function is a "fixture".
# A fixture is basically a setup function. Whenever a test asks for a 'client',
# pytest will automatically run this function and pass its result into the test.
@pytest.fixture
def client():
    # TestClient is a tool provided by FastAPI (built on top of httpx).
    # It acts like a browser or API client (like Postman) but it runs locally
    # and directly calls your FastAPI app's code without needing a real network port.
    # This makes testing endpoints super fast and isolated!
    with TestClient(app) as test_client:
        # We 'yield' the client instead of 'return' so that if we needed to do
        # any cleanup after the test finishes, we could put it after the yield statement.
        yield test_client
