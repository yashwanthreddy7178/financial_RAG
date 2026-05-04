import pytest
import io

# =====================================================================
# WHAT IS AN INTEGRATION TEST?
# Unlike Unit Tests that test one isolated function, Integration Tests
# verify that multiple pieces of the system work together. 
# Here, we test if the FastAPI web server correctly receives a request,
# routes it to the right function, and returns the expected HTTP status code.
# =====================================================================


def test_health_check(client):
    # 'client' is magically passed in from tests/conftest.py!
    
    # Act: We simulate a user opening a web browser and navigating to "http://localhost/health"
    response = client.get("/health")
    
    # Assert: We expect the server to say "200 OK", which means success.
    assert response.status_code == 200
    
    # We also check that the JSON returned contains 'status: online'.
    data = response.json()
    assert data["status"] == "online"


def test_search_empty_question(client):
    # Act: We simulate a user sending a POST request to "/search" but the question is blank.
    # The 'json' parameter automatically converts the Python dictionary to a JSON payload.
    response = client.post("/search", json={"question": "   "})
    
    # Assert: Our code in main.py specifically checks for empty strings and raises a 400 Error.
    # 400 means "Bad Request" (the user messed up, not the server).
    assert response.status_code == 400
    
    # We verify the error message matches what we wrote in our code.
    data = response.json()
    assert data["detail"]["error"] == "Question cannot be empty."


def test_ingest_invalid_file_type(client):
    # Act: We simulate a user trying to upload a text file instead of a PDF.
    
    # We create a fake text file in memory using io.BytesIO so we don't have to save a real file.
    fake_file_content = b"This is just some text."
    fake_file = io.BytesIO(fake_file_content)
    
    # To upload a file with httpx/TestClient, we use the 'files' parameter.
    # The format is 'form_field_name': ('filename', file_object, 'mime_type')
    files = {"file": ("document.txt", fake_file, "text/plain")}
    
    response = client.post("/ingest", files=files)
    
    # Assert: Our code checks if the filename ends with ".pdf". If not, it throws a 400.
    assert response.status_code == 400
    data = response.json()
    assert "Only PDF files are accepted" in data["detail"]["error"]
