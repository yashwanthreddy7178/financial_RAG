import pytest
import os
from unittest.mock import patch
from services.document_processor import DocumentProcessor

# =====================================================================
# WHAT IS A UNIT TEST?
# A unit test checks one single "unit" of code (like a single function or class)
# in complete isolation. It shouldn't depend on databases, external APIs,
# or even real files if we can avoid it.
# =====================================================================

# We use @pytest.fixture to create a fresh instance of DocumentProcessor for every test.
# This ensures that one test doesn't accidentally mess up the state for the next test.
@pytest.fixture
def processor():
    return DocumentProcessor()

# =====================================================================
# WHAT IS MOCKING?
# Sometimes our code calls external libraries (like reading a PDF from the disk).
# In a unit test, we don't want to actually read a real PDF on the disk because
# the disk might be slow, the file might be missing, or the format might change.
# Instead, we "mock" (fake) the behavior of those external functions.
# =====================================================================

# The @patch decorator tells Python: "Whenever the code tries to use 'os.path.exists'
# inside the test, intercept it and replace it with a fake (mock) object."
@patch("os.path.exists")
# We also patch 'pymupdf4llm.to_markdown' so it doesn't actually try to parse a PDF.
@patch("pymupdf4llm.to_markdown")
def test_process_pdf_success(mock_to_markdown, mock_exists, processor):
    # This is a test for the "Happy Path" (when everything works correctly).
    
    # 1. Arrange: Set up our fake environment.
    # We tell our fake os.path.exists to always return True (pretend the file is there).
    mock_exists.return_value = True
    
    # We tell our fake PDF parser to return a simple markdown string instead of parsing a real file.
    fake_markdown = "This is a very long financial sentence that we want to split. " * 30
    mock_to_markdown.return_value = fake_markdown

    # 2. Act: Call the actual function we are trying to test.
    # We pass in a fake file path. Our mocks will intercept it.
    full_text, chunks = processor.process_pdf("fake_financial_report.pdf")

    # 3. Assert: Check if the function did what we expected.
    # We expect the full_text returned to be exactly our fake_markdown.
    assert full_text == fake_markdown
    
    # We expect the text to be split into chunks. Let's make sure there is more than 1 chunk.
    assert len(chunks) > 1
    
    # Since our chunk size is 1000, we assert that the first chunk is roughly 1000 characters (or less).
    # It might be slightly less because LangChain tries to split neatly at spaces or newlines.
    assert len(chunks[0]) <= 1000

@patch("os.path.exists")
def test_process_pdf_file_not_found(mock_exists, processor):
    # This tests the "Sad Path" (what happens if a user uploads a missing file?).
    
    # Arrange: Tell the fake 'exists' function to say the file doesn't exist.
    mock_exists.return_value = False

    # Act & Assert: We use pytest.raises to tell pytest: 
    # "I expect the following block of code to crash with a FileNotFoundError."
    # If it DOESN'T crash, the test will fail!
    with pytest.raises(FileNotFoundError):
        processor.process_pdf("missing_file.pdf")

@patch("os.path.exists")
@patch("pymupdf4llm.to_markdown")
def test_process_pdf_empty_text(mock_to_markdown, mock_exists, processor):
    # This tests what happens if the PDF is technically there, but it's totally blank.
    
    # Arrange: File exists, but the markdown parser returns an empty string.
    mock_exists.return_value = True
    mock_to_markdown.return_value = "   " # Just spaces

    # Act & Assert: Our code should throw a ValueError complaining about empty text.
    with pytest.raises(ValueError, match="The extracted text is empty."):
        processor.process_pdf("blank_document.pdf")
