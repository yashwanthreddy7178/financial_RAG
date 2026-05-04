import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self):
        """
        We initialize our text splitter when the class is created.
        - chunk_size: The target length of each chunk (1000 characters).
        - chunk_overlap: We overlap by 200 characters so we don't accidentally 
          slice a crucial financial sentence in half!
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            # This tells LangChain to try splitting by Double Newlines (paragraphs) first, 
            # then Single Newlines, then Spaces. It respects grammar!
            separators=["\n\n", "\n", " ", ""] 
        )

    def process_pdf(self, file_path: str) -> tuple[str, list[str]]:
        """
        Takes a PDF file, extracts it to markdown (preserving tables),
        and splits it into semantic chunks.
        
        Returns a TUPLE: (full_markdown_text, list_of_chunks)
        We return the full text too so the Contextual Retrieval step can
        build the sliding window context around each chunk.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file: {file_path}")

        # 1. EXTRACTION: Convert PDF to Markdown
        md_text = pymupdf4llm.to_markdown(file_path)

        if not md_text or len(md_text.strip()) == 0:
            raise ValueError("The extracted text is empty.")

        # 2. CHUNKING: Split the massive string into a list of smaller strings
        chunks = self.text_splitter.split_text(md_text)
        
        # We return BOTH: the full raw text (for context window) and the chunks (for embedding)
        return md_text, chunks
