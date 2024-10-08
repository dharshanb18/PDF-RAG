
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class DocumentAgent:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)  # Dimension of the model output
        self.chunks = []
        self.extracted_text = ""
        self.metadata = {}

    def load_pdf(self, pdf):
        self.metadata = self.extract_metadata(pdf)
        self.extracted_text = self.extract_text_from_pdf(pdf)
        self.chunks = self.split_text(self.extracted_text)
        self.create_embeddings(self.chunks)

    def extract_metadata(self, pdf):
        reader = PyPDF2.PdfReader(pdf)
        metadata = reader.metadata
        return metadata

    def extract_text_from_pdf(self, pdf):
        reader = PyPDF2.PdfReader(pdf)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text

    def split_text(self, text, chunk_size=500, overlap=100):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def create_embeddings(self, chunks):
        # Generate embeddings
        embeddings = self.embed_model.encode(chunks)

        # Check the shape of embeddings
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)  # Reshape if it's a single vector

        # Convert to float32 for FAISS compatibility
        embeddings = np.array(embeddings).astype('float32')

        # Add embeddings to FAISS index
        self.index.add(embeddings)
        return embeddings

    def get_extracted_text(self):
        return self.extracted_text

    def get_chunks(self):
        return self.chunks
