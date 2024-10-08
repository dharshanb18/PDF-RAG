
import numpy as np
import re

class QueryAgent:
    def __init__(self, document_agent):
        self.document_agent = document_agent

    def keyword_search(self, query):
        keyword_results = []
        for chunk in self.document_agent.get_chunks():
            if re.search(r'\b' + re.escape(query) + r'\b', chunk, re.IGNORECASE):
                keyword_results.append(chunk)
        return keyword_results

    def get_relevant_chunks(self, query, top_k=3):
        # Semantic search using FAISS
        query_embedding = self.document_agent.embed_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.document_agent.index.search(query_embedding, top_k)
        semantic_results = [self.document_agent.get_chunks()[i] for i in indices[0]]

        # Perform keyword search
        keyword_results = self.keyword_search(query)

        # Combine and deduplicate results
        combined_results = list(set(semantic_results + keyword_results))
        return " ".join(combined_results)
