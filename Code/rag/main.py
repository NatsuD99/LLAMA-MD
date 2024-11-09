query = "I eat a lot do I have cancer ?"
import os
from vector_db import VectorDB
vector_db = VectorDB(pinecone_api_key=os.getenv("PINECONE_API_KEY"), pinecone_env="us-west1-gcp", index_name="test-index", dimension=768, metric="cosine", cloud="gcp")
search_results = vector_db(query, top_k=5)
print(search_results)


