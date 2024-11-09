import uuid
from typing import List

import torch
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from embedding import EmbeddingModel
from dotenv import load_dotenv
load_dotenv()
import os


class VectorDB:
    def __init__(self, pinecone_api_key: str, pinecone_env: str, index_name: str, dimension: int,
                 metric: str, cloud: str):
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.cloud = cloud
        self.index_name = index_name
        self.dimension = int(dimension)
        self.metric = metric
        self.pinecone_client = self.setup_pinecone()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def setup_pinecone(self):
        pc = Pinecone(api_key=self.pinecone_api_key)
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.pinecone_env
                )
            )
        return pc.Index(self.index_name)

    def get_embedding(self, text: str) -> List[List[float]]:
        embedding_model = EmbeddingModel()
        embeddings = embedding_model.get_embeddings([text], "document")
        return embeddings


    def process_and_store_documents(self, documents: List[Document]):
        for document in documents:
            embedding = self.get_embedding(document.page_content)
            pinecone_id = str(uuid.uuid4())
            pinecone_metadata = document.metadata
            pinecone_metadata['text'] = document.page_content
            self.pinecone_client.upsert(vectors=[{"id": pinecone_id,
                                                  "values": embedding,
                                                  "metadata": pinecone_metadata}], show_progress=True)


# example usage
if __name__ == '__main__':
    pinecone_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENV')
    cloud = os.getenv('PINECONE_CLOUD')
    index_name = os.getenv('PINECONE_INDEX_NAME')
    dimension = os.getenv('DIMENSION')
    metric = os.getenv('METRIC')

    embedding_manager = VectorDB(
        pinecone_api_key=pinecone_key,
        pinecone_env=pinecone_env,
        cloud=cloud,
        index_name=index_name,
        dimension=dimension,
        metric=metric
    )
    documents = [Document(page_content="hello", metadata={"source": "test"})]
    embedding_manager.process_and_store_documents(documents)