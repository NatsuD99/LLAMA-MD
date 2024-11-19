import json
import os

import boto3
from botocore.exceptions import ClientError

from vector_db import VectorDB


def create_bedrock_client():
    aws_access_key_id, aws_secret_access_key, aws_session_token = self.get_bedrock_key()
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )
    return session.client("bedrock-runtime", region_name="us-east-1")


def get_response(query):
    # Get Context
    vector_db = VectorDB(pinecone_api_key=os.getenv("PINECONE_API_KEY"), pinecone_env="us-west1-gcp",
                         index_name="test-index", dimension=768, metric="cosine", cloud="gcp")
    search_results = vector_db(query, top_k=5)
    # Prepare Prompt
    prompt = f"""
    Question: {query}
    Context: {search_results}
    """
    body_content = {
        "prompt": prompt
    }
    # Get Response
    try:
        client = create_bedrock_client()
        response = client.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            body=json.dumps(body_content),
        )
        return response
    except ClientError as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    query = "I eat a lot do I have cancer ?"
    get_response(query)
