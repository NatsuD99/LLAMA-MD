import json
import os
from typing import List

import boto3
import torch
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    RETRIEVAL_INSTRUCT = "Represent this sentence for searching relevant passages:"

    def __init__(self, model_name: str = 'WhereIsAI/UAE-Large-V1'):
        """
        Initialize the UAE-Large-V1 model and tokenizer.

        Args:
            model_name (str): The name of the model to load.
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # print(f"Using Model: {model_name}")

    @staticmethod
    def get_bedrock_key():
        load_dotenv()
        return os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"), os.getenv("AWS_SESSION_TOKEN")

    def create_bedrock_client(self):
        aws_access_key_id, aws_secret_access_key, aws_session_token = self.get_bedrock_key()
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        return session.client("bedrock-runtime", region_name="us-east-1")

    @torch.no_grad()
    def _get_embeddings_huggingface(self, docs: List[str], input_type: str) -> List[List[float]]:
        """
        Get embeddings from huggingface model.

        Args:
            docs (List[str]): List of texts to embed
            input_type (str): Type of input to embed. Can be "document" or "query".

        Returns:
            List[List[float]]: Array of embedddings
        """
        if input_type == "query":
            docs = ["{}{}".format(self.RETRIEVAL_INSTRUCT, q) for q in docs]

        # Tokenize input texts
        inputs = self.tokenizer(docs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(
            self.device)

        # Pass tokenized inputs to the model, and obtain the last hidden state
        last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state

        # Extract embeddings from the last hidden state
        embeddings = last_hidden_state[:, 0]
        return embeddings.cpu().numpy()

    def _get_embedding_bedrock(self, model_id, text):
        """
        Get embeddings using the Bedrock API.
        :param model_id:
        :param text:
        :return: List[float]
        """
        print(f"Embedding from Bedrock for text: {text}")
        # Body content structure for embedding API
        body_content = {
            "inputText": text
        }
        try:
            client = self.create_bedrock_client()
            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body_content)
            )
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
        except ClientError as e:
            print(f"An error occurred: {e}")
            return None

    def get_embeddings(self, docs: List[str], input_type: str, platform: str) -> List[List[float]]:
        """
        Get embeddings using the specified platform.
        :param docs:
        :param input_type:
        :param platform:
        :return: List[List[float]]
        """
        if platform == "huggingface":
            return self._get_embeddings_huggingface(docs, input_type)
        elif platform == "bedrock":
            model_id = "amazon.titan-embed-text-v2:0"
            return [self._get_embedding_bedrock(model_id, doc) for doc in docs]
        else:
            raise ValueError("Model not supported")

    def __call__(self, docs: List[str], input_type: str, platform: str) -> List[List[float]]:
        return self.get_embeddings(docs, input_type, platform)


# Example usage
if __name__ == "__main__":
    embedding_model = EmbeddingModel(model_name="dmis-lab/biobert-base-cased-v1.1")
    docs = ["The capital of France is Paris", "The capital of Spain is Madrid"]
    embeddings = embedding_model.get_embeddings(docs, "document", "bedrock")
    print(embeddings)
