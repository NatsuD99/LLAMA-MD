from typing import List
from transformers import AutoModel, AutoTokenizer
import torch


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

    @torch.no_grad()
    def get_embeddings(self, docs: List[str], input_type: str) -> List[List[float]]:
        """
        Get embeddings using the UAE-Large-V1 model.

        Args:
            docs (List[str]): List of texts to embed
            input_type (str): Type of input to embed. Can be "document" or "query".

        Returns:
            List[List[float]]: Array of embedddings
        """
        # Prepend retrieval instruction to queries
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


# Example usage
if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    docs = ["The capital of France is Paris", "The capital of Spain is Madrid"]
    embeddings = embedding_model.get_embeddings(docs, "document")
    print(embeddings)
