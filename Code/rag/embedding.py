#Sampel code to get embeddings using the UAE-Large-V1 model for testing Pinecone setup.

from typing import List
from transformers import AutoModel, AutoTokenizer
import torch


RETRIEVAL_INSTRUCT = "Represent this sentence for searching relevant passages:"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Load the UAE-Large-V1 model
model = AutoModel.from_pretrained('WhereIsAI/UAE-Large-V1').to(device)

#Load the tokenizesr
tokenizer = AutoTokenizer.from_pretrained('WhereIsAI/UAE-Large-V1')

@torch.no_grad()
def get_embeddings(docs: List[str], input_type: str) -> List[List[float]]:
    """
    Get embeddings using the UAE-Large-V1 model.

    Args:
        docs (List[str]): List of texts to embed
        input_type (str): Type of input to embed. Can be "document" or "query".

    Returns:
        List[List[float]]: Array of embedddings
    """
    #Prepend retrieval instruction to queries
    if input_type == "query":
        docs = ["{}{}".format(RETRIEVAL_INSTRUCT, q) for q in docs]
    #Tokenize input texts
    inputs = tokenizer(docs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    #Pass tokenized inputs to the model, and obtain the last hidden state
    last_hidden_state = model(**inputs, return_dict=True).last_hidden_state
    #Extract embeddings from the last hidden state
    embeddings = last_hidden_state[:, 0]
    return embeddings.cpu().numpy()

# Example usage
docs = ["The capital of France is Paris", "The capital of Spain is Madrid"]
embeddings = get_embeddings(docs, "document")
print(embeddings)
