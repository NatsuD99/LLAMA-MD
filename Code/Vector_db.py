import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import re

# Currently scraping Wiki data. Will change code once Data files have been prepared.
url = "https://en.wikipedia.org/wiki/Cancer" # Sample cancer page data
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

text = ''
for paragraph in soup.find_all('p'):
    text += paragraph.get_text()
text = ' '.join(text.split())
# Clean up the text: removing citation brackets [7], [1] etc.
text = re.sub(r'\[\d+\]', '', text)
# Define the desired number of chunks
chunk_size = 10
words = text.split()  # Split the text into individual words
total_words = len(words)
words_per_chunk = max(1, total_words // chunk_size)  # Calculate words per chunk

chunks = [' '.join(words[i:i + words_per_chunk]) for i in range(0, total_words, words_per_chunk)]

# Adjust the number of chunks if we have extra words in the last chunk
if len(chunks) > chunk_size:
    last_chunk = ' '.join(chunks[chunk_size - 1:])  # Combine excess words into the last chunk
    chunks = chunks[:chunk_size - 1] + [last_chunk]

# TOKENIZATION- Different methods written. Plan is to use BioBERT
# Using BERT to tokenize text
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")

# Using Senetence-BERT
# from sentence_transformers import SentenceTransformer
# sentence_model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
# sentences = text.split('. ')  # Simple split based on periods for sentence-level splitting
# sentence_embeddings = sentence_model.encode(sentences)
# print("Sentence embeddings shape:", sentence_embeddings.shape)

# Using BioBERT to tokenize text
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Generate embeddings for each chunk using mean pooling
chunk_embeddings = []
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)

    # Apply mean pooling to get the average embedding across all tokens in the chunk
    attention_mask = inputs['attention_mask']
    last_hidden_state = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    chunk_embedding = sum_embeddings / sum_mask  # Mean pooled embedding vector for the chunk

    chunk_embeddings.append(chunk_embedding)

print("Generated embeddings for each chunk.")
print("Number of chunks:", len(chunks))
print("Embedding vector shape for each chunk:", chunk_embeddings[0].shape if chunk_embeddings else "No embeddings generated.")
print("The embedding", chunk_embeddings)