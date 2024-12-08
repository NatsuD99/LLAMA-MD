# Dr. GPT

## Directory Structure
```bash

├── Code
│   ├── data
│   ├── finetune
│   ├── rag
│   ├── utils
│   ├── ReadMe.md
│   ├── requirements.txt
│   ├── app.py

```

## Data Directory

```bash
├── data
│   ├── data.json
│   ├── data.csv
│   ├── data.txt
│   ├── data.pkl
│   ├── data.npy

```

## Finetune Directory

```bash
├── finetune
│   ├── finetune.py
│   ├── inference.py

```

### finetune.py
This file contains our code for finetuning the LLama model

### inference.py
This file contains our code for inference using the finetuned model. 

## RAG Directory

```bash
├── rag
│   ├── embeddings.py
│   ├── vector_db.py
│   ├── test.py
│   ├── utils.py

```

### embeddings.py
This file contains the code to generate the embeddings for the documents in the dataset. The embedding can be generated using the following methods:
1. Hugging Face Model
    - WhereIsAI/UAE-Large-V1
    - dmis-lab/biobert-base-cased-v1.1
2. Bedrock API
    - amazon.titan-embed-text-v2:0

Sample code is provided in the file to generate the embeddings using the above methods.

### vector_db.py
This file contains the code to set up and query the vector database. The vector database is used to store the embeddings of the documents in the dataset.
Sample code is provided in the file to set up and query the vector database.

### test.py
This file contains the code to test the RAG.
- The code generates the embeddings for the query.
- Retrieve the context from Pinecone.
- Generate the answer using the RAG.

### utils.py
 This file contains code to get response for the query from bedrock for RAG testing purpose.

## Utils Directory

```bash 
├── utils
│   ├── preprocess.py

```
Code to preprocess the data for RAG.

## Streamlit App

```bash
├── Code
│   ├── app.py

```
`app.py` is the streamlit app for the project which contains a chatbot interface for the user to interact with the model.

Run Command:
```bash
streamlit run app.py
```
