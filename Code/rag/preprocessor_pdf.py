#%%
from typing import List
import re

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd

class DataProcessor:
    def __init__(self, documents: List[List[Document]], chunk_size: int = 1500, chunk_overlap: int = 150):
        self.documents = documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and formatting."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation brackets
        text = text.replace('\n', ' ')  # Replace new lines with space
        text = re.sub(r'[^a-zA-Z0-9.,;:!?()\'\" ]', '',
                      text)  # Keep only some punctuation and alphanumeric characters
        return text.strip()

    def process_documents(self):
        """Process all documents by cleaning, chunking, and appending metadata."""
        processed_documents = []
        for sublist in self.documents:  # loop to handle each sublist
            for document in sublist:  # Existing loop now iterates within each sublist
                cleaned_text = self.clean_text(document.page_content)
                chunks = self.text_splitter.split_text(cleaned_text)
                for chunk in chunks:
                    processed_document = Document(page_content=chunk, metadata=document.metadata.copy())
                    processed_documents.append(processed_document)
        print(f"Processed {len(processed_documents)} chunks from {len(self.documents)} documents.")
        return processed_documents


# example usage
if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(current_dir, '..', 'data', 'Raw_Text')
    loaded_documents = []

    #looping txt files
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)


        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()

                loaded_documents.append(Document(page_content=text_content, metadata={"title": filename}))


    documents = [loaded_documents]
    data_processor = DataProcessor(documents)
    processed_documents = data_processor.process_documents()
#%%

data = {"content": [doc.page_content for doc in processed_documents],
        "metadata": [doc.metadata for doc in processed_documents]}

df = pd.DataFrame(data)

print(df)
unique_filenames = set(doc.metadata["title"] for doc in processed_documents)
print("Unique filenames:", unique_filenames)
first_content = df.iloc[0]["content"]
print(first_content)


