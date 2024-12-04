from datasets import load_dataset
import pandas as pd

def read_medical_chatbot_dataset():
    # Read
    dataset = load_dataset("ruslanmv/ai-medical-chatbot")
    df = pd.DataFrame(dataset['train'])
    # print(df.head())
    return df


