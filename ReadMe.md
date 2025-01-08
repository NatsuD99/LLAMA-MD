# LLAMA, MD: Specializing in Gynecology

## Project Overview
Welcome to the **LLAMA, MD** project repository! This project focuses on fine-tuning a Large Language Model (LLM) to specialize in gynecology, particularly in pregnancy-related topics, to serve as a virtual medical assistant. **LLAMA, MD** aims to adapt the Llama 3.2 (1B parameters) model for specialized medical assistance in gynecology. By leveraging topic modeling, data retrieval, and fine-tuning techniques, the project creates a Retrieval-Augmented Generation (RAG) system capable of providing accurate and contextually relevant responses in the domain of pregnancy. This was the final project for our NLP course.

## Features

- **Topic Modeling**: Extracted top topics to identify pregnancy as the focus area.
- **Data Retrieval**: Compiled a corpus of comprehensive books and Wikipedia data on pregnancy.
- **Fine-Tuning**: Adapted Llama 3.2 using the AI Medical Chatbot dataset for medical conversational tasks.
- **RAG Integration**: Combined the fine-tuned model with a RAG framework for enhanced information retrieval.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/NatsuD99/LLAMA-MD.git
   cd LLAMA-MD/Code

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt

## Usage

To interact with the LLAMA, MD virtual assistant:

1. **Run the Streamlit application:**

  ```bash
  streamlit run app.py --server.port 8888
  ```
2. **Access the application:**

    Open your web browser and navigate to http://localhost:8888 to interact with the chatbot interface.

## Future Improvements
Potential enhancements include:

* Larger Models: Utilizing models with more parameters to improve contextual understanding.

* QLoRA Integration: Implementing Quantized LoRA for efficient training of larger models.

* Dataset Expansion: Including additional gynecology subdomains and multimodal data to broaden expertise. Currently [this dataset](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)
 was used.

## How to contribute

1. Push All the code in the 'Code' folder.
2. Make a PR and assign other members for approval before merging.
3. Do not directly merge the PR without approval.
