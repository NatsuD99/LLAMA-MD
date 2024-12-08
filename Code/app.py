import time
import streamlit as st
import os
import json
from botocore.exceptions import ClientError
from rag.vector_db import VectorDB
from rag.embedding import EmbeddingModel
from rag.utils import create_bedrock_client
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)


def generate_response(msg):
    return "demo"

def doctor_response(question, max_length=5000, temperature=0.5):
    """
    Generates a doctor's response based on the patient's question.

    Parameters:
        question (str): The patient's question.
        max_length (int): The maximum token length of the response.
        temperature (float): Sampling temperature for diversity.

    Returns:
        str: The doctor's response.
    """
    prompt = (
        "Patient: {question}\n"
        "Answer:".format(question=question)
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:", 1)[-1].strip()
    return response


def handle_rag(query):
    # Get Context
    vector_db = VectorDB(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_env=os.getenv("PINECONE_ENV"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        dimension=int(os.getenv("DIMENSION")),
        metric=os.getenv("METRIC"),
        cloud=os.getenv("PINECONE_CLOUD")
    )
    search_results = vector_db(query, top_k=5)

    # Prepare Prompt
    prompt = f"""
    If query is not related to gynacology, pregnancy, or obstetrics, answer with "I am a Gynacology Bot. I can't help you with this query."
    If you cannot find context, answer with existing knowledge.
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
        raw_response= response['body'].read()
        cleaned_response = json.loads(raw_response).get("generation", "No response available.").strip()
        return cleaned_response
    except ClientError as e:
        print(f"Error: {e}")
        return "Error in processing the query. Please try again."


def handle_fine_tune(query):
    # Process user query in Fine Tune mode
    return f"Fine Tune response for: {query}"


def handle_base(query):
    # Process user query in Base mode
    # return f"Base mode response for: {query}"
    response = doctor_response(query)
    return response

def main():
    st.set_page_config(page_title="DR GPT", page_icon="🤖", layout="wide")

    # Sidebar
    with st.sidebar:
        st.markdown("## About DR GPT")
        st.write(
            "DR GPT is a Gynecology and Obstetrics chatbot that can answer questions about pregnancy, childbirth"
        )

        # Display tips
        with st.expander("Tips for using DR GPT"):
            st.info("""
            - Ask questions related to pregnancy, childbirth, and gynecology.
            - Be polite and respectful.
            - Avoid sharing personal information.
            - If you encounter any issues, please report them.
            """)

        # Dropdown selector
        option = st.selectbox(
            "Choose a mode:",
            ("Select an option", "Fine Tune", "Base", "RAG"),
            index=0,
            help="Select a mode to activate the respective functionality.",
        )
        st.session_state["selected_mode"] = option  # Save the selected mode

        st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
        if st.button("Clear Chat"):
            st.session_state.messages = []

        st.markdown("</div>", unsafe_allow_html=True)

    # Main chat interface
    st.markdown(
        "<h1 style='text-align: center; font-family: Arial; font-size: 36px; font-weight: bold;'>DR GPT - Your  🤖</h1>",
        unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='display: flex; align-items: flex-start; margin: 10px 0; justify-content: flex-end;'>
                    <div style='text-align: left; padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                        {msg['content']}
                    </div>
                    <img src="https://img.icons8.com/ios-filled/50/000000/user.png" width="30" height="30" style='margin-left: 10px;' />
                </div>
                """,
                unsafe_allow_html=True
            )
        elif msg["role"] == "assistant":
            st.markdown(
                f"""
                <div style='display: flex; align-items: flex-start; margin: 10px 0;'>
                    <div style='text-align: left; padding: 10px; background-color: #e0e0e0; border-radius: 5px; color: black;'>
                        {msg['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Chat input
    if prompt := st.chat_input("What would you like to know about pregnancy, childbirth, or gynecology?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Dr. GPT is thinking... 🤔"):
            start_time = time.time()

            # Pass the user input to the selected function
            if st.session_state["selected_mode"] == "RAG":
                response = handle_rag(prompt)
            elif st.session_state["selected_mode"] == "Fine Tune":
                response = handle_fine_tune(prompt)
            elif st.session_state["selected_mode"] == "Base":
                response = handle_base(prompt)
            else:
                st.warning("Please select a mode from the sidebar.")
                return

            response_time = time.time() - start_time

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(
            f"""
            <div style='display: flex; align-items: flex-start; margin: 10px 0;'>
                <div style='text-align: left; padding: 10px; background-color: #e0e0e0; border-radius: 5px; color: black;'>
                    {response}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='text-align: right; color: #888;'>Response time: {response_time:.2f}s</p>",
                    unsafe_allow_html=True)




if __name__ == "__main__":
    main()

