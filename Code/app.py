import time

import streamlit as st


def generate_response(msg):
    return "demo"


def main():
    st.set_page_config(page_title="DR GPT", page_icon="🤖", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    .stButton > button {
        background-color: #333333;
        color: white;
        border: none;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #333333;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .sidebar .css-1d391kg {
        padding-top: 1rem;
        background-color: #333333;
    }
    .stExpander {
        background-color: #333333;
    }
    .css-1v0mbdj p, .css-1v0mbdj h2 {
        color: #ffffff;
    }
    .css-2trqyj {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## About DR GPT")
        st.write(
            "DR GPT is a Gynecology and Obstetrics chatbot that can answer questions about pregnancy, childbirth")

        # Display tips
        with st.expander("Tips for using DR GPT"):
            st.info("""
            - Ask questions related to pregnancy, childbirth, and gynecology.
            - Be polite and respectful.
            - Avoid sharing personal information.
            - If you encounter any issues, please report them.
            """)

    # Main chat interface
    st.markdown(
        "<h1 style='text-align: center; font-family: Arial; font-size: 36px; font-weight: bold; color: #FFFFFF;'>DR GPT - Your  🤖</h1>",
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
                    <div style='text-align: left; padding: 10px; background-color: #333333; border-radius: 5px;'>
                        {msg['content']}
                    </div>
                    <img src="https://img.icons8.com/ios-filled/50/ffffff/user.png" width="30" height="30" style='margin-left: 10px;'/>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif msg["role"] == "assistant":
            st.markdown(
                f"""
                <div style='display: flex; align-items: flex-start; margin: 10px 0;'>
                    <div style='text-align: left; padding: 10px; background-color: #444444; border-radius: 5px; color: white;'>
                        {msg['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='display: flex; align-items: flex-start; margin: 10px 0;'>
                    <div style='text-align: center; padding: 10px; background-color: #444444; border-radius: 5px; color: white;'>
                        {msg['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Chat input
    if prompt := st.chat_input("What would you like to know about pregnancy, childbirth, or gynecology?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(
            f"""
            <div style='display: flex; align-items: flex-start; margin: 10px 0; justify-content: flex-end;'>
                <div style='text-align: left; padding: 10px; background-color: #333333; border-radius: 5px;'>
                    {prompt}
                </div>
                <img src="https://img.icons8.com/ios-filled/50/ffffff/user.png" width="30" height="30" style='margin-left: 10px;'/>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.spinner("Dr. GPT is thinking... 🤔"):
            start_time = time.time()
            msg = prompt
            response = generate_response(msg)
            response_time = time.time() - start_time

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(
            f"""
            <div style='display: flex; align-items: flex-start; margin: 10px 0;'>
                <div style='text-align: left; padding: 10px; background-color: #444444; border-radius: 5px; color: white;'>
                    {response}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='text-align: right; color: #888;'>Response time: {response_time:.2f}s</p>",
                    unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
