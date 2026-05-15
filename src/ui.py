import streamlit as st


def setup_ui():

    st.set_page_config(
        page_title="Arabic RAG Chatbot",
        layout="wide"
    )

    st.title("Arabic RAG Chatbot")


def user_input():

    return st.chat_input("Ask your question...")


def display_response(role,
                     content):

    with st.chat_message(role):

        st.write(content)


def display_chunks(docs):

    st.sidebar.title("Retrieved Chunks")

    for doc, score in docs:

        st.sidebar.write(
            f"Episode: {doc.metadata['episode']}"
        )

        st.sidebar.write(
            f"Score: {score}"
        )

        st.sidebar.write(doc.page_content)

        st.sidebar.divider()