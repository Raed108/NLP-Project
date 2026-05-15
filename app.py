import time

from src.retrieval import RetrieverSystem
from src.llm import LLMClient

from src.prompts import STRICT_ARABIC_PROMPT

from src.utils import (
    format_context,
    calculate_latency
)

from src.ui import (
    setup_ui,
    user_input,
    display_response,
    display_chunks
)


import streamlit as st


setup_ui()

retriever = RetrieverSystem()

llm = LLMClient()


if "messages" not in st.session_state:

    st.session_state.messages = []


for msg in st.session_state.messages:

    display_response(
        msg["role"],
        msg["content"]
    )


query = user_input()

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    display_response("user", query)

    start = time.time()

    docs = retriever.retrieve(query)
    display_chunks(docs)
    context = format_context(docs)
    prompt = STRICT_ARABIC_PROMPT.format(
        context=context,
        question=query
    )
    response = llm.generate(prompt)

    latency = calculate_latency(start)

    display_response(
        "assistant",
        response
    )

    st.sidebar.write(
        f"Latency: {latency:.2f}s"
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })