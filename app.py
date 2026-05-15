import time

from src.retrieval import RetrieverSystem
from src.llm import LLMClient
# from cache import SemanticCache

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

# cache = SemanticCache(
#     threshold=0.85
# )


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

    # cache_result = cache.search_cache(query)

    # if cache_result["hit"]:

    #     response = cache_result["response"]

    #     st.sidebar.success(
    #         f"Cache Hit ({cache_result['similarity']:.2f})"
    #     )

    # else:

    docs = retriever.retrieve(query)
    display_chunks(docs)
    context = format_context(docs)
    prompt = STRICT_ARABIC_PROMPT.format(
        context=context,
        question=query
    )
    response = llm.generate(prompt)
    # cache.add_to_cache(
    #     query,
    #     response
    # )

    latency = calculate_latency(start)

    display_response(
        "assistant",
        response
    )

    st.sidebar.write(
        f"Latency: {latency:.2f}s"
    )

    # st.sidebar.write(
    #     f"Cache Hit Rate: {cache.hit_rate():.2f}"
    # )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })