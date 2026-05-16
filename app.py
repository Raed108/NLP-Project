import time

from src.retrieval import RetrieverSystem
from src.llm import LLMClient
from src.memory import sliding_window_memory, summary_memory
from src.cache import SemanticCache
from src.langchain_llm import get_langchain_llm
import streamlit as st
from src.preprocessing import normalize_arabic


from src.prompts import STRICT_ARABIC_PROMPT, STRICT_ENGLISH_PROMPT

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


setup_ui()

if "retriever" not in st.session_state:

    st.session_state.retriever = RetrieverSystem()

retriever = st.session_state.retriever

# Model selector in sidebar
MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "deepseek-v4-flash",
    "mistralai/devstral-2512",
]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_OPTIONS[0]

st.sidebar.selectbox("Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.selected_model), key="selected_model")

llm = LLMClient(model=st.session_state.selected_model)

if "langchain_llm" not in st.session_state:

    st.session_state.langchain_llm = get_langchain_llm(
        "deepseek-v4-flash"
    )

langchain_llm = st.session_state.langchain_llm

if "memory" not in st.session_state:

    st.session_state.memory = summary_memory(
        langchain_llm
    )

memory = st.session_state.memory

if "cache" not in st.session_state:

    st.session_state.cache = SemanticCache(
        threshold=0.85
    )

cache = st.session_state.cache


if "messages" not in st.session_state:

    st.session_state.messages = []


for msg in st.session_state.messages:

    display_response(
        msg["role"],
        msg["content"]
    )


query = user_input()    

normalized_query = normalize_arabic(query)

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    display_response("user", query)

    memory.chat_memory.add_user_message(query)

    start = time.time()

    cache_result = cache.search_cache(
        normalized_query
    )



    if cache_result["hit"]:

        response = cache_result["response"]

        st.sidebar.success(
            f"Cache Hit ({cache_result['similarity']:.2f})"
        )

    else:

        docs = retriever.retrieve(normalized_query)
        
        display_chunks(docs)

        context = format_context(docs)

        chat_history = memory.load_memory_variables({})

        history_text = chat_history["history"]
        context = format_context(docs)

        # language detection (simple heuristic): prefer English-only prompts
        def detect_language(text: str) -> str:
            arabic_count = sum(1 for ch in text if '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' or '\u08A0' <= ch <= '\u08FF' or '\uFB50' <= ch <= '\uFDFF' or '\uFE70' <= ch <= '\uFEFF')
            latin_count = sum(1 for ch in text if ('a' <= ch.lower() <= 'z'))
            if latin_count > 0 and arabic_count == 0:
                return 'en'
            if arabic_count > 0 and latin_count == 0:
                return 'ar'
            return 'mixed'

        lang = detect_language(query)

        if lang == 'en':
            prompt = STRICT_ENGLISH_PROMPT.format(
                context=context, 
                question=query, 
                history_text=history_text
            )
        else:
            prompt = STRICT_ARABIC_PROMPT.format(
                context=context,
                question=query,
                history_text=history_text
            )

        response, used_model, attempts = llm.generate(prompt)

        # show fallback information if model used differs from selected
        if used_model != "none" and used_model != st.session_state.selected_model:
            st.sidebar.warning(f"Fell back to model: {used_model}")

        # show retry/attempt count in the sidebar
        if attempts and attempts > 1:
            st.sidebar.info(f"LLM attempts: {attempts}")

        memory.chat_memory.add_ai_message(response)

        cache.add_to_cache(
            normalized_query,
            response
        )

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


    st.sidebar.write(
        f"Cache Hit Rate: {cache.hit_rate():.2f}"
    )

    st.sidebar.write(
        f"Saved LLM Calls: {cache.saved_model_calls()}"
    )

    st.sidebar.write(
        f"Cache Size: {len(cache.cache)}"
    )