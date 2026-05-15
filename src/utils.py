import time


def calculate_latency(start_time):

    return time.time() - start_time


def format_context(docs):

    context = ""

    for doc, score in docs:

        context += doc.page_content + "\n\n"

    return context