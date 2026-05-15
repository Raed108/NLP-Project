from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)


def full_memory():

    return ConversationBufferMemory(
        return_messages=True
    )


def sliding_window_memory():

    return ConversationBufferWindowMemory(
        k=4,
        return_messages=True
    )


def summary_memory(llm):

    return ConversationSummaryMemory(
        llm=llm,
        return_messages=True
    )