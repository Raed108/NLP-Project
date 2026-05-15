from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)


def full_memory():

    return ConversationBufferMemory(
        return_messages=True,
        chat_memory=ChatMessageHistory()
    )


def sliding_window_memory():

    return ConversationBufferWindowMemory(
        k=4,
        return_messages=True,
        chat_memory=ChatMessageHistory()
    )


def summary_memory(llm):

    return ConversationSummaryMemory(
        llm=llm,
        return_messages=True,
        chat_memory=ChatMessageHistory()
    )


def truncation_memory(messages,
                      max_messages=4):

    return messages[-max_messages:]