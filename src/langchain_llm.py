from langchain_openai import ChatOpenAI
import os
import dotenv


dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_langchain_llm(model_name):

    if model_name == "mistral-7b-instruct":
        llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0.3
        )
    elif model_name == "deepseek-v4-flash":
        llm = ChatOpenAI(
            model="deepseek/deepseek-v4-flash:free",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0.3
        )

    return llm