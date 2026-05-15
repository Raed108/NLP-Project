import os
import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    def __init__(self, model="google"):
        self.model_name = model

        if model == "deepseek-v4-flash":
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is required for the deepseek-v4-flash model.")
            self.model = ChatOpenAI(
                model="deepseek/deepseek-v4-flash:free",
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )
        elif model == "mistralai/devstral-2512":
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is required for the mistralai/devstral-2512 model.")
            self.model = ChatOpenAI(
                model="mistralai/devstral-2512:free",
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )
        else:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for the gemini-2.5-flash model.")
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                max_retries=0,
                temperature=0.2,
                api_key=GOOGLE_API_KEY,
            )

    def generate(self, prompt):
        response = self.model.invoke(prompt)
        return response.content
