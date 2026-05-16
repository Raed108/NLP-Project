import os
import time
import logging
import dotenv
from typing import Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Order of preference for primary model + fallbacks
FALLBACK_CHAIN = [
    "gemini-2.5-flash",
    "deepseek-v4-flash",
    "mistralai/devstral-2512",
]


class LLMClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model_name = model
        self.model = None
        self._initialize_model(model)

    def _initialize_model(self, model: str):
        """Initialize the underlying model object for the given model name."""
        if model == "deepseek-v4-flash":
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is required for the deepseek-v4-flash model.")
            self.model = ChatOpenAI(
                model="deepseek/deepseek-v4-flash:free",
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )
            self.model_name = model

        elif model == "mistralai/devstral-2512":
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is required for the mistralai/devstral-2512 model.")
            self.model = ChatOpenAI(
                model="mistralai/devstral-2512:free",
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )
            self.model_name = model

        else:
            # default -> Gemini
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for the gemini-2.5-flash model.")
            # Set provider-side retries to 0 so we control retry behavior here
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                max_retries=0,
                temperature=0.2,
                api_key=GOOGLE_API_KEY,
            )
            self.model_name = "gemini-2.5-flash"

    def generate(self, prompt: str, max_retries: int = 2, enable_fallback: bool = True) -> Tuple[str, str]:
        """
        Generate a response for `prompt`.

        - `max_retries` controls in-process retry attempts for transient errors (Gemini max 2 recommended).
        - On quota (429 / ResourceExhausted) errors, we fast-fail and attempt fallback models if enabled.

        Returns (response_text, model_used).
        Never raises for common provider errors — returns a friendly error string instead.
        """
        # Try primary model with controlled retries
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.model.invoke(prompt)
                content = getattr(resp, "content", None) or str(resp)
                return content, self.model_name
            except Exception as e:
                last_exc = e
                err_txt = str(e)
                logger.warning(f"LLM attempt {attempt+1} for {self.model_name} failed: {err_txt}")
                # Fast-fail on quota / resource exhausted to avoid long provider-side backoff
                if "ResourceExhausted" in err_txt or "429" in err_txt or "quota" in err_txt.lower():
                    logger.warning("Detected quota/resource-exhausted error from provider; will attempt fallback immediately.")
                    break
                # simple backoff between attempts
                if attempt < max_retries:
                    time.sleep(1 * (2 ** attempt))
                    continue

        # If primary failed and fallback enabled, try the remaining models once each
        if enable_fallback:
            logger.info("Attempting fallback models...")
            for fallback_model in FALLBACK_CHAIN:
                if fallback_model == self.model_name:
                    continue
                try:
                    self._initialize_model(fallback_model)
                    resp = self.model.invoke(prompt)
                    content = getattr(resp, "content", None) or str(resp)
                    return content, self.model_name
                except Exception as e:
                    logger.warning(f"Fallback {fallback_model} failed: {e}")
                    last_exc = e

        # All attempts failed — do not raise; return a friendly message
        logger.error(f"All LLM attempts failed. Last error: {last_exc}")
        return (f"⚠️ All models are currently unavailable. Last error: {str(last_exc)}", "none")
