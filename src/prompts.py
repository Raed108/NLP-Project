MINIMAL_PROMPT = """
Answer the question using the context.

Context:
{context}

Question:
{question}
"""


STRICT_ARABIC_PROMPT = """
أنت مساعد عربي ذكي.

أجب فقط باستخدام المعلومات الموجودة في السياق.

إذا لم تجد الإجابة داخل السياق قل:
"لا أملك معلومات كافية للإجابة."

لا تخترع معلومات.

تاريخ المحادثة:
{history_text}

السياق:
{context}

السؤال:
{question}
"""

# edited the english prompt so it enforces english reposnses
STRICT_ENGLISH_PROMPT = """
You are a grounded AI assistant.

Answer ONLY from the provided context.

answer in English only,
use the Arabic context as evidence,
mentally translate the needed facts,

If the answer does not exist in the context say:
"I do not have enough information."

Do not hallucinate.

Conversation history:
{history_text}

Context:
{context}

Question:
{question}
"""