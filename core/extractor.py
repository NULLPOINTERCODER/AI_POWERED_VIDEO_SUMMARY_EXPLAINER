# Action items, decisions, and questions extractor

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    model = os.getenv("MISTRAL_MODEL", "open-mistral-7b")
    return ChatMistralAI(
        model=model,
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def _safe_invoke(chain, transcript: str, fallback: str) -> str:
    """Invoke a chain, returning fallback message on empty/error."""
    if not transcript or not transcript.strip():
        return fallback
    try:
        return chain.invoke(transcript.strip())
    except Exception as e:
        print(f"[extractor] Error: {e}", flush=True)
        return f"{fallback} (Error: {e})"


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return _safe_invoke(chain, transcript, "No action items found.")


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return _safe_invoke(chain, transcript, "No key decisions found.")


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return _safe_invoke(chain, transcript, "No open questions found.")