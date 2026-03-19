"""Lightweight intent routing before the full policy RAG pipeline."""

from __future__ import annotations

import json
from typing import Any, TypedDict

import cohere

from .app_config import CHAT_MODEL

INTENT_GATE_MODEL = CHAT_MODEL
INTENT_GATE_MAX_TOKENS = 140

SUPPORTED_ROUTES = {
    "policy_question",
    "needs_clarification",
    "non_policy",
    "greeting",
    "thanks",
    "goodbye",
    "capability",
}


class IntentDecision(TypedDict):
    route: str
    clarifying_question: str


def _format_recent_history(chat_history: list[dict[str, Any]] | None) -> str:
    lines: list[str] = []
    for item in list(chat_history or [])[-4:]:
        role = str(item.get("role", "")).strip().upper()
        message = str(item.get("message", "")).strip()
        if role and message:
            lines.append(f"{role}: {message}")
    return "\n".join(lines) if lines else "(none)"


def _default_decision(route: str = "policy_question", clarifying_question: str = "") -> IntentDecision:
    return {
        "route": route if route in SUPPORTED_ROUTES else "policy_question",
        "clarifying_question": clarifying_question.strip(),
    }


def _parse_classifier_decision(raw_text: str) -> IntentDecision:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return _default_decision()

    route = str(payload.get("route") or payload.get("label") or "").strip()
    clarifying_question = str(payload.get("clarifying_question", "")).strip()
    if route not in SUPPORTED_ROUTES:
        return _default_decision()
    return _default_decision(route=route, clarifying_question=clarifying_question)


def build_intent_reply(route: str, language: str, clarifying_question: str = "") -> str:
    use_french = str(language or "").strip().lower() == "fr"
    messages = {
        "greeting": {
            "en": "Hello. I'm PolicyAI. Ask me a procurement policy question and I'll look for guidance in the policy documents.",
            "fr": "Bonjour. Je suis PolicyAI. Posez-moi une question sur les politiques d'approvisionnement et je chercherai la guidance pertinente dans les documents.",
        },
        "thanks": {
            "en": "You're welcome. If you have a procurement policy question, send it over and I'll help you work through it.",
            "fr": "Avec plaisir. Si vous avez une question sur les politiques d'approvisionnement, envoyez-la-moi et je vous aiderai.",
        },
        "goodbye": {
            "en": "Goodbye. Come back anytime if you need help with a procurement policy scenario.",
            "fr": "Au revoir. Revenez quand vous voulez si vous avez besoin d'aide pour un scenario de politique d'approvisionnement.",
        },
        "capability": {
            "en": "I help with procurement policy questions using the policy documents loaded into this system. Ask about approvals, solicitation methods, documentation requirements, late offers, supply arrangements, taxes and duties, or similar topics.",
            "fr": "Je peux aider avec des questions sur les politiques d'approvisionnement a partir des documents charges dans ce systeme. Vous pouvez demander des renseignements sur les approbations, les methodes de sollicitation, la documentation requise, les offres tardives, les arrangements d'approvisionnement, les taxes et droits, ou des sujets semblables.",
        },
        "needs_clarification": {
            "en": "I can help with procurement policy questions, but I need one more detail before I search the policy documents.",
            "fr": "Je peux aider avec les questions sur les politiques d'approvisionnement, mais j'ai besoin d'un detail de plus avant de chercher dans les documents de politique.",
        },
        "non_policy": {
            "en": "I'm focused on procurement policy guidance in this project. If you have a procurement or contracting policy question, ask me about the rule, scenario, or document you need help with.",
            "fr": "Je suis specialise dans la guidance sur les politiques d'approvisionnement pour ce projet. Si vous avez une question sur l'approvisionnement ou les politiques contractuelles, demandez-moi la regle, le scenario ou le document qui vous interesse.",
        },
    }
    locale = "fr" if use_french else "en"
    base_message = messages.get(route, messages["needs_clarification"])[locale]
    if route == "needs_clarification" and clarifying_question:
        return f"{base_message}\n\n{clarifying_question}"
    return base_message


# Route all non-empty chat through a compact LLM classifier so greetings,
# capability questions, broad procurement prompts, and concrete policy
# questions all use the same decision logic.
def classify_message_intent(
    client: cohere.Client,
    message: str,
    chat_history: list[dict[str, Any]] | None = None,
) -> IntentDecision:
    classifier_prompt = (
        "You classify messages for a procurement policy assistant.\n"
        "Return strict JSON with keys: "
        "{\"route\": \"policy_question\" | \"needs_clarification\" | \"non_policy\" | "
        "\"greeting\" | \"thanks\" | \"goodbye\" | \"capability\", "
        "\"clarifying_question\": string}.\n\n"
        "Choose greeting for hello/hi style openers.\n"
        "Choose thanks for gratitude like thanks or thank you.\n"
        "Choose goodbye for goodbye or sign-off style messages.\n"
        "Choose capability when the user is asking what the assistant can help with or what kinds of questions they can ask.\n"
        "Choose policy_question when the user gives a concrete enough procurement, contracting, approval, solicitation, documentation, tax, duty, compliance, or policy question to search directly.\n"
        "Choose needs_clarification when the user seems to be asking about procurement policy, but the request is too broad, underspecified, or missing one key scenario detail to answer well.\n"
        "Choose non_policy for unrelated small talk or requests outside procurement policy scope.\n\n"
        "If route is needs_clarification, provide exactly one short clarifying question that asks for the most important missing detail.\n"
        "For greeting, thanks, goodbye, capability, policy_question, and non_policy, set clarifying_question to an empty string.\n\n"
        "Examples:\n"
        "- \"hi\" -> {\"route\":\"greeting\",\"clarifying_question\":\"\"}\n"
        "- \"thanks\" -> {\"route\":\"thanks\",\"clarifying_question\":\"\"}\n"
        "- \"what kinds of questions can i ask\" -> {\"route\":\"capability\",\"clarifying_question\":\"\"}\n"
        "- \"what approvals are required before award\" -> {\"route\":\"policy_question\",\"clarifying_question\":\"\"}\n"
        "- \"can you help me with procurement\" -> {\"route\":\"needs_clarification\",\"clarifying_question\":\"What procurement scenario or rule do you want help with?\"}\n"
        "- \"what taxes do i need to consider\" -> {\"route\":\"needs_clarification\",\"clarifying_question\":\"Are you asking about bid pricing, contract payment, or supplier tax obligations?\"}\n"
        "- \"tell me a joke\" -> {\"route\":\"non_policy\",\"clarifying_question\":\"\"}\n\n"
        f"Recent conversation:\n{_format_recent_history(chat_history)}\n\n"
        f"User message:\n{message}\n"
    )

    try:
        response = client.chat(
            model=INTENT_GATE_MODEL,
            message=classifier_prompt,
            temperature=0.0,
            max_tokens=INTENT_GATE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception:
        return _default_decision()

    return _parse_classifier_decision((response.text or "").strip())
