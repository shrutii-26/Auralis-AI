import os
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

RESPONSE_SYSTEM_PROMPT = """You are an empathetic customer support assistant.

Given the user's transcript, detected emotion, and intent, generate a helpful and appropriate response.

Rules:
- If emotion is anger or frustration, acknowledge it first before addressing the issue
- Be concise (2-4 sentences max)
- Be human and warm, not robotic
- Do NOT mention that you are an AI unless asked

Respond ONLY in JSON:
{
  "response": "<your response to the user>",
  "internal_note": "<one sentence note for human agents if they take over>"
}"""


def action_agent(state: AgentState) -> dict:
    """Returns only changed fields."""
    print("[Action Agent] Generating final output...")

    final_action = state.get("final_action", "respond")
    priority = state.get("priority", "low")
    transcript = state.get("transcript", "")
    emotion_result = state.get("emotion_result", {})
    intent_result = state.get("intent_result", {})
    escalation_risk = state.get("escalation_risk", 0.0)
    session_history = state.get("session_history", [])

    # --- ESCALATION PATH ---
    if final_action == "escalate":
        print("[Action Agent] Escalating to human operator...")
        history_summary = ""
        if session_history:
            emotions_seen = [t.get("emotion") for t in session_history]
            history_summary = f"Emotion progression: {' -> '.join(emotions_seen)}"

        return {
            "response_text": "I understand this has been frustrating. Let me connect you with a specialist right away.",
            "final_action": "escalate",
            "priority": "high",
        }

    # --- RESPOND PATH ---
    try:
        model_name = os.getenv("RESPONSE_MODEL", "llama3:latest")
        llm = ChatOllama(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.7,
        )

        context = f"""
Transcript: {transcript}
Detected Emotion: {emotion_result.get('emotion', 'neutral')} (intensity: {emotion_result.get('intensity', 0)})
User Intent: {intent_result.get('intent', 'inquiry')}
Summary: {intent_result.get('summary', '')}
"""
        messages = [
            SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        response_text = parsed.get(
            "response", "Thank you for reaching out. How can I help you?"
        )

        print(f"[Action Agent] Response generated (priority={priority})")
        return {"response_text": response_text}

    except Exception as e:
        print(f"[Action Agent] LLM response failed: {e}")
        return {
            "response_text": "Thank you for reaching out. A team member will assist you shortly."
        }
