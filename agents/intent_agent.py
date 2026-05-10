import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

INTENT_SYSTEM_PROMPT = """You are an intent classification expert for a customer support system.

Given a transcript, classify the user's PRIMARY intent into one of these categories:
- complaint: User is unhappy about something
- inquiry: User wants information
- feedback: User is providing feedback (positive or neutral)
- request: User wants something done
- escalation_request: User explicitly wants to speak to a human

Respond ONLY in valid JSON, no extra text, no markdown:
{
  "intent": "<category>",
  "confidence": <float between 0 and 1>,
  "summary": "<one sentence summary of what the user wants>"
}"""


def intent_agent(state: AgentState) -> dict:
    """Returns ONLY the fields it updates — required for parallel execution."""
    print("[Intent Agent] Detecting intent via Groq...")

    transcript = state.get("transcript")
    if not transcript:
        return {"error_message": "No transcript available for intent detection"}

    try:
        llm = ChatGroq(
            model=os.getenv("INTENT_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )

        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=f"Transcript: {transcript}"),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        intent_result = {
            "intent": parsed.get("intent", "inquiry"),
            "confidence": float(parsed.get("confidence", 0.5)),
            "summary": parsed.get("summary", ""),
        }

        print(
            f"[Intent Agent] Intent: {intent_result['intent']} (conf={intent_result['confidence']:.2f})"
        )
        return {"intent_result": intent_result}

    except json.JSONDecodeError:
        return {
            "intent_result": {
                "intent": "inquiry",
                "confidence": 0.4,
                "summary": transcript[:100],
            }
        }

    except Exception as e:
        return {"error_message": f"Intent detection failed: {str(e)}"}
