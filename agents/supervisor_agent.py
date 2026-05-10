from state import AgentState

# ── Thresholds ──────────────────────────────────
ASR_CONFIDENCE_THRESHOLD = 0.3
ESCALATION_RISK_THRESHOLD = 0.75
ANGER_INTENSITY_THRESHOLD = 0.80


def supervisor_agent(state: AgentState) -> AgentState:
    """
    Central routing brain. Called at entry and after ASR + Memory.
    Emotion retry logic lives in fan_in — supervisor stays clean.

    Routing decisions:
    0. Error detected        → error node (breaks infinite loops)
    1. No transcript         → ASR
    2. Low ASR confidence    → early exit to action (clarify)
    3. No emotion/intent     → parallel_dispatch (runs both simultaneously)
    4. Memory not updated    → memory
    5. Everything ready      → action (with priority decision)
    """

    # ── Step 0: Error check first — breaks any infinite loop ──
    if state.get("error_message"):
        print(f"[Supervisor] Error detected: {state.get('error_message')}")
        return {**state, "next_step": "error"}

    transcript = state.get("transcript")
    asr_confidence = state.get("asr_confidence")
    emotion_result = state.get("emotion_result")
    intent_result = state.get("intent_result")
    escalation_risk = state.get("escalation_risk", 0.0)
    session_history = state.get("session_history")

    print(
        f"[Supervisor] State check — transcript={bool(transcript)}, "
        f"emotion={bool(emotion_result)}, intent={bool(intent_result)}, "
        f"memory={'done' if session_history is not None else 'pending'}"
    )

    # ── Step 1: Need transcription ───────────────
    if not transcript:
        return {**state, "next_step": "asr"}

    # ── Step 2: ASR confidence too low ──────────
    if asr_confidence is not None and asr_confidence < ASR_CONFIDENCE_THRESHOLD:
        return {
            **state,
            "next_step": "action",
            "final_action": "clarify",
            "priority": "medium",
            "response_text": "I couldn't understand the audio clearly. Could you please repeat?",
        }

    # ── Step 3: Dispatch emotion + intent in parallel ──
    if not emotion_result or not intent_result:
        return {**state, "next_step": "parallel_dispatch"}

    # ── Step 4: Update memory ────────────────────
    if session_history is None:
        return {**state, "next_step": "memory"}

    # ── Step 5: Final routing ────────────────────
    emotion = emotion_result.get("emotion", "neutral")
    intensity = emotion_result.get("intensity", 0.0)
    intent = intent_result.get("intent", "inquiry")

    # HIGH — escalate to human
    if (
        escalation_risk >= ESCALATION_RISK_THRESHOLD
        or intent == "escalation_request"
        or (
            emotion in {"anger", "frustration"}
            and intensity >= ANGER_INTENSITY_THRESHOLD
        )
    ):
        return {
            **state,
            "next_step": "action",
            "final_action": "escalate",
            "priority": "high",
        }

    # MEDIUM — complaint or frustration
    elif emotion == "frustration" or intent == "complaint":
        return {
            **state,
            "next_step": "action",
            "final_action": "respond",
            "priority": "medium",
        }

    # LOW — standard
    else:
        return {
            **state,
            "next_step": "action",
            "final_action": "respond",
            "priority": "low",
        }


def route_from_supervisor(state: AgentState) -> str:
    """LangGraph conditional edge — reads next_step set by supervisor."""
    return state.get("next_step", "error")


def route_from_fan_in(state: AgentState) -> str:
    """
    LangGraph conditional edge for fan_in node.
    fan_in sets next_step to either 'emotion_retry' or 'memory'.
    """
    return state.get("next_step", "memory")
