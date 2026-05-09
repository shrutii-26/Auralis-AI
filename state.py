from typing import Optional, List
from typing_extensions import TypedDict


class EmotionResult(TypedDict):
    emotion: str  # e.g. "anger", "neutral", "happiness"
    confidence: float  # 0.0 - 1.0
    intensity: float  # 0.0 - 1.0
    pitch_mean: float
    energy_mean: float
    speaking_rate: float


class IntentResult(TypedDict):
    intent: str  # e.g. "complaint", "inquiry", "feedback"
    confidence: float
    summary: str


class AgentState(TypedDict):
    # --- Input ---
    audio_path: str
    session_id: str

    # --- ASR Output ---
    transcript: Optional[str]
    asr_confidence: Optional[float]

    # --- Emotion Output ---
    emotion_result: Optional[EmotionResult]
    emotion_retry_count: int  # tracks reflection loop

    # --- Intent Output ---
    intent_result: Optional[IntentResult]

    # --- Memory ---
    session_history: List[dict]  # previous turns in this session
    escalation_risk: float  # 0.0 - 1.0, rises over conversation

    # --- Supervisor Control ---
    next_step: Optional[str]  # supervisor routes to this
    error_message: Optional[str]

    # --- Final Output ---
    final_action: Optional[str]  # "escalate" | "respond" | "clarify"
    response_text: Optional[str]
    priority: Optional[str]  # "high" | "medium" | "low"
