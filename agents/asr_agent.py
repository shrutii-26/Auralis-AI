import os
from groq import Groq
from state import AgentState

_client = None


def get_groq_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def asr_agent(state: AgentState) -> dict:
    """Returns only changed fields for clean state updates."""
    print("[ASR Agent] Starting transcription via Groq Whisper...")

    audio_path = state.get("audio_path")

    if not audio_path or not os.path.exists(audio_path):
        return {"error_message": f"Audio file not found: {audio_path}"}

    try:
        client = get_groq_client()
        model_name = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")

        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file),
                model=model_name,
                language="en",
                response_format="verbose_json",
            )

        transcript = result.text.strip()

        # Groq Whisper verbose_json returns segments with avg_logprob
        segments = getattr(result, "segments", []) or []
        if segments:
            avg_log_prob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(
                segments
            )
            avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(
                segments
            )
            confidence = min(max((avg_log_prob + 1.0), 0.0), 1.0) * (
                1.0 - avg_no_speech
            )
        else:
            # Groq doesn't always return segments — fall back to length-based heuristic
            confidence = 0.8 if transcript and len(transcript.split()) >= 3 else 0.2

        if not transcript or len(transcript.split()) < 3:
            confidence = min(confidence, 0.2)

        print(f"[ASR Agent] Transcript: {transcript}")
        print(f"[ASR Agent] Confidence: {confidence:.2f}")

        return {
            "transcript": transcript,
            "asr_confidence": round(confidence, 2),
        }

    except Exception as e:
        return {"error_message": f"ASR failed: {str(e)}"}
