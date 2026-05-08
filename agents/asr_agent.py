import whisper
import os
from state import AgentState

_model = None


def get_whisper_model():
    global _model
    if _model is None:
        model_name = os.getenv("WHISPER_MODEL", "base")
        print(f"[ASR Agent] Loading Whisper model: {model_name}")
        _model = whisper.load_model(model_name)
    return _model


def asr_agent(state: AgentState) -> dict:
    """Returns only changed fields for clean state updates."""
    print("[ASR Agent] Starting transcription...")

    audio_path = state.get("audio_path")

    if not audio_path or not os.path.exists(audio_path):
        return {
            "error_message": f"Audio file not found: {audio_path}",
        }

    try:
        model = get_whisper_model()
        result = model.transcribe(audio_path, language="en")

        transcript = result["text"].strip()

        # Better confidence calculation — average across all segments
        segments = result.get("segments", [])
        if segments:
            avg_log_prob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(
                segments
            )
            # no_speech_prob — if high, the segment is likely silence
            avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(
                segments
            )
            # Map log prob from [-1, 0] to [0, 1] and penalize silence
            confidence = min(max((avg_log_prob + 1.0), 0.0), 1.0) * (
                1.0 - avg_no_speech
            )
        else:
            confidence = 0.0

        # If transcript is empty or too short, low confidence
        if not transcript or len(transcript.split()) < 3:
            confidence = min(confidence, 0.2)

        print(f"[ASR Agent] Transcript: {transcript}")
        print(f"[ASR Agent] Confidence: {confidence:.2f}")

        return {
            "transcript": transcript,
            "asr_confidence": round(confidence, 2),
        }

    except Exception as e:
        return {
            "error_message": f"ASR failed: {str(e)}",
        }
