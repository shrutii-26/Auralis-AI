import librosa
import numpy as np
import os
from state import AgentState


def classify_emotion(pitch_mean, energy_mean, speaking_rate, mfcc_var):
    if energy_mean > 0.05 and pitch_mean > 180:
        if speaking_rate > 4.0:
            return "anger", 0.82, 0.85
        else:
            return "excitement", 0.74, 0.70
    elif energy_mean < 0.01 and pitch_mean < 130:
        return "sadness", 0.71, 0.60
    elif 130 <= pitch_mean <= 180 and 0.01 <= energy_mean <= 0.04:
        return "neutral", 0.88, 0.20
    elif pitch_mean > 200 and speaking_rate > 3.5:
        return "frustration", 0.78, 0.75
    else:
        return "neutral", 0.65, 0.30


def emotion_agent(state: AgentState) -> dict:
    """Returns ONLY the fields it updates — required for parallel execution."""
    print("[Emotion Agent] Analyzing audio features...")

    audio_path = state.get("audio_path")
    retry_count = state.get("emotion_retry_count", 0)

    if not audio_path or not os.path.exists(audio_path):
        return {"error_message": f"Audio file missing for emotion analysis"}

    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.var(mfccs))

        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        pitch_mean = float(np.nanmean(f0)) if f0 is not None else 150.0

        rms = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(rms))

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        speaking_rate = float(np.mean(onset_env))

        emotion, confidence, intensity = classify_emotion(
            pitch_mean, energy_mean, speaking_rate, mfcc_var
        )

        print(
            f"[Emotion Agent] Detected: {emotion} (conf={confidence:.2f}, retry={retry_count})"
        )

        return {
            "emotion_result": {
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "intensity": round(intensity, 2),
                "pitch_mean": round(pitch_mean, 2),
                "energy_mean": round(energy_mean, 4),
                "speaking_rate": round(speaking_rate, 2),
            }
        }

    except Exception as e:
        return {"error_message": f"Emotion analysis failed: {str(e)}"}
