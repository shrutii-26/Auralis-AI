import os
import sys
import numpy as np
import librosa
import joblib
from state import AgentState


# ── Find emotion_model.pkl — try multiple paths ──
def find_model():
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "emotion_model.pkl",
        ),
        os.path.join(os.getcwd(), "emotion_model.pkl"),
        os.path.join(
            os.path.dirname(sys.argv[0]) if sys.argv[0] else ".", "emotion_model.pkl"
        ),
        "emotion_model.pkl",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[Emotion Agent] Found model at: {path}")
            return path
    print(f"[Emotion Agent] No model found. Searched: {candidates}")
    return None


MODEL_PATH = find_model()
_model_data = None


def get_model():
    global _model_data
    if _model_data is None and MODEL_PATH is not None:
        try:
            print(f"[Emotion Agent] Loading trained model from {MODEL_PATH}")
            _model_data = joblib.load(MODEL_PATH)
            print(
                f"[Emotion Agent] Model loaded successfully! Classes: {list(_model_data['label_encoder'].classes_)}"
            )
        except Exception as e:
            print(f"[Emotion Agent] Failed to load model: {e}")
            _model_data = None
    return _model_data


def extract_features(y, sr):
    """Extract the same 34 features used during training."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    f0, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    pitch_mean = (
        float(np.nanmean(f0)) if f0 is not None and not np.all(np.isnan(f0)) else 0.0
    )
    pitch_std = (
        float(np.nanstd(f0)) if f0 is not None and not np.all(np.isnan(f0)) else 0.0
    )

    rms = librosa.feature.rms(y=y)
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    speaking_rate = float(np.mean(onset_env))

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            [pitch_mean, pitch_std],
            [energy_mean, energy_std],
            [speaking_rate],
            [spectral_centroid],
            [spectral_rolloff],
            [zcr],
        ]
    )
    return features, pitch_mean, energy_mean, speaking_rate


def classify_rule_based(pitch_mean, energy_mean, speaking_rate):
    """Fallback rule-based classifier."""
    if energy_mean > 0.05 and pitch_mean > 180:
        if speaking_rate > 4.0:
            return "angry", 0.82, 0.85
        return "excited", 0.74, 0.70
    elif energy_mean < 0.01 and pitch_mean < 130:
        return "sad", 0.71, 0.60
    elif 130 <= pitch_mean <= 180 and 0.01 <= energy_mean <= 0.04:
        return "neutral", 0.88, 0.20
    elif pitch_mean > 200 and speaking_rate > 3.5:
        return "angry", 0.78, 0.75
    else:
        return "neutral", 0.65, 0.30


INTENSITY_MAP = {
    "angry": 0.85,
    "fearful": 0.70,
    "disgust": 0.65,
    "sad": 0.60,
    "surprised": 0.55,
    "happy": 0.50,
    "calm": 0.20,
    "neutral": 0.20,
}


def emotion_agent(state: AgentState) -> dict:
    """Detects emotion using trained RAVDESS model or rule-based fallback."""
    print("[Emotion Agent] Analyzing audio features...")

    audio_path = state.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        return {"error_message": "Audio file missing for emotion analysis"}

    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=5)
        features, pitch_mean, energy_mean, speaking_rate = extract_features(y, sr)

        model_data = get_model()

        if model_data is not None:
            model = model_data["model"]
            le = model_data["label_encoder"]

            proba = model.predict_proba([features])[0]
            pred_idx = np.argmax(proba)
            emotion = le.inverse_transform([pred_idx])[0]
            confidence = float(proba[pred_idx])
            intensity = INTENSITY_MAP.get(emotion, 0.30)

            print(f"[Emotion Agent] TRAINED MODEL -> {emotion} (conf={confidence:.2f})")
        else:
            emotion, confidence, intensity = classify_rule_based(
                pitch_mean, energy_mean, speaking_rate
            )
            print(f"[Emotion Agent] RULE-BASED -> {emotion} (conf={confidence:.2f})")

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
