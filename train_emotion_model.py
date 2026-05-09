"""
RAVDESS Emotion Model Trainer
Trains a Random Forest classifier on librosa features extracted from RAVDESS dataset.
Saves model as emotion_model.pkl for use in emotion_agent.py

Usage:
    python train_emotion_model.py --data_dir "path/to/audio_speech_actors_01-24"
"""

import os
import sys
import argparse
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# RAVDESS emotion mapping (from filename)
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def extract_features(file_path):
    """Extract audio features from a single file."""
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=5)

        # 1. MFCCs (13 coefficients — mean + std = 26 features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # 2. Pitch (F0)
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        pitch_mean = (
            np.nanmean(f0) if f0 is not None and not np.all(np.isnan(f0)) else 0.0
        )
        pitch_std = (
            np.nanstd(f0) if f0 is not None and not np.all(np.isnan(f0)) else 0.0
        )

        # 3. Energy (RMS)
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        # 4. Speaking rate (onset strength)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        speaking_rate = np.mean(onset_env)

        # 5. Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Combine all features into one vector
        features = np.concatenate(
            [
                mfcc_mean,  # 13
                mfcc_std,  # 13
                [pitch_mean, pitch_std],  # 2
                [energy_mean, energy_std],  # 2
                [speaking_rate],  # 1
                [spectral_centroid],  # 1
                [spectral_rolloff],  # 1
                [zcr],  # 1
            ]
        )  # Total: 34 features

        return features

    except Exception as e:
        print(f"  [SKIP] Error processing {file_path}: {e}")
        return None


def load_ravdess(data_dir):
    """Load all RAVDESS audio files and extract features + labels."""
    features = []
    labels = []
    total = 0
    skipped = 0

    for actor_dir in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue

        print(f"Processing {actor_dir}...")

        for filename in sorted(os.listdir(actor_path)):
            if not filename.endswith(".wav"):
                continue

            total += 1
            parts = filename.replace(".wav", "").split("-")

            # parts[2] is emotion code
            emotion_code = parts[2]
            emotion_label = RAVDESS_EMOTIONS.get(emotion_code, None)
            if emotion_label is None:
                skipped += 1
                continue

            file_path = os.path.join(actor_path, filename)
            feat = extract_features(file_path)

            if feat is not None:
                features.append(feat)
                labels.append(emotion_label)
            else:
                skipped += 1

    print(f"\nProcessed: {total} | Used: {len(features)} | Skipped: {skipped}")
    return np.array(features), np.array(labels)


def train_model(data_dir, output_path="emotion_model.pkl"):
    """Main training function."""
    print("=" * 50)
    print("RAVDESS Emotion Model Trainer")
    print("=" * 50)

    # Load data
    print(f"\nLoading data from: {data_dir}\n")
    X, y = load_ravdess(data_dir)

    if len(X) == 0:
        print("ERROR: No data loaded. Check your data_dir path.")
        sys.exit(1)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nClasses: {list(le.classes_)}")
    print(f"Samples per class:")
    for cls in le.classes_:
        print(f"  {cls}: {np.sum(y == cls)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Train
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"ACCURACY: {accuracy * 100:.1f}%")
    print(f"{'=' * 50}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    # Save model + label encoder
    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
            "feature_names": [
                *[f"mfcc_mean_{i}" for i in range(13)],
                *[f"mfcc_std_{i}" for i in range(13)],
                "pitch_mean",
                "pitch_std",
                "energy_mean",
                "energy_std",
                "speaking_rate",
                "spectral_centroid",
                "spectral_rolloff",
                "zcr",
            ],
        },
        output_path,
    )

    print(f"\nModel saved to: {output_path}")
    print(f"Use this in emotion_agent.py to replace rule-based classification.")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAVDESS emotion model")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to audio_speech_actors_01-24 folder",
    )
    parser.add_argument(
        "--output", type=str, default="emotion_model.pkl", help="Output model file path"
    )
    args = parser.parse_args()

    train_model(args.data_dir, args.output)
