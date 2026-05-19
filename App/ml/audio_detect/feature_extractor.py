import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import warnings
warnings.filterwarnings("ignore")


def extract_features(audio_path: str) -> dict:
    """
    Extract acoustic features from an audio file.
    These features are the foundation of AI vs Natural audio detection.
    """

    print(f"Analyzing: {audio_path}")

    # ── Load Audio ──────────────────────────────────────────────────
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    features = {
        "filename": audio_path,
        "duration_sec": round(duration, 4),
        "sample_rate": sr,
    }

    # ── 1. TEMPORAL FEATURES ─────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = round(float(np.mean(rms)), 6)
    features["rms_std"]  = round(float(np.std(rms)), 6)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = round(float(np.mean(zcr)), 6)
    features["zcr_std"]  = round(float(np.std(zcr)), 6)

    # ── 2. SPECTRAL FEATURES ─────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = round(float(np.mean(centroid)), 4)
    features["spectral_centroid_std"]  = round(float(np.std(centroid)), 4)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = round(float(np.mean(rolloff)), 4)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = round(float(np.mean(bandwidth)), 4)
    features["spectral_bandwidth_std"]  = round(float(np.std(bandwidth)), 4)

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features["spectral_flatness_mean"] = round(float(np.mean(flatness)), 6)
    features["spectral_flatness_std"]  = round(float(np.std(flatness)), 6)

    # ── 3. MFCCs ─────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = round(float(np.mean(mfccs[i])), 4)
        features[f"mfcc_{i+1}_std"]  = round(float(np.std(mfccs[i])), 4)

    # MFCC delta — rate of change (KEY: AI audio changes too smoothly)
    mfcc_delta = librosa.feature.delta(mfccs)
    features["mfcc_delta_mean"] = round(float(np.mean(np.abs(mfcc_delta))), 4)
    features["mfcc_delta_std"]  = round(float(np.std(mfcc_delta)), 4)

    # ── 4. PITCH / F0 PROSODY ────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        f0_voiced = f0[voiced_flag]

        if len(f0_voiced) > 10:
            features["f0_mean"]  = round(float(np.mean(f0_voiced)), 4)
            features["f0_std"]   = round(float(np.std(f0_voiced)), 4)   # LOW = AI
            features["f0_range"] = round(float(np.max(f0_voiced) - np.min(f0_voiced)), 4)
            features["f0_voiced_ratio"] = round(float(np.sum(voiced_flag) / len(voiced_flag)), 4)
        else:
            features["f0_mean"] = features["f0_std"] = features["f0_range"] = features["f0_voiced_ratio"] = 0.0

    except Exception as e:
        print(f"  F0 extraction failed: {e}")
        features["f0_mean"] = features["f0_std"] = features["f0_range"] = features["f0_voiced_ratio"] = 0.0

    # ── 5. JITTER / SHIMMER / HNR (via Praat) ────────────────────────
    # Praat only reads .wav/.aiff — convert to a temp wav first so formats
    # like .m4a, .mp3, .ogg etc. are handled correctly.
    try:
        import tempfile, soundfile as sf, os
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        sf.write(tmp_wav.name, y, sr)
        snd = parselmouth.Sound(tmp_wav.name)
        os.unlink(tmp_wav.name)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

        features["jitter_local"] = round(
            call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 6
        )
        features["shimmer_local"] = round(
            call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 6
        )

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr_mean"] = round(call(harmonicity, "Get mean", 0, 0), 4)

    except Exception as e:
        print(f"  Praat extraction failed: {e}")
        features["jitter_local"] = features["shimmer_local"] = features["hnr_mean"] = 0.0

    print(f"  ✅ Extracted {len(features)} features")
    return features


def print_report(features: dict):
    """Print a human-readable analysis report."""
    print("\n" + "="*50)
    print("AUDIO ANALYSIS REPORT")
    print("="*50)
    print(f"File        : {features['filename']}")
    print(f"Duration    : {features['duration_sec']}s")
    print()
    print("── Temporal ──")
    print(f"  RMS Energy     : {features['rms_mean']} (std: {features['rms_std']})")
    print(f"  Zero Cross Rate: {features['zcr_mean']}")
    print()
    print("── Pitch (F0) ──")
    print(f"  Mean F0        : {features['f0_mean']} Hz")
    print(f"  F0 Std Dev     : {features['f0_std']}  ← LOW = AI signature")
    print(f"  F0 Range       : {features['f0_range']} Hz")
    print()
    print("── Voice Quality (Praat) ──")
    print(f"  Jitter         : {features['jitter_local']}  ← LOW = AI signature")
    print(f"  Shimmer        : {features['shimmer_local']}  ← LOW = AI signature")
    print(f"  HNR            : {features['hnr_mean']} dB  ← HIGH = AI signature")
    print()
    print("── Spectral ──")
    print(f"  Centroid       : {features['spectral_centroid_mean']} Hz")
    print(f"  Bandwidth      : {features['spectral_bandwidth_mean']} Hz")
    print(f"  Flatness       : {features['spectral_flatness_mean']}")
    print("="*50)


# ── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/feature_extractor.py <path_to_audio_file>")
        print("Example: python src/feature_extractor.py data/natural/sample.wav")
    else:
        features = extract_features(sys.argv[1])
        print_report(features)