"""
src/pipeline.py — Single-file inference for AI audio detection.

Usage:
    python src/pipeline.py audio.wav
    python src/pipeline.py audio.wav --model models/detector.pkl
    python src/pipeline.py folder/ --batch
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pickle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run src/model.py first to train and save a model."
        )
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def _extract(audio_path: str | Path) -> np.ndarray:
    """Extract features using the project's feature_extractor."""
    from feature_extractor import extract_features  # type: ignore

    features = extract_features(str(audio_path))

    # extract_features returns a dict — drop non-numeric keys and convert to array
    SKIP = {"filename", "label", "label_name", "source_dir"}
    numeric = {k: v for k, v in features.items() if k not in SKIP and isinstance(v, (int, float))}
    return np.array(list(numeric.values()), dtype=float)


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

class AudioDetector:
    """Load a saved model bundle and run inference on audio files."""

    LABEL_MAP = {0: "natural", 1: "ai_generated"}

    def __init__(self, model_path: str | Path = "models/detector.pkl"):
        bundle = _load_model(model_path)

        # Support dict bundle, (model, feature_cols) tuple, or bare model
        if isinstance(bundle, dict):
            self.model = bundle["model"]
            self.scaler = bundle.get("scaler")
            self.feature_names: list[str] = bundle.get("feature_names", [])
            self.metadata: dict = bundle.get("metadata", {})
        elif isinstance(bundle, tuple):
            self.model, feature_cols = bundle
            self.scaler = None
            self.feature_names = list(feature_cols) if feature_cols is not None else []
            self.metadata = {}
        else:
            self.model = bundle
            self.scaler = None
            self.feature_names = []
            self.metadata = {}

    # ------------------------------------------------------------------

    def predict(self, audio_path: str | Path) -> dict:
        """
        Predict whether an audio file is natural or AI-generated.

        Returns
        -------
        dict with keys:
            file        – resolved file path
            label       – "natural" | "ai_generated"
            confidence  – float 0-1 (probability of predicted class)
            probabilities – {"natural": float, "ai_generated": float}
            features_used – number of features extracted
            inference_ms  – wall-clock inference time
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

        t0 = time.perf_counter()

        # 1. Extract features
        features = _extract(audio_path)

        # 2. Optionally scale
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)

        # 3. Predict
        pred_idx = int(self.model.predict(features)[0])
        label = self.LABEL_MAP.get(pred_idx, str(pred_idx))

        # 4. Probabilities (not all sklearn estimators expose predict_proba)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            probs = {
                self.LABEL_MAP.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(proba)
            }
            confidence = round(float(proba[pred_idx]), 4)
        else:
            probs = {label: 1.0}
            confidence = 1.0

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "file": str(audio_path),
            "label": label,
            "confidence": confidence,
            "probabilities": probs,
            "features_used": features.shape[1],
            "inference_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------

    def predict_batch(self, folder: str | Path, extensions=(".wav", ".mp3", ".flac", ".ogg")) -> list[dict]:
        """Run predict() on every audio file in a folder (non-recursive)."""
        folder = Path(folder)
        audio_files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in extensions]

        if not audio_files:
            print(f"[warn] No audio files found in '{folder}'")
            return []

        results = []
        for i, fp in enumerate(audio_files, 1):
            try:
                result = self.predict(fp)
                results.append(result)
                _print_result(result, prefix=f"[{i}/{len(audio_files)}] ")
            except Exception as exc:
                err = {"file": str(fp), "error": str(exc)}
                results.append(err)
                print(f"[{i}/{len(audio_files)}] ERROR  {fp.name}: {exc}")

        _print_batch_summary(results)
        return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_result(result: dict, prefix: str = "") -> None:
    if "error" in result:
        print(f"{prefix}ERROR  {Path(result['file']).name} — {result['error']}")
        return

    label = result["label"]
    conf = result["confidence"]
    bar_len = int(conf * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)

    icon = "🤖" if label == "ai_generated" else "🎙️"
    print(
        f"{prefix}{icon}  {Path(result['file']).name}\n"
        f"       Label      : {label.upper()}\n"
        f"       Confidence : [{bar}] {conf:.1%}\n"
        f"       Probs      : {result['probabilities']}\n"
        f"       Time       : {result['inference_ms']} ms\n"
    )


def _print_batch_summary(results: list[dict]) -> None:
    good = [r for r in results if "label" in r]
    ai_count = sum(1 for r in good if r["label"] == "ai_generated")
    nat_count = len(good) - ai_count
    err_count = len(results) - len(good)
    avg_conf = np.mean([r["confidence"] for r in good]) if good else 0.0

    print("─" * 50)
    print(f"  Batch summary  ({len(results)} files)")
    print(f"  🎙️  Natural      : {nat_count}")
    print(f"  🤖  AI-generated : {ai_count}")
    if err_count:
        print(f"  ⚠️  Errors       : {err_count}")
    print(f"  Avg confidence : {avg_conf:.1%}")
    print("─" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI Audio Detector — predict natural vs AI-generated speech.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py sample.wav
  python src/pipeline.py sample.wav --model models/detector.pkl
  python src/pipeline.py data/test_folder/ --batch
  python src/pipeline.py sample.wav --json
        """,
    )
    p.add_argument("input", help="Path to an audio file or folder (with --batch)")
    p.add_argument(
        "--model",
        default="models/detector.pkl",
        help="Path to saved model bundle (default: models/detector.pkl)",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Process all audio files in the given folder",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print result(s) as JSON instead of formatted text",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        detector = AudioDetector(model_path=args.model)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.batch:
        results = detector.predict_batch(args.input)
        if args.output_json:
            print(json.dumps(results, indent=2))
    else:
        try:
            result = detector.predict(args.input)
        except FileNotFoundError as e:
            print(f"[error] {e}", file=sys.stderr)
            sys.exit(1)

        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            _print_result(result)


if __name__ == "__main__":
    main()