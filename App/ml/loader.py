import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "trained_model.pkl"

_cache = {}  # модель держим в памяти

def get_model():
    if "model" not in _cache:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Модель не найдена. Сначала обучи: python manage.py train_model"
            )
        with open(MODEL_PATH, "rb") as f:
            _cache["model"] = pickle.load(f)  # (model, tfidf, feat_cols)
    return _cache["model"]