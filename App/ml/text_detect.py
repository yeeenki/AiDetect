
import json
import re
import math
import argparse
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.sparse import hstack, csr_matrix



def load_dataset(path: str) -> pd.DataFrame:
    print(f"\n[1/5] Загружаем датасет: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("Неизвестный формат")
    except json.JSONDecodeError:
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
    if "text" not in df.columns or "source" not in df.columns:
        print(f"  Колонки в файле: {list(df.columns)}")
        raise ValueError("Нужны колонки 'text' и 'source'. Проверь файл.")
    # Нормализуем метки: AI→1, HUMAN→0
    df["source"] = df["source"].str.upper().str.strip()
    df["label"] = df["source"].map({"AI": 1, "HUMAN": 0})
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)
    print(f"  ✓ Загружено записей: {len(df)}")
    print(f"  ✓ AI-текстов:        {(df['label'] == 1).sum()}")
    print(f"  ✓ Человеческих:      {(df['label'] == 0).sum()}")
    return df


# ══════════════════════════════════════════════════════
# ШАГ 2: СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ══════════════════════════════════════════════════════

def extract_statistical_features(text: str) -> dict:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {k: 0 for k in [
            "word_count", "sentence_count", "avg_sentence_len",
            "sentence_len_variance", "uniformity_score",
            "type_token_ratio", "avg_word_length",
            "comma_rate", "ai_phrase_score", "punct_diversity"
        ]}
    # --- Слова и предложения ---
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_len = np.mean(sent_lengths) if sent_lengths else 0
    variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
    # Равномерность: чем меньше дисперсия относительно среднего — тем "роботнее"
    uniformity = max(0, 1 - (variance / (avg_len ** 2 + 1e-6)))
    # --- Словарное богатство ---
    unique_words = set(words)
    ttr = len(unique_words) / (word_count + 1e-6)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    # --- Пунктуация ---
    comma_rate = text.count(",") / (word_count + 1e-6)
    punct_chars = set(re.findall(r'[^\w\s]', text))
    punct_diversity = len(punct_chars)
    # --- ИИ-фразы (расширенный список) ---
    ai_patterns = [
        "таким образом", "следует отметить", "важно подчеркнуть",
        "в заключение", "подводя итог", "рассмотрим", "несомненно",
        "безусловно", "в данном контексте", "необходимо отметить",
        "данный вопрос", "следует рассмотреть", "в рамках",
        "furthermore", "moreover", "additionally", "in conclusion",
        "it is worth noting", "notably", "certainly", "of course",
        "it should be noted", "in this context", "overall",
    ]
    text_lower = text.lower()
    found = sum(1 for p in ai_patterns if p in text_lower)
    ai_phrase_score = min(found / 5.0, 1.0)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": round(avg_len, 2),
        "sentence_len_variance": round(variance, 2),
        "uniformity_score": round(uniformity, 4),
        "type_token_ratio": round(ttr, 4),
        "avg_word_length": round(avg_word_len, 2),
        "comma_rate": round(comma_rate, 4),
        "ai_phrase_score": round(ai_phrase_score, 4),
        "punct_diversity": punct_diversity,
    }


def run_statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает статистические признаки для всего датасета."""
    print("\n[2/5] Статистический анализ...")

    features_list = []
    for text in df["text"]:
        features_list.append(extract_statistical_features(text))

    feat_df = pd.DataFrame(features_list)

    # Выводим сравнение AI vs HUMAN по ключевым признакам
    combined = pd.concat([df["label"].reset_index(drop=True),
                          feat_df.reset_index(drop=True)], axis=1)

    print("\n  Средние значения признаков (AI vs HUMAN):")
    print(f"  {'Признак':<28} {'AI':>8} {'HUMAN':>8}")
    print("  " + "-" * 46)

    key_features = [
        "avg_sentence_len", "uniformity_score",
        "type_token_ratio", "ai_phrase_score", "comma_rate"
    ]
    for feat in key_features:
        ai_val = combined[combined["label"] == 1][feat].mean()
        hu_val = combined[combined["label"] == 0][feat].mean()
        print(f"  {feat:<28} {ai_val:>8.3f} {hu_val:>8.3f}")

    return feat_df


# ══════════════════════════════════════════════════════
# ШАГ 3: ML МОДЕЛЬ (TF-IDF + Logistic Regression)
# ══════════════════════════════════════════════════════

def train_ml_model(df: pd.DataFrame, feat_df: pd.DataFrame):
    """
    Обучает ML-модель на комбинации:
      - TF-IDF признаков (частотность слов/n-грамм)
      - Статистических признаков

    Возвращает обученную модель и результаты тестирования.
    """
    print("\n[3/5] Обучаем ML модель...")

    X_text = df["text"].fillna("")
    X_stat = feat_df.fillna(0).values
    y = df["label"].values

    # Разбивка на train/test (80% / 20%)
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF (биграммы + униграммы, топ 10000 признаков)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
    )
    X_tfidf = tfidf.fit_transform(X_text)

    # Объединяем TF-IDF и статистические признаки
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([X_tfidf, csr_matrix(X_stat)])

    X_train = X_combined[idx_train]
    X_test = X_combined[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    # Логистическая регрессия
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n  ✓ Точность (Accuracy): {acc*100:.1f}%")
    print(f"  ✓ ROC-AUC:             {auc:.3f}")
    print("\n  Детальный отчёт:")
    print(classification_report(
        y_test, y_pred,
        target_names=["HUMAN", "AI"],
        digits=3
    ))

    return model, tfidf, feat_df.columns.tolist(), {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "auc": auc,
    }

def train_ml_model(df: pd.DataFrame, feat_df: pd.DataFrame):
    """
    Обучает ML-модель на комбинации:
      - TF-IDF признаков (частотность слов/n-грамм)
      - Статистических признаков

    Возвращает обученную модель и результаты тестирования.
    """
    print("\n[3/5] Обучаем ML модель...")

    X_text = df["text"].fillna("")
    X_stat = feat_df.fillna(0).values
    y = df["label"].values

    # Разбивка на train/test (80% / 20%)
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF (биграммы + униграммы, топ 10000 признаков)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
    )
    X_tfidf = tfidf.fit_transform(X_text)

    # Объединяем TF-IDF и статистические признаки
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([X_tfidf, csr_matrix(X_stat)])

    X_train = X_combined[idx_train]
    X_test = X_combined[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    # Логистическая регрессия
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n  ✓ Точность (Accuracy): {acc*100:.1f}%")
    print(f"  ✓ ROC-AUC:             {auc:.3f}")
    print("\n  Детальный отчёт:")
    print(classification_report(
        y_test, y_pred,
        target_names=["HUMAN", "AI"],
        digits=3
    ))

    return model, tfidf, feat_df.columns.tolist(), {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "auc": auc,
    }


# ══════════════════════════════════════════════════════
# ШАГ 4: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ══════════════════════════════════════════════════════

def plot_results(df: pd.DataFrame, feat_df: pd.DataFrame, eval_results: dict):
    """Строит 4 графика для диплома."""
    print("\n[4/5] Строим графики...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Анализ детекции ИИ-текста", fontsize=16, fontweight="bold", y=1.01)

    colors = {"AI": "#E05C5C", "HUMAN": "#5C8FE0"}

    # --- График 1: Распределение классов ---
    ax1 = axes[0, 0]
    counts = df["source"].value_counts()
    bars = ax1.bar(counts.index, counts.values,
                   color=[colors.get(c, "#aaa") for c in counts.index],
                   edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha="center", fontsize=11, fontweight="bold")
    ax1.set_title("Распределение классов в датасете", fontweight="bold")
    ax1.set_ylabel("Количество текстов")
    ax1.set_xlabel("")
    ax1.spines[["top", "right"]].set_visible(False)

    # --- График 2: Матрица ошибок ---
    ax2 = axes[0, 1]
    cm = confusion_matrix(eval_results["y_test"], eval_results["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
                xticklabels=["HUMAN", "AI"], yticklabels=["HUMAN", "AI"],
                linewidths=0.5, cbar=False, annot_kws={"size": 14})
    ax2.set_title("Матрица ошибок (Confusion Matrix)", fontweight="bold")
    ax2.set_ylabel("Реальный класс")
    ax2.set_xlabel("Предсказанный класс")

    # --- График 3: Равномерность предложений (AI vs HUMAN) ---
    ax3 = axes[1, 0]
    combined = pd.concat([
        df["label"].reset_index(drop=True),
        feat_df.reset_index(drop=True)
    ], axis=1)
    ai_vals = combined[combined["label"] == 1]["uniformity_score"]
    hu_vals = combined[combined["label"] == 0]["uniformity_score"]

    ax3.hist(hu_vals, bins=30, alpha=0.65, color=colors["HUMAN"],
             label="HUMAN", edgecolor="white", linewidth=0.5)
    ax3.hist(ai_vals, bins=30, alpha=0.65, color=colors["AI"],
             label="AI", edgecolor="white", linewidth=0.5)
    ax3.set_title("Равномерность длин предложений", fontweight="bold")
    ax3.set_xlabel("Индекс равномерности")
    ax3.set_ylabel("Количество текстов")
    ax3.legend()
    ax3.spines[["top", "right"]].set_visible(False)

    # --- График 4: ROC-кривая ---
    ax4 = axes[1, 1]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(eval_results["y_test"], eval_results["y_prob"])
    ax4.plot(fpr, tpr, color="#5C8FE0", linewidth=2,
             label=f"ROC-кривая (AUC = {eval_results['auc']:.3f})")
    ax4.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Случайный классификатор")
    ax4.fill_between(fpr, tpr, alpha=0.1, color="#5C8FE0")
    ax4.set_title("ROC-кривая модели", fontweight="bold")
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.legend(loc="lower right")
    ax4.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("  ✓ Графики сохранены в results.png")
    plt.show()


# ══════════════════════════════════════════════════════
# ШАГ 5: ПРЕДСКАЗАНИЕ ДЛЯ НОВОГО ТЕКСТА
# ══════════════════════════════════════════════════════

def predict_text(text: str, model, tfidf, feat_columns: list) -> dict:
    # --- Статистический анализ ---
    stat_feats = extract_statistical_features(text)
    stat_score = (
        stat_feats["uniformity_score"] * 0.30 +
        stat_feats["ai_phrase_score"] * 0.40 +
        (1 - min(stat_feats["type_token_ratio"] * 2, 1)) * 0.15 +
        min(stat_feats["comma_rate"] * 3, 0.15)
    )
    stat_score = round(min(stat_score, 1.0), 3)

    # --- ML модель ---
    X_tfidf = tfidf.transform([text])
    X_stat = csr_matrix([[stat_feats.get(c, 0) for c in feat_columns]])
    X = hstack([X_tfidf, X_stat])
    ml_prob = model.predict_proba(X)[0][1]  # вероятность AI

    # --- Итоговый вывод (среднее двух методов) ---
    final_prob = round((stat_score * 0.35 + ml_prob * 0.65), 3)
    verdict = "🤖 ИИ-контент" if final_prob > 0.5 else "👤 Человек"

    return {
        "verdict": verdict,
        "final_probability": final_prob,
        "statistical_score": stat_score,
        "ml_probability": round(float(ml_prob), 3),
        "details": stat_feats,
    }


def print_prediction(result: dict):
    """Красивый вывод финального результата."""
    prob = result["final_probability"]
    bar_len = 40
    filled = int(prob * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "═" * 55)
    print(f"  ВЕРДИКТ:  {result['verdict']}")
    print(f"  [{bar}]")
    print(f"  Итоговая вероятность ИИ: {prob*100:.1f}%")
    print("─" * 55)
    print(f"  Статистический метод:    {result['statistical_score']*100:.1f}%")
    print(f"  ML модель (TF-IDF+LR):   {result['ml_probability']*100:.1f}%")
    print("═" * 55)


# ══════════════════════════════════════════════════════
# ГЛАВНАЯ ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Детектор ИИ-текста")
    parser.add_argument("--data", required=True, help="Путь к JSON датасету")
    parser.add_argument("--predict", type=str, default=None,
                        help="Текст для предсказания (в кавычках)")
    args = parser.parse_args()

    # 1. Загружаем данные
    df = load_dataset(args.data)

    # 2. Статистический анализ
    feat_df = run_statistical_analysis(df)

    # 3. Обучаем модель
    model, tfidf, feat_cols, eval_results = train_ml_model(df, feat_df)

    # 4. Визуализация
    plot_results(df, feat_df, eval_results)

    # 5. Предсказание
    if args.predict:
        text = args.predict
    else:
        print("\n[5/5] Введите текст для проверки (END для завершения):")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        text = "\n".join(lines)

    result = predict_text(text, model, tfidf, feat_cols)
    print_prediction(result)


if __name__ == "__main__":
    main()