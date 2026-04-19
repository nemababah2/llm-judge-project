import os
import re
import time
import unicodedata
from io import StringIO
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "data/questions.csv"
OUTPUT_DIR = "outputs"
SLEEP_BETWEEN_CALLS = 1.0
MAX_QUESTIONS = 3000

MODELS_TO_RUN = [
    "openai:gpt-4o-mini",
]

USE_PRECOMPUTED_RESPONSES = False
PRECOMPUTED_RESPONSES_CSV = "precomputed_responses.csv"


# ============================================================
# OUTILS TEXTE / SCORING
# ============================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_gold_score(pred: str, gold: str) -> int:
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    return int(gold_n in pred_n) if gold_n else 0


def build_prompt(question: str) -> str:
    return f"""
Réponds à la question suivante.
Donne uniquement la réponse finale, très courte.
Ne donne aucune explication.
Ne reformule pas la question.
Écris seulement la réponse.

Question: {question}
Réponse:
""".strip()


# ============================================================
# INTERFACES LLM
# ============================================================

def ask_openai(model_name: str, prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquante dans les variables d'environnement.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model_name,
        instructions="Tu es un assistant précis qui répond de façon très courte.",
        input=prompt,
    )

    return response.output_text.strip()


def ask_model(model_id: str, question: str) -> str:
    provider, model_name = model_id.split(":", 1)
    prompt = build_prompt(question)

    if provider == "openai":
        return ask_openai(model_name, prompt)

    raise ValueError(f"Provider non supporté: {provider}")


# ============================================================
# PIPELINE
# ============================================================

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(csv_path: str) -> pd.DataFrame:
    raw = open(csv_path, "rb").read()

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    text = None
    used_encoding = None

    for enc in encodings_to_try:
        try:
            text = raw.decode(enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            pass

    if text is None:
        text = raw.decode("cp1252", errors="replace")
        used_encoding = "cp1252-with-replace"

    print(f"Fichier décodé avec: {used_encoding}")

    # Premier essai : avec en-tête
    df = pd.read_csv(StringIO(text), on_bad_lines="skip")

    # Si pas d'en-tête, on force les noms
    if "question" not in df.columns or "response" not in df.columns:
        df = pd.read_csv(
            StringIO(text),
            header=None,
            names=["question", "response"],
            on_bad_lines="skip",
        )

    if "question" not in df.columns or "response" not in df.columns:
        raise ValueError("Impossible de construire les colonnes 'question' et 'response'.")

    return df


def generate_responses(df: pd.DataFrame, model_ids: List[str]) -> pd.DataFrame:
    rows = []

    partial_save_path = os.path.join(OUTPUT_DIR, "raw_model_responses_partial.csv")

    for model_id in model_ids:
        print(f"\n=== Génération des réponses pour {model_id} ===")

        for i, row in df.iterrows():
            question = str(row["question"])
            gold = str(row["response"])

            try:
                model_response = ask_model(model_id, question)
            except Exception as e:
                print(f"[ERREUR] modèle={model_id} idx={i}: {e}")
                model_response = ""

            result_row = {
                "row_id": i,
                "model": model_id,
                "question": question,
                "gold_answer": gold,
                "model_response": model_response,
            }

            rows.append(result_row)

            # Sauvegarde progressive toutes les 20 questions
            if (i + 1) % 20 == 0:
                pd.DataFrame(rows).to_csv(partial_save_path, index=False)
                print(f"{i + 1} questions traitées pour {model_id} | sauvegarde partielle OK")

            time.sleep(SLEEP_BETWEEN_CALLS)

    # Sauvegarde finale
    final_df = pd.DataFrame(rows)
    final_df.to_csv(partial_save_path, index=False)

    return final_df


def load_precomputed_responses(df_gold: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    pre = pd.read_csv(csv_path)

    required = {"model", "question", "model_response"}
    if not required.issubset(set(pre.columns)):
        raise ValueError(
            "Le CSV de réponses pré-calculées doit contenir: model, question, model_response"
        )

    merged = pre.merge(
        df_gold.rename(columns={"response": "gold_answer"}),
        on="question",
        how="left"
    )
    return merged


def evaluate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()

    df["gold_norm"] = df["gold_answer"].apply(normalize_text)
    df["pred_norm"] = df["model_response"].apply(normalize_text)
    df["exact_match"] = (df["pred_norm"] == df["gold_norm"]).astype(int)
    df["contains_gold"] = df.apply(
        lambda r: contains_gold_score(r["model_response"], r["gold_answer"]),
        axis=1
    )
    df["response_length_chars"] = df["model_response"].fillna("").astype(str).str.len()

    return df


def compute_summary_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    grouped = eval_df.groupby("model").agg(
        n_questions=("question", "count"),
        accuracy_exact=("exact_match", "mean"),
        accuracy_contains_gold=("contains_gold", "mean"),
        avg_response_length=("response_length_chars", "mean"),
    ).reset_index()

    grouped["accuracy_exact"] = (grouped["accuracy_exact"] * 100).round(2)
    grouped["accuracy_contains_gold"] = (grouped["accuracy_contains_gold"] * 100).round(2)
    grouped["avg_response_length"] = grouped["avg_response_length"].round(2)

    return grouped.sort_values("accuracy_exact", ascending=False)


def save_error_samples(eval_df: pd.DataFrame, output_dir: str, max_errors_per_model: int = 30) -> None:
    errors = eval_df[eval_df["exact_match"] == 0].copy()
    if not errors.empty:
        parts = []
        for model_name, sub in errors.groupby("model"):
            parts.append(sub.head(max_errors_per_model))
        pd.concat(parts, ignore_index=True).to_csv(
            os.path.join(output_dir, "error_samples.csv"),
            index=False
        )


# ============================================================
# VISUALISATIONS
# ============================================================

def plot_accuracy(summary_df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(9, 6))
    plt.bar(summary_df["model"], summary_df["accuracy_exact"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Accuracy exacte (%)")
    plt.title("Accuracy par modèle")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_model.png"), dpi=200)
    plt.close()


def plot_contains_gold(summary_df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(9, 6))
    plt.bar(summary_df["model"], summary_df["accuracy_contains_gold"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Contains-gold (%)")
    plt.title("Présence de la vraie réponse dans la sortie")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contains_gold_by_model.png"), dpi=200)
    plt.close()


def plot_response_length(eval_df: pd.DataFrame, output_dir: str) -> None:
    for model_name, sub in eval_df.groupby("model"):
        plt.figure(figsize=(9, 6))
        plt.hist(sub["response_length_chars"], bins=20)
        plt.xlabel("Longueur des réponses (caractères)")
        plt.ylabel("Nombre de réponses")
        plt.title(f"Longueur des réponses - {model_name}")
        plt.tight_layout()
        safe_name = model_name.replace(":", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"response_length_{safe_name}.png"), dpi=200)
        plt.close()


def plot_correct_vs_incorrect(eval_df: pd.DataFrame, output_dir: str) -> None:
    data = (
        eval_df.groupby(["model", "exact_match"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    if 0 not in data.columns:
        data[0] = 0
    if 1 not in data.columns:
        data[1] = 0

    plt.figure(figsize=(9, 6))
    x = range(len(data))
    plt.bar(x, data[1], label="Correct")
    plt.bar(x, data[0], bottom=data[1], label="Incorrect")
    plt.xticks(list(x), data["model"], rotation=20, ha="right")
    plt.ylabel("Nombre de réponses")
    plt.title("Correct vs incorrect par modèle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correct_vs_incorrect.png"), dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_output_dir(OUTPUT_DIR)

    df = load_dataset(INPUT_CSV)
    df = df.dropna(subset=["question", "response"]).copy()

    if MAX_QUESTIONS is not None:
        df = df.head(MAX_QUESTIONS).copy()

    print(f"Dataset chargé: {len(df)} questions")

    if USE_PRECOMPUTED_RESPONSES:
        results_df = load_precomputed_responses(df, PRECOMPUTED_RESPONSES_CSV)
    else:
        if not MODELS_TO_RUN:
            raise ValueError("MODELS_TO_RUN est vide. Ajoute au moins un modèle.")
        results_df = generate_responses(df, MODELS_TO_RUN)

    results_path = os.path.join(OUTPUT_DIR, "raw_model_responses.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Réponses brutes sauvegardées: {results_path}")

    eval_df = evaluate_results(results_df)
    eval_path = os.path.join(OUTPUT_DIR, "evaluated_results.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Résultats évalués sauvegardés: {eval_path}")

    summary_df = compute_summary_metrics(eval_df)
    summary_path = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Métriques sauvegardées: {summary_path}")

    save_error_samples(eval_df, OUTPUT_DIR)

    plot_accuracy(summary_df, OUTPUT_DIR)
    plot_contains_gold(summary_df, OUTPUT_DIR)
    plot_response_length(eval_df, OUTPUT_DIR)
    plot_correct_vs_incorrect(eval_df, OUTPUT_DIR)

    print("\n=== RÉSUMÉ ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()