import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "outputs"

FILES = {
    "OpenAI juge Claude": "outputs/judgments_openai_on_claude.csv",
    "Claude juge OpenAI": "outputs/judgments_claude_on_openai.csv",
}


# ============================================================
# UTILS
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_verdict(value):
    if pd.isna(value):
        return "unknown"

    value = str(value).strip().lower()

    if value in ["correct", "correcte", "true", "1"]:
        return "correct"

    if value in ["incorrect", "false", "0"]:
        return "incorrect"

    if "correct" in value and "incorrect" not in value:
        return "correct"

    if "incorrect" in value:
        return "incorrect"

    return "unknown"


def load_judgment_file(path, experiment_name):
    df = pd.read_csv(path)

    # Enlever ligne parasite si elle existe
    if "question" in df.columns:
        df = df[df["question"] != "question"]

    df["experiment"] = experiment_name
    df["judge_verdict_clean"] = df["judge_verdict"].apply(clean_verdict)

    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")

    return df


def load_all_files():
    dfs = []

    for experiment_name, path in FILES.items():
        if os.path.exists(path):
            print(f"Lecture : {path}")
            dfs.append(load_judgment_file(path, experiment_name))
        else:
            print(f"Fichier absent, ignoré : {path}")

    if not dfs:
        raise FileNotFoundError("Aucun fichier de jugement trouvé dans outputs/.")

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# STATS
# ============================================================

def compute_summary(df):
    summary = df.groupby("experiment").agg(
        n_examples=("question", "count"),
        correct_rate=("judge_verdict_clean", lambda x: (x == "correct").mean() * 100),
        incorrect_rate=("judge_verdict_clean", lambda x: (x == "incorrect").mean() * 100),
        unknown_rate=("judge_verdict_clean", lambda x: (x == "unknown").mean() * 100),
        avg_score=("judge_score", "mean"),
        median_score=("judge_score", "median"),
        min_score=("judge_score", "min"),
        max_score=("judge_score", "max"),
    ).reset_index()

    numeric_cols = [
        "correct_rate",
        "incorrect_rate",
        "unknown_rate",
        "avg_score",
        "median_score",
        "min_score",
        "max_score",
    ]

    for col in numeric_cols:
        summary[col] = summary[col].round(2)

    return summary


def save_error_examples(df):
    errors = df[df["judge_verdict_clean"] == "incorrect"].copy()

    if not errors.empty:
        errors.to_csv(
            os.path.join(OUTPUT_DIR, "analysis_incorrect_examples.csv"),
            index=False
        )

    correct = df[df["judge_verdict_clean"] == "correct"].copy()

    if not correct.empty:
        correct.to_csv(
            os.path.join(OUTPUT_DIR, "analysis_correct_examples.csv"),
            index=False
        )


def analyze_score_strictness(df):
    strictness = df.groupby("experiment").agg(
        zero_score_rate=("judge_score", lambda x: (x == 0).mean() * 100),
        perfect_score_rate=("judge_score", lambda x: (x == 100).mean() * 100),
        middle_score_rate=("judge_score", lambda x: ((x > 0) & (x < 100)).mean() * 100),
    ).reset_index()

    for col in ["zero_score_rate", "perfect_score_rate", "middle_score_rate"]:
        strictness[col] = strictness[col].round(2)

    strictness.to_csv(
        os.path.join(OUTPUT_DIR, "analysis_judge_strictness.csv"),
        index=False
    )

    return strictness


# ============================================================
# GRAPHES
# ============================================================

def plot_correct_rate(summary):
    plt.figure(figsize=(9, 6))
    plt.bar(summary["experiment"], summary["correct_rate"])
    plt.ylabel("Réponses jugées correctes (%)")
    plt.title("Taux de réponses jugées correctes par expérience")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "analysis_correct_rate_by_experiment.png"), dpi=200)
    plt.close()


def plot_average_score(summary):
    plt.figure(figsize=(9, 6))
    plt.bar(summary["experiment"], summary["avg_score"])
    plt.ylabel("Score moyen attribué")
    plt.title("Score moyen attribué par chaque juge")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "analysis_average_score_by_experiment.png"), dpi=200)
    plt.close()


def plot_score_distribution(df):
    for experiment, sub in df.groupby("experiment"):
        plt.figure(figsize=(9, 6))
        plt.hist(sub["judge_score"].dropna(), bins=20)
        plt.xlabel("Score attribué par le juge")
        plt.ylabel("Nombre de réponses")
        plt.title(f"Distribution des scores — {experiment}")
        plt.tight_layout()

        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", experiment.lower()).strip("_")
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"analysis_score_distribution_{safe_name}.png"),
            dpi=200
        )
        plt.close()


def plot_verdict_distribution(df):
    counts = (
        df.groupby(["experiment", "judge_verdict_clean"])
        .size()
        .unstack(fill_value=0)
    )

    counts.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Nombre de réponses")
    plt.title("Distribution des verdicts par expérience")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "analysis_verdict_distribution.png"), dpi=200)
    plt.close()


def plot_strictness(strictness):
    plot_df = strictness.set_index("experiment")[
        ["zero_score_rate", "middle_score_rate", "perfect_score_rate"]
    ]

    plot_df.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Proportion (%)")
    plt.title("Sévérité du juge : scores 0, intermédiaires et 100")
    plt.xticks(rotation=20, ha="right")
    plt.legend(["Score 0", "Score intermédiaire", "Score 100"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "analysis_judge_strictness.png"), dpi=200)
    plt.close()


def plot_score_boxplot(df):
    experiments = []
    scores = []

    for experiment, sub in df.groupby("experiment"):
        experiments.append(experiment)
        scores.append(sub["judge_score"].dropna())

    plt.figure(figsize=(9, 6))
    plt.boxplot(scores, labels=experiments)
    plt.ylabel("Score attribué")
    plt.title("Dispersion des scores attribués par les juges")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "analysis_score_boxplot.png"), dpi=200)
    plt.close()


# ============================================================
# ANALYSE TEXTUELLE SIMPLE
# ============================================================

def extract_common_reasons(df):
    """
    Analyse très simple des raisons données par les juges.
    On compte quelques mots-clés fréquents utiles pour le rapport.
    """

    keywords = {
        "réponse vague": ["vague", "trop général", "générale"],
        "mauvaise entité": ["mauvaise entité", "autre personne", "différent", "incorrect"],
        "réponse incomplète": ["incomplet", "partiel", "manque"],
        "bonne reformulation": ["équivalent", "synonyme", "reformulation", "correct"],
        "contradiction": ["contradictoire", "contredit", "contradiction"],
    }

    rows = []

    for experiment, sub in df.groupby("experiment"):
        reasons = sub["judge_reason"].fillna("").astype(str).str.lower()

        row = {"experiment": experiment}

        for category, words in keywords.items():
            count = 0
            for reason in reasons:
                if any(w in reason for w in words):
                    count += 1
            row[category] = count

        rows.append(row)

    reason_df = pd.DataFrame(rows)
    reason_df.to_csv(os.path.join(OUTPUT_DIR, "analysis_reason_keywords.csv"), index=False)

    return reason_df


def plot_reason_keywords(reason_df):
    if reason_df.empty:
        return

    plot_df = reason_df.set_index("experiment")

    plt.figure(figsize=(11, 6))
    plot_df.T.plot(kind="bar", figsize=(11, 6))
    plt.ylabel("Nombre d'occurrences")
    plt.title("Catégories fréquentes dans les justifications des juges")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "analysis_reason_keywords.png"), dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_output_dir()

    df = load_all_files()

    df.to_csv(os.path.join(OUTPUT_DIR, "analysis_all_judgments_cleaned.csv"), index=False)

    summary = compute_summary(df)
    summary.to_csv(os.path.join(OUTPUT_DIR, "analysis_summary.csv"), index=False)

    strictness = analyze_score_strictness(df)
    save_error_examples(df)

    reason_df = extract_common_reasons(df)

    plot_correct_rate(summary)
    plot_average_score(summary)
    plot_score_distribution(df)
    plot_verdict_distribution(df)
    plot_strictness(strictness)
    plot_score_boxplot(df)
    plot_reason_keywords(reason_df)

    print("\n=== RÉSUMÉ GLOBAL ===")
    print(summary.to_string(index=False))

    print("\n=== SÉVÉRITÉ DES JUGES ===")
    print(strictness.to_string(index=False))

    print("\nFichiers générés dans outputs/:")
    print("- analysis_summary.csv")
    print("- analysis_judge_strictness.csv")
    print("- analysis_all_judgments_cleaned.csv")
    print("- analysis_correct_rate_by_experiment.png")
    print("- analysis_average_score_by_experiment.png")
    print("- analysis_verdict_distribution.png")
    print("- analysis_judge_strictness.png")
    print("- analysis_score_boxplot.png")
    print("- analysis_reason_keywords.png")


if __name__ == "__main__":
    main()