import os
import json
import time
import pandas as pd
from typing import Dict, Any

from openai import OpenAI
import anthropic


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "outputs/raw_claude_responses.csv"
OUTPUT_CSV = "outputs/judgments_openai_on_claude.csv"

#Pour que claude juge openAI
#INPUT_CSV = "outputs/raw_model_responses.csv"
#OUTPUT_CSV = "outputs/judgments_claude_on_openai.csv"


# OpenAI juge Claude
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-4o-mini"

# Claude juge OpenAI
#JUDGE_PROVIDER = "anthropic"
#JUDGE_MODEL = "claude-haiku-4-5"

MAX_EXAMPLES = 3000
SLEEP_BETWEEN_CALLS = 1.0


# ============================================================
# PROMPT
# ============================================================

def build_judge_prompt(question: str, gold_answer: str, candidate_answer: str) -> str:
    return f"""
Tu es un évaluateur automatique de réponses.

Tu dois juger si la réponse candidate est correcte par rapport à la réponse de référence.

Question :
{question}

Réponse de référence :
{gold_answer}

Réponse candidate :
{candidate_answer}

Critères d'évaluation :
- La réponse est correcte si elle est sémantiquement équivalente à la réponse de référence.
- Accepte les synonymes, reformulations, abréviations et variantes raisonnables.
- Refuse les réponses trop vagues, contradictoires ou qui donnent une mauvaise entité.
- Si la réponse contient la bonne réponse avec du texte additionnel non contradictoire, considère-la correcte.
- Ne sois pas trop strict sur la formulation exacte.

Retourne uniquement un JSON valide avec ce format exact :
{{
  "verdict": "correct",
  "score": 100,
  "reason": "explication courte"
}}

Contraintes :
- verdict doit être exactement "correct" ou "incorrect".
- score doit être un nombre entre 0 et 100.
- reason doit être court.
""".strip()


# ============================================================
# CLIENTS
# ============================================================

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquante.")
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY manquante.")
    return anthropic.Anthropic(api_key=api_key)


# ============================================================
# PARSING JSON
# ============================================================

def parse_json_safely(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Cas où le modèle met ```json ... ```
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
        return {
            "verdict": data.get("verdict", "parse_error"),
            "score": data.get("score", None),
            "reason": data.get("reason", ""),
            "raw_judge_output": text,
        }
    except json.JSONDecodeError:
        return {
            "verdict": "parse_error",
            "score": None,
            "reason": "Impossible de parser la sortie JSON.",
            "raw_judge_output": text,
        }


# ============================================================
# APPELS AUX JUGES
# ============================================================

def ask_openai_judge(client: OpenAI, prompt: str) -> Dict[str, Any]:
    response = client.responses.create(
        model=JUDGE_MODEL,
        instructions="Tu es un juge automatique. Tu réponds uniquement en JSON valide.",
        input=prompt,
        temperature=0,
    )

    return parse_json_safely(response.output_text)


def ask_anthropic_judge(client: anthropic.Anthropic, prompt: str) -> Dict[str, Any]:
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    text = " ".join(text_parts).strip()
    return parse_json_safely(text)


def ask_judge(client, prompt: str) -> Dict[str, Any]:
    if JUDGE_PROVIDER == "openai":
        return ask_openai_judge(client, prompt)

    if JUDGE_PROVIDER == "anthropic":
        return ask_anthropic_judge(client, prompt)

    raise ValueError(f"Juge non supporté : {JUDGE_PROVIDER}")


# ============================================================
# PIPELINE
# ============================================================

def main():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required = {"question", "gold_answer", "model_response"}
    if not required.issubset(df.columns):
        raise ValueError(f"Le fichier doit contenir les colonnes : {required}")

    df = df.dropna(subset=["question", "gold_answer", "model_response"]).copy()

    if MAX_EXAMPLES is not None:
        df = df.head(MAX_EXAMPLES).copy()

    # Reprise si fichier déjà partiellement généré
    rows = []
    already_done_ids = set()

    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        rows = existing.to_dict("records")
        already_done_ids = set(existing["row_id"].tolist())
        print(f"Reprise détectée : {len(already_done_ids)} jugements déjà présents.")

    if JUDGE_PROVIDER == "openai":
        client = get_openai_client()
    elif JUDGE_PROVIDER == "anthropic":
        client = get_anthropic_client()
    else:
        raise ValueError(f"JUDGE_PROVIDER invalide : {JUDGE_PROVIDER}")

    print(f"Juge utilisé : {JUDGE_PROVIDER}:{JUDGE_MODEL}")
    print(f"Nombre d'exemples à juger : {len(df)}")

    for i, row in df.iterrows():
        row_id = int(row["row_id"]) if "row_id" in row else int(i)

        if row_id in already_done_ids:
            continue

        question = str(row["question"])
        gold_answer = str(row["gold_answer"])
        candidate_answer = str(row["model_response"])
        candidate_model = str(row["model"]) if "model" in row else "unknown"

        prompt = build_judge_prompt(question, gold_answer, candidate_answer)

        try:
            judgment = ask_judge(client, prompt)
        except Exception as e:
            judgment = {
                "verdict": "api_error",
                "score": None,
                "reason": str(e),
                "raw_judge_output": "",
            }

        rows.append({
            "row_id": row_id,
            "candidate_model": candidate_model,
            "judge_provider": JUDGE_PROVIDER,
            "judge_model": JUDGE_MODEL,
            "question": question,
            "gold_answer": gold_answer,
            "candidate_answer": candidate_answer,
            "judge_verdict": judgment.get("verdict"),
            "judge_score": judgment.get("score"),
            "judge_reason": judgment.get("reason"),
            "raw_judge_output": judgment.get("raw_judge_output"),
        })

        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
            print(f"{len(rows)} jugements sauvegardés...")

        time.sleep(SLEEP_BETWEEN_CALLS)

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Terminé. Résultats sauvegardés dans : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()