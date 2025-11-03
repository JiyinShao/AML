import os
from typing import List, Dict, Tuple
import csv
import torch
from transformers import pipeline
import evaluate
from data_loader import Sample

DEVICE = 0 if torch.cuda.is_available() else -1

MODELS = [
    "t5-small",
    "t5-base",
    "t5-large",
    "google/flan-t5-base",
    "Vamsi/T5_Paraphrase_Paws"
]

def _truncate_words(text: str, max_words: int) -> str:
    ws = text.split()
    return " ".join(ws[:max_words])

def build_zero_shot_prompt(article: str, model_name: str) -> str:
    art = _truncate_words(article, 420)
    if "t5" in model_name and "paraphraser" not in model_name:
        return f"summarize: {art}"
    elif "paraphraser" in model_name:
        return f"Paraphrase the following article into a concise 1-2 sentence summary:\n\n{art}"
    else:
        return art

def build_few_shot_prompt(article: str, exemplar_article: str, exemplar_summary: str, model_name: str) -> str:
    ex_art = _truncate_words(exemplar_article, 120)
    ex_sum = _truncate_words(exemplar_summary, 60)
    art    = _truncate_words(article, 360)
    if "t5" in model_name and "paraphraser" not in model_name:
        prefix = "summarize: "
        return (
            f"{prefix}Example:\n"
            f"Article: {ex_art}\n"
            f"Summary: {ex_sum}\n"
            "----\n"
            f"Article: {art}\n"
            "Summary:"
        )
    elif "paraphraser" in model_name:
        return (
            "Example:\n"
            f"Article: {ex_art}\n"
            f"1-2 sentence summary: {ex_sum}\n"
            "----\n"
            f"Article: {art}\n"
            "1-2 sentence summary:"
        )
    else:
        return (
            "Example:\n"
            f"Article: {ex_art}\n"
            f"Summary: {ex_sum}\n"
            "----\n"
            f"Article: {art}\n"
            "Summary:"
        )

def _build_pipeline_for(model_name: str):
    if "paraphraser" in model_name:
        return pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=DEVICE)
    else:
        return pipeline("summarization", model=model_name, tokenizer=model_name, device=DEVICE)

def _run_pipe(p, prompt: str, max_new_tokens=160, min_new_tokens=40):
    out = p(prompt, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, do_sample=False)
    if isinstance(out, list) and out:
        if "summary_text" in out[0]:
            return out[0]["summary_text"]
        if "generated_text" in out[0]:
            return out[0]["generated_text"]
    return str(out)

def summarize_with_models(samples: List[Sample], model_names: List[str] = None) -> List[dict]:
    if model_names is None:
        model_names = MODELS
    records = []
    exemplar = samples[0]
    for model_name in model_names:
        print(f"[Init model] {model_name}")
        p = _build_pipeline_for(model_name)
        for idx, s in enumerate(samples):
            # zero-shot
            z_prompt = build_zero_shot_prompt(s.article, model_name)
            z_pred = _run_pipe(p, z_prompt)
            records.append({
                "article_index": idx,
                "model": model_name,
                "prompt": "zero",
                "prediction": z_pred,
                "reference": s.reference
            })
            # few-shot
            f_prompt = build_few_shot_prompt(s.article, exemplar.article, exemplar.reference, model_name)
            f_pred = _run_pipe(p, f_prompt)
            records.append({
                "article_index": idx,
                "model": model_name,
                "prompt": "few",
                "prediction": f_pred,
                "reference": s.reference
            })
            print(f"[Generated] {model_name} | article #{idx}: zero & few")
    return records

# ---------- evaluation (ROUGE-1/2/L, BLEU, BERTScore) ----------
def eval_and_save(records: List[dict], outdir="outputs") -> Dict:
    os.makedirs(outdir, exist_ok=True)
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bert = evaluate.load("bertscore")

    per_item = []
    for r in records:
        ref, pred = r["reference"], r["prediction"]
        r_ = rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)
        b_ = bleu.compute(predictions=[pred], references=[[ref]])
        bs = bert.compute(predictions=[pred], references=[ref],
                          model_type="bert-base-uncased", lang="en")
        per_item.append({
            "article_index": r["article_index"],
            "model": r["model"],
            "prompt": r["prompt"],
            "rouge1": r_["rouge1"], "rouge2": r_["rouge2"], "rougeL": r_["rougeL"],
            "bleu": b_["bleu"],
            "bertscore_f1": bs["f1"][0],
            "ref_len": len(ref.split()),
            "pred_len": len(pred.split())
        })

    csv_path = os.path.join(outdir, "scores_by_article.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_item[0].keys()))
        writer.writeheader(); writer.writerows(per_item)
    print(f"[Saved] {csv_path}")

    agg: Dict[Tuple[str, str], Dict[str, float]] = {}
    keys = {(x["model"], x["prompt"]) for x in per_item}
    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
    for m, p in keys:
        group = [x for x in per_item if x["model"] == m and x["prompt"] == p]
        agg[(m, p)] = {k: sum(x[k] for x in group)/len(group) for k in metrics}

    return {"agg": agg, "per_item": per_item}
