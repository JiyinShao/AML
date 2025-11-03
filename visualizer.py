import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from data_loader import Sample

# ---------- Length Distribution ----------
def visualize_lengths(samples: List[Sample], outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    article_lens = [len(s.article.split()) for s in samples]
    ref_lens = [len(s.reference.split()) for s in samples]

    plt.figure()
    plt.hist(article_lens, bins=30)
    plt.title("Article Length Distribution (words)")
    plt.xlabel("Words"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "article_length_hist.png"))
    plt.close()

    plt.figure()
    plt.hist(ref_lens, bins=30)
    plt.title("Reference Summary Length Distribution (words)")
    plt.xlabel("Words"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reference_length_hist.png"))
    plt.close()

    print("[Saved] article_length_hist.png / reference_length_hist.png")


# ---------- Aggregate Comparisons ----------
def plot_agg_prompt_compare(agg: Dict[Tuple[str, str], Dict[str, float]], model_name: str, outdir="outputs"):
    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
    zero_vals = [agg[(model_name, "zero")][m] for m in metrics]
    few_vals = [agg[(model_name, "few")][m] for m in metrics]

    plt.figure()
    x = range(len(metrics))
    w = 0.38
    plt.bar([i - w/2 for i in x], zero_vals, w, label="Zero-shot")
    plt.bar([i + w/2 for i in x], few_vals, w, label="Few-shot")
    xt = [m.upper() if m != "bertscore_f1" else "BERT(F1)" for m in metrics]
    plt.xticks(list(x), xt)
    plt.ylabel("Score")
    plt.title(f"{model_name}: Zero vs Few (ROUGE/BLEU/BERTScore)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"agg_by_prompt_{model_name.replace('/', '_')}.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")


def plot_agg_model_compare(agg: Dict[Tuple[str, str], Dict[str, float]], prompt: str, outdir="outputs"):
    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
    models = sorted({m for (m, p) in agg.keys() if p == prompt})
    vals = {m: [agg[(m, prompt)][k] for k in metrics] for m in models}

    plt.figure()
    x = range(len(metrics))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        plt.bar([j - 0.4 + w/2 + i*w for j in x], vals[m], w, label=m)
    xt = [m.upper() if m != "bertscore_f1" else "BERT(F1)" for m in metrics]
    plt.xticks(list(x), xt)
    plt.ylabel("Score")
    plt.title(f"Model Comparison under {prompt.title()}-shot")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"agg_by_model_{prompt}.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")


# ---------- Per-Article Comparisons ----------
def plot_per_article_metric(per_item_scores: List[dict], metric: str, prompt: str, outdir="outputs"):
    models = sorted({x["model"] for x in per_item_scores})
    indices = sorted({x["article_index"] for x in per_item_scores})
    width = 0.8 / len(models)

    plt.figure(figsize=(max(7, len(indices)*0.7), 4.2))
    for i, m in enumerate(models):
        vals = []
        for idx in indices:
            item = next(
                x for x in per_item_scores
                if x["article_index"] == idx and x["prompt"] == prompt and x["model"] == m
            )
            vals.append(item[metric])
        xs = [j - 0.4 + width/2 + i*width for j in range(len(indices))]
        plt.bar(xs, vals, width, label=m)

    plt.xticks(range(len(indices)), [str(i) for i in indices])
    ylabel = metric.upper() if metric != "bertscore_f1" else "BERT(F1)"
    plt.ylabel(ylabel)
    plt.xlabel("Article Index")
    plt.title(f"{ylabel} per Article | Prompt={prompt.title()}-shot")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"per_article_{metric}_{prompt}.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")


def plot_per_article_rouge_compare(per_item_scores: List[dict], prompt: str, outdir="outputs"):
    for metric in ["rouge1", "rouge2"]:
        models = sorted({x["model"] for x in per_item_scores})
        indices = sorted({x["article_index"] for x in per_item_scores})
        width = 0.8 / len(models)

        plt.figure(figsize=(max(7, len(indices) * 0.7), 4.2))
        for i, m in enumerate(models):
            vals = []
            for idx in indices:
                item = next(
                    x for x in per_item_scores
                    if x["article_index"] == idx and x["prompt"] == prompt and x["model"] == m
                )
                vals.append(item[metric])
            xs = [j - 0.4 + width/2 + i*width for j in range(len(indices))]
            plt.bar(xs, vals, width, label=m)

        plt.xticks(range(len(indices)), [str(i) for i in indices])
        ylabel = metric.upper()
        plt.ylabel(ylabel)
        plt.xlabel("Article Index")
        plt.title(f"{ylabel} per Article | Prompt={prompt.title()}-shot")
        plt.legend()
        plt.tight_layout()

        path = os.path.join(outdir, f"per_article_{metric}_{prompt}.png")
        plt.savefig(path)
        plt.close()
        print(f"[Saved] {path}")


# ---------- Radar Plot ----------
def plot_radar_model_compare(agg: Dict[Tuple[str, str], Dict[str, float]], prompt: str, outdir="outputs"):
    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
    labels = [m.upper() if m != "bertscore_f1" else "BERT(F1)" for m in metrics]

    models = sorted({m for (m, p) in agg.keys() if p == prompt})
    data = np.array([[agg[(m, prompt)][k] for k in metrics] for m in models])

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    data = np.concatenate((data, data[:, [0]]), axis=1)
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, m in enumerate(models):
        ax.plot(angles, data[i], label=m)
        ax.fill(angles, data[i], alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"Model Comparison Radar ({prompt.title()}-shot)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()

    path = os.path.join(outdir, f"radar_model_{prompt}.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")


# ---------- CLI Utility ----------
if __name__ == "__main__":
    import pandas as pd

    outdir = "outputs"
    csv_path = os.path.join(outdir, "scores_by_article.csv")

    if not os.path.exists(csv_path):
        print("[Error] Missing CSV file. Run main.py first to generate it.")
    else:
        print("[Visualizer] Loading from CSV...")
        df = pd.read_csv(csv_path)
        per_item_scores = df.to_dict(orient="records")

        # Compute aggregated metrics
        agg_scores = {}
        for (m, p), group in df.groupby(["model", "prompt"]):
            agg_scores[(m, p)] = {
                "rouge1": group["rouge1"].mean(),
                "rouge2": group["rouge2"].mean(),
                "rougeL": group["rougeL"].mean(),
                "bleu": group["bleu"].mean(),
                "bertscore_f1": group["bertscore_f1"].mean() if "bertscore_f1" in group else 0,
            }

        print("[Visualizer] Generating charts...")
        for (model, prompt) in sorted(agg_scores.keys()):
            plot_agg_prompt_compare(agg_scores, model)
        for prompt in ["zero", "few"]:
            plot_agg_model_compare(agg_scores, prompt)
            for metric in ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]:
                plot_per_article_metric(per_item_scores, metric, prompt)
            plot_per_article_rouge_compare(per_item_scores, prompt)
            plot_radar_model_compare(agg_scores, prompt)

        # ---------- Summary Tables ----------
        print("Generating summary tables...")
        metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]
        models = sorted({m for (m, p) in agg_scores.keys()})
        prompts = ["zero", "few"]

        def save_table_as_image(df, title, path):
            fig, ax = plt.subplots(figsize=(7, 2 + len(df) * 0.4))
            ax.axis("off")
            ax.axis("tight")
            ax.table(
                cellText=df.round(3).values,
                colLabels=df.columns,
                rowLabels=df.index,
                loc="center"
            )
            plt.title(title, fontsize=12, pad=10)
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()
            print(f"[Saved] {path}")

        # Table 1: Average per prompt type
        table1_data = []
        for prompt in prompts:
            avg_row = []
            for k in metrics:
                vals = [agg_scores[(m, prompt)][k] for m in models]
                avg_row.append(sum(vals) / len(vals))
            table1_data.append(avg_row)
        table1 = pd.DataFrame(
            table1_data,
            index=["Zero-shot", "Few-shot"],
            columns=["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore"]
        )
        path1_csv = os.path.join(outdir, "table1_prompt_avg.csv")
        path1_img = os.path.join(outdir, "table1_prompt_avg.png")
        table1.to_csv(path1_csv)
        save_table_as_image(table1, "Average Metrics (Prompt Type)", path1_img)

        # Table 2: Average per model
        table2_data = []
        for prompt in prompts:
            row = []
            for m in models:
                vals = [agg_scores[(m, prompt)][k] for k in metrics]
                row.append(sum(vals) / len(vals))
            table2_data.append(row)
        table2 = pd.DataFrame(table2_data, index=["Zero-shot", "Few-shot"], columns=models)
        path2_csv = os.path.join(outdir, "table2_model_avg.csv")
        path2_img = os.path.join(outdir, "table2_model_avg.png")
        table2.to_csv(path2_csv)
        save_table_as_image(table2, "Average Score per Model", path2_img)

        # Table 3: Average per metric across prompts
        table3_data = []
        for m in models:
            row = []
            for k in metrics:
                vals = [agg_scores[(m, p)][k] for p in prompts]
                row.append(sum(vals) / len(vals))
            table3_data.append(row)
        table3 = pd.DataFrame(
            table3_data,
            index=models,
            columns=["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore"]
        )
        path3_csv = os.path.join(outdir, "table3_model_metric.csv")
        path3_img = os.path.join(outdir, "table3_model_metric.png")
        table3.to_csv(path3_csv)
        save_table_as_image(table3, "Metrics per Model (Averaged Prompts)", path3_img)

        print("All visualizations and tables saved in 'outputs/' folder.")
