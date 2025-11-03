from data_loader import get_clean_samples
from llm_summarizer import summarize_with_models, eval_and_save, MODELS
from visualizer import (
    visualize_lengths,
    plot_agg_prompt_compare,
    plot_agg_model_compare,
    plot_per_article_metric
)

def main():
    # ---------- 1. Load and clean dataset ----------
    print("Loading & cleaning samples…")
    viz_samples, demo_samples = get_clean_samples(split="train")
    print(f"Prepared {len(viz_samples)} for viz, {len(demo_samples)} for demo.")

    # ---------- 2. Visualize dataset statistics ----------
    print("Visualizing dataset length distributions…")
    visualize_lengths(viz_samples)

    # ---------- 3. Generate summaries ----------
    print("Running summarization with multiple models…")
    records = summarize_with_models(demo_samples, MODELS)

    # ---------- 4. Evaluate results ----------
    print("Evaluating outputs (ROUGE-1/2/L, BLEU, BERTScore)…")
    res = eval_and_save(records)
    agg, per_item = res["agg"], res["per_item"]

    # ---------- 5. Plot comparisons ----------
    print("Plotting aggregate comparisons…")
    for m in MODELS:
        plot_agg_prompt_compare(agg, m)
    for p in ["zero", "few"]:
        plot_agg_model_compare(agg, p)

    print("Plotting per-article comparisons…")
    for metric in ["rougeL", "bleu", "bertscore_f1"]:
        for prompt in ["zero", "few"]:
            plot_per_article_metric(per_item, metric, prompt)

# ---------- Entry Point ----------
if __name__ == "__main__":
    main()
