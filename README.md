# SLM vs LLM: LoRA fine-tuning comparison

A small research project testing whether LoRA fine-tuning lets a small language model (SLM) close the performance gap with a large language model (LLM) on a narrow classification task.

## TL;DR

Fine-tuned a 1.5B parameter SLM and a 7B parameter LLM on the same intent classification task, using identical LoRA hyperparameters for both. The accuracy gap between the two models did **not** close after fine-tuning, both gained roughly the same number of accuracy points. The SLM did deliver 2.6x lower inference latency.

| | Zero-shot baseline | After LoRA fine-tuning | Accuracy gain | Inference latency |
|---|---|---|---|---|
| SLM (Qwen2.5-1.5B-Instruct) | 48.0% | 67.0% | +19.0 pts | 311 ms / example |
| LLM (Qwen2.5-7B-Instruct, 4-bit) | 63.0% | 83.3% | +20.3 pts | 814 ms / example |

Full write-up: [`medium_article.md`](./medium_article.md)

## Motivation

SLMs are cheaper to serve and have lower inference latency than LLMs. A common claim is that task-specific fine-tuning lets SLMs close most of the performance gap with LLMs, which would make them a strong default choice for production systems. This project tests that claim directly, with a controlled comparison rather than taking it at face value.

## Task and dataset

Intent classification on [Banking77](https://huggingface.co/datasets/PolyAI/banking77): short customer support messages, labeled with one of 77 fine-grained intents (e.g. `card_not_working`, `exchange_rate`). Classification was chosen deliberately, since it has a clean, objective accuracy metric, unlike generation tasks which need subjective or noisier scoring.

- Training set: 1,500 examples (subsampled, seeded shuffle)
- Test set: 300 held-out examples
- Same train/test split and prompt format used for both models

## Models

Both models are from the same family, to keep the comparison controlled (isolating scale as the variable rather than confounding it with architecture differences):

- **SLM:** `unsloth/Qwen2.5-1.5B-Instruct`, full LoRA, bf16
- **LLM:** `unsloth/Qwen2.5-7B-Instruct`, QLoRA, 4-bit quantization

## Methodology

1. **Zero-shot baseline** — both models prompted directly on the test set, no fine-tuning, to establish a pre-training reference point
2. **LoRA fine-tuning** — identical hyperparameters used for both models:
   - rank `r=16`, `alpha=32`, dropout `0.05`
   - target modules: `q_proj, k_proj, v_proj, o_proj`
   - 3 epochs, learning rate `2e-4`, effective batch size 16
3. **Post-training evaluation** — same test set, same prompt format, same scoring function as the baseline
4. **Comparison** — accuracy, macro-F1, inference latency, and trainable parameter percentage logged for both models and compared side by side

## Tech stack

| Tool | Purpose |
|---|---|
| [Hugging Face `transformers`](https://github.com/huggingface/transformers) | model and tokenizer loading |
| [`peft`](https://github.com/huggingface/peft) | LoRA implementation |
| [`trl`](https://github.com/huggingface/trl) (`SFTTrainer`) | supervised fine-tuning loop |
| [`datasets`](https://github.com/huggingface/datasets) | dataset loading and preprocessing |
| [Unsloth](https://github.com/unslothai/unsloth) | faster, lower-memory LoRA training |
| [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) | 4-bit quantization (QLoRA) for the LLM |
| `scikit-learn` | accuracy / macro-F1 scoring |
| Google Colab (free tier, T4 GPU) | compute |

## Repo structure

```
.
├── slm_vs_llm_comparison.ipynb   # main notebook: data prep, training, evaluation, comparison
├── linkedin_post.md              # short-form write-up
├── medium_article.md             # full write-up
└── README.md
```

## Running this yourself

The notebook is built for Google Colab's free T4 GPU and trains one model at a time, since a free T4 generally can't hold both a 1.5B and a 7B model in memory simultaneously.

1. Open `slm_vs_llm_comparison.ipynb` in Colab
2. Run the setup and data-loading cells
3. In the config cell, set `WHICH_MODEL = "slm"`, then run the training and evaluation cells
4. `Runtime > Restart runtime` to clear GPU memory
5. Set `WHICH_MODEL = "llm"`, re-run setup and data-loading, then run training and evaluation again
6. Run the comparison cells at the bottom, they only read the two saved result JSON files, no model needs to be loaded in memory at that point

Results are saved to `RESULTS_DIR` (configurable in the config cell) as `slm_summary.json` and `llm_summary.json`, plus the trained LoRA adapters and a comparison plot.

## Results in detail

- **Accuracy gap did not close.** The LLM led by 15.0 points zero-shot and 16.3 points after fine-tuning, both models improved by a comparable margin (SLM +19.0, LLM +20.3).
- **Macro-F1** followed the same pattern: SLM 61.9%, LLM 78.4% after fine-tuning.
- **Inference latency:** SLM averaged 311ms per example, LLM averaged 814ms, a 2.6x difference.
- **Trainable parameters:** both models trained under 0.3% of total parameters via LoRA (SLM 0.282%, LLM 0.206%).

## Limitations

- Single training data volume (1,500 examples) tested, results may differ at larger scale
- Single task type (classification), other task types (generation, extraction) may behave differently
- Single LoRA configuration tested, larger rank may benefit the SLM disproportionately
- No statistical significance testing across multiple seeds (single run per model)

## Possible next steps

- Re-run with varying training set sizes to test whether the SLM catches up given more data
- Test a higher LoRA rank for the SLM specifically
- Per-class error analysis to see where each model's mistakes cluster
- Repeat on a generation or extraction task instead of classification

## License

Add a license of your choice (e.g. MIT) if you intend to share this publicly.
