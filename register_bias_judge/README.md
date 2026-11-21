# Register-Invariant LLM Judges for Egyptian Arabic, Franco and Code-switching

Course project (Topic 4), Milestone 1. **Research question:** do automatic evaluators score *equivalent* answers
differently depending on register (MSA, Egyptian, Franco, code-switched), and can a small judge be trained to assess
content independently of register?

## Repository layout
```
01_dataset_creation.ipynb          EDRAC -> 30 questions x 3 quality levels x 4 registers (+ annotation sheets)
02_embedding_encoder_baselines.ipynb   3 text-embedding models + BERTScore with 3 contextual encoders
03_llm_judge_baselines.ipynb       2 open LLM judges (4-bit) x 2 prompt variants
04_limitation_experiment.ipynb     register-bias measurements, figures, tables (no GPU)
05_lora_judge.ipynb                proposed method: QLoRA judge with a register-consistency loss + ablations
src/common.py                      data loading, pair construction, judge prompt, bias statistics (numpy/scipy only)
src/models.py                      embedding / BERTScore evaluators, logit-based LLM judge scorer
src/lora.py                        training-data construction, LoRA training loop, evaluation helpers
src/edrac.py                       EDRAC loading + question filter (same logic as notebook 01)
data/                              dataset, LLM cache (llm_cache.jsonl), annotation files
results/                           scores_*.csv, bias_summary.csv, figures/
```

## Setup (Google Colab, free T4)
1. Upload `register_bias_judge.zip` to Colab and run `!unzip -q register_bias_judge.zip`.
2. Open the notebooks from inside the `register_bias_judge/` folder (each notebook `chdir`s into it automatically).
3. Add Colab Secrets: `GEMINI_API_KEY` (notebook 01 only) and `HF_TOKEN` (Llama is gated; EDRAC may need a login).
4. Each notebook installs its own dependencies in its first cells; the full list is in `requirements.txt`.
5. **Seeds:** every notebook calls `set_seed(42)`; LLM outputs used to build the dataset are cached in `data/llm_cache.jsonl`,
   so re-running notebook 01 reproduces the same data without new API calls.

## How to produce each result
| Result | Command (or open and Run all) | Output |
|---|---|---|
| Dataset + annotation sheets | `jupyter nbconvert --to notebook --execute 01_dataset_creation.ipynb` | `data/candidates_long.csv`, `data/annotator_{1,2}.csv`, `data/equivalence_check.csv` |
| Inter-annotator agreement | last cell of notebook 01, after annotators return their files | `data/human_ratings.csv` (weighted kappa printed) |
| Embedding + encoder baselines | `jupyter nbconvert --to notebook --execute 02_embedding_encoder_baselines.ipynb` | `results/scores_embedding_encoder.csv` |
| LLM-judge baselines | `jupyter nbconvert --to notebook --execute 03_llm_judge_baselines.ipynb` | `results/scores_llm.csv`, `results/llm_probs/` |
| Limitation experiment (register bias) | `jupyter nbconvert --to notebook --execute 04_limitation_experiment.ipynb` | `results/bias_summary.csv`, `results/pairwise_register_gaps.csv`, `results/figures/fig1..3` |
| Proposed method (LoRA judge + ablation) | `jupyter nbconvert --to notebook --execute 05_lora_judge.ipynb` | `results/scores_lora.csv`, `results/lora_validation.csv`, `checkpoints/` |

Run order: 01 -> (annotation) -> 02 -> 03 -> 05 -> 04. Notebook 04 must be re-run after 05 to include the LoRA judges.
If notebook 01 was run elsewhere, copy its `data/` folder into this repository folder first.

## Data
* Source: EDRAC (Abo Mokh et al., Findings of EMNLP 2026), Egyptian subset. **Licence: TODO - copy it from the dataset card and state it here.**
* 30 benchmark questions (+6 reserve), passage-disjoint from the LoRA training questions.
* Each question has answers at three quality levels (`correct`, `partial`, `incorrect`) in four registers. Only the Egyptian
  gold answer is original; partial / incorrect answers and the MSA / Franco / code-switched rewrites are generated from it by an LLM
  and then checked (rule checks, LLM equivalence check, human review of equivalence).
* Annotation: two annotators rate content correctness 1-5 with `data/rubric.md`; agreement is reported by notebook 01 (**TODO: paste numbers**).
* Until human ratings exist, notebook 04 uses the quality level as a provisional ground truth and says so in its first output line.

## Models
Text embedding: BGE-M3, multilingual-E5-large, Qwen3-Embedding-0.6B. Contextual encoders (BERTScore): CAMeLBERT-mix, MARBERTv2, XLM-R-large.
LLM judges (4-bit): Qwen3-8B, Llama-3.1-8B-Instruct. LoRA judge: Qwen2.5-3B-Instruct (QLoRA). Training-data rewrites: Qwen3-8B (local).
Check each model's licence before redistribution. No result depends on paid access: dataset generation used a free-tier API
and its outputs are cached in the repository.

## Disclosure
Code in this repository was written with the help of an AI assistant (Claude). It was tested on unit-level fixtures and
tiny local models; results must be checked by the author. All references in the report must be verified manually.
