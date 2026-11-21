"""torch-dependent helpers: embedding / BERTScore evaluators and the logit-based LLM judge scorer."""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch

from .common import judge_messages

DIGITS = [1, 2, 3, 4, 5]


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hf_login() -> None:
    """Log in to Hugging Face from $HF_TOKEN or Colab Secrets (needed for gated models such as Llama)."""
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception:
            tok = None
    if tok:
        from huggingface_hub import login
        login(token=tok)
    else:
        print("No HF_TOKEN found (env or Colab Secrets): gated models such as Llama will fail to download.")


# ----------------------------------------------------------------------------------- embedding evaluators
def embedding_cosine(model_id: str, pairs: pd.DataFrame, prefix: str = "", batch_size: int = 32) -> np.ndarray:
    """Cosine similarity between reference and candidate under a text-embedding model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=device(), trust_remote_code=False)
    texts = sorted(set(pairs.reference) | set(pairs.candidate))
    emb = model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    lookup = {t: i for i, t in enumerate(texts)}
    ref = emb[[lookup[t] for t in pairs.reference]]
    cand = emb[[lookup[t] for t in pairs.candidate]]
    del model
    free_gpu()
    return (ref * cand).sum(axis=1)


# ----------------------------------------------------------------------------------- contextual-encoder evaluators
def bertscore_f1(model_type: str, num_layers: int, candidates: list[str], references: list[str],
                 batch_size: int = 32) -> np.ndarray:
    """BERTScore F1 (greedy token matching on contextual embeddings from `num_layers`), no IDF, no baseline rescaling."""
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type=model_type, num_layers=num_layers, batch_size=batch_size,
                        device=device(), idf=False, rescale_with_baseline=False)
    tok = scorer._tokenizer
    if tok.model_max_length is None or tok.model_max_length > 100_000:    # some tokenizers ship "infinite" max length
        tok.model_max_length = 512
    _, _, f1 = scorer.score(candidates, references)
    del scorer
    free_gpu()
    return f1.cpu().numpy()


# ----------------------------------------------------------------------------------- LLM judge
def load_causal_lm(name: str, four_bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                      # we read the logits at the last position
    kw = {}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if not four_bit:
        model = model.to(device())
        if torch.cuda.is_available():
            model = model.half()
    model.eval()
    return model, tok


def chat_prompt(tok, messages: list[dict]) -> str:
    """Chat-template text ending right where the assistant must answer. Qwen3's thinking mode is switched off."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def digit_token_ids(tok) -> list[int]:
    ids = []
    for d in DIGITS:
        t = tok.encode(str(d), add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"'{d}' is not a single token for {tok.name_or_path!r}: {t}")
        ids.append(t[0])
    return ids


@torch.no_grad()
def digit_distribution(model, tok, prompts: list[str], batch_size: int = 8, desc: str = "") -> np.ndarray:
    """P(score = 1..5): softmax over the logits of the five digit tokens at the answer position.

    Reading probabilities instead of parsing generated text means no parsing failures, deterministic scores,
    and a continuous expected score (a 4.6 vs 4.2 difference is information, not noise)."""
    from tqdm.auto import tqdm
    ids = torch.tensor(digit_token_ids(tok), device=model.device)
    probs = np.zeros((len(prompts), len(DIGITS)), dtype=float)
    order = np.argsort([len(p) for p in prompts])            # similar lengths together -> less padding
    for s in tqdm(range(0, len(prompts), batch_size), desc=desc, leave=False):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        probs[idx] = torch.softmax(logits[:, ids].float(), dim=-1).cpu().numpy()
    return probs


def expected_score(probs: np.ndarray) -> np.ndarray:
    return probs @ np.array(DIGITS, dtype=float)


def judge_pairs(model, tok, pairs: pd.DataFrame, variant: str = "plain", batch_size: int = 8, desc: str = ""):
    prompts = [chat_prompt(tok, judge_messages(r.question, r.reference, r.candidate, variant))
               for r in pairs.itertuples()]
    probs = digit_distribution(model, tok, prompts, batch_size=batch_size, desc=desc)
    return expected_score(probs), probs
