"""Register-invariant LoRA judge: training-data construction, training loop, evaluation helpers."""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .common import REGISTERS, SEED, judge_messages, rule_check
from .models import DIGITS, chat_prompt, digit_distribution, digit_token_ids, expected_score

# ------------------------------------------------------------------------------------------ register conversion
REG_PROMPT = """Rewrite the Egyptian Arabic answer below in three other writing registers WITHOUT changing its meaning. Do not add, remove or correct anything; keep names and numbers identical.

Question (context only): <<q>>
Egyptian answer: <<a>>

Output exactly three lines and nothing else:
MSA: <Modern Standard Arabic, Arabic script>
FRANCO: <Egyptian Arabic written in Latin letters (Arabizi); digits 2 3 5 7 8 9 6 stand for ء ع خ ح غ ق ط; no Arabic script at all>
CODESWITCH: <Egyptian Arabic in Arabic script where 1-3 content words are written in English>

Example for "القاهرة عاصمة مصر":
MSA: القاهرة هي عاصمة جمهورية مصر العربية.
FRANCO: El Qahera 3asmet Masr.
CODESWITCH: Cairo هي عاصمة مصر."""

_LINE = re.compile(r"^\s*(MSA|FRANCO|CODESWITCH)\s*:\s*(.+?)\s*$", re.I | re.M)


def parse_register_output(text: str):
    """-> {"msa","franco","codeswitch"} or None if any of the three lines is missing. If the model echoes the
    example first, the LAST occurrence of each line (its own answer) wins."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    found = {k.lower(): v.strip() for k, v in _LINE.findall(text)}
    return found if all(found.get(k) for k in ("msa", "franco", "codeswitch")) else None


@torch.no_grad()
def convert_answers_local(model, tok, jobs: list[tuple[str, str]], batch_size: int = 8,
                          max_new_tokens: int = 220, desc: str = "convert"):
    """jobs = [(question, egyptian_answer)] -> list of dict|None. Greedy decoding (deterministic)."""
    from tqdm.auto import tqdm
    outs: list = [None] * len(jobs)
    prompts = [chat_prompt(tok, [{"role": "user", "content": REG_PROMPT.replace("<<q>>", q).replace("<<a>>", a)}])
               for q, a in jobs]
    order = np.argsort([len(p) for p in prompts])
    for s in tqdm(range(0, len(prompts), batch_size), desc=desc, leave=False):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = parse_register_output(t)
    return outs


def validate_conversion(answer: str, conv):
    """Keep a conversion only if all three registers pass the rule checks (script / digits)."""
    if conv is None:
        return None
    for reg in ("msa", "franco", "codeswitch"):
        if not rule_check(reg, conv[reg], answer):
            return None
    return {"egyptian": answer, **conv}


# ------------------------------------------------------------------------------------------ training data
def _jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(len(A | B), 1)


def select_train_questions(pool: pd.DataFrame, excluded_sources, max_per_passage: int = 3,
                           val_frac: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    """Passage-disjoint training/validation questions with a hard negative each.

    * passages used by the benchmark (test + reserve) are excluded -> no leakage;
    * the hard negative of a question is the answer to ANOTHER question of the same passage: same topic and
      style, but wrong for this question (no LLM needed, so it is free and reproducible);
    * train/val are split by passage."""
    rng = random.Random(seed)
    pool = pool[~pool.source_id.isin(set(excluded_sources))]
    counts = pool.groupby("source_id").size()
    passages = sorted(counts[counts >= 2].index)
    rng.shuffle(passages)
    n_val = max(1, round(val_frac * len(passages))) if len(passages) > 1 else 0
    val_set = set(passages[:n_val])
    rows = []
    for pid in passages:
        g = pool[pool.source_id == pid].reset_index(drop=True)
        take = g.sample(n=min(max_per_passage, len(g)), random_state=seed)
        for _, r in take.iterrows():
            cands = sorted({(a, q) for a, q in zip(g.answer, g.question) if a != r.answer and _jaccard(a, r.answer) <= 0.5})
            if not cands:
                continue
            neg_a, neg_q = rng.choice(cands)
            rows.append({"source_id": pid, "question": r.question, "answer": r.answer, "neg_answer": neg_a,
                         "neg_question": neg_q, "split": "val" if pid in val_set else "train"})
    return pd.DataFrame(rows)


def build_groups(sel: pd.DataFrame, converted: dict, seed: int = SEED) -> list[dict]:
    """A *group* = the same content in every register: 1 label, 4 candidate texts.
    correct group  -> candidates = the gold answer in 4 registers, label 5
    incorrect group-> candidates = the hard negative in 4 registers, label 1
    The reference is the gold answer, in a randomly chosen register (Egyptian or MSA), like the test protocol."""
    rng = random.Random(seed)
    groups = []
    for r in sel.itertuples():
        gold, neg = converted.get(r.answer), converted.get(r.neg_answer)
        if gold is None or neg is None:
            continue
        ref_reg = rng.choice(["egyptian", "msa"])
        for quality, label, src in (("correct", 5, gold), ("incorrect", 1, neg)):
            groups.append({"split": r.split, "source_id": r.source_id, "question": r.question,
                           "reference": gold[ref_reg], "ref_register": ref_reg, "quality": quality,
                           "label": label, "candidates": {reg: src[reg] for reg in REGISTERS}})
    return groups


def group_prompts(tok, group: dict, regs) -> list[str]:
    return [chat_prompt(tok, judge_messages(group["question"], group["reference"], group["candidates"][r], "plain"))
            for r in regs]


# ------------------------------------------------------------------------------------------ training
def train_lora_judge(model, tok, groups: list[dict], regs, steps: int = 120, groups_per_step: int = 2,
                     lr: float = 2e-4, lam: float = 0.0, seed: int = SEED, r: int = 16, alpha: int = 32,
                     dropout: float = 0.05, warmup: float = 0.1, log_every: int = 10):
    """LoRA fine-tuning of the judge on the 5-way digit distribution (the same readout used at inference).

    loss = CE(score digit)  +  lam * mean_over_groups Var_registers(expected score)
    The second term is the register-consistency penalty: the SAME content written in different registers must
    receive the SAME score. lam = 0 gives plain LoRA. `regs` decides which registers the model may see
    (["egyptian"] = no register augmentation)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(seed)
    rng = random.Random(seed)
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.config.use_cache = False
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_warm = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / n_warm if s < n_warm else max(0.0, (steps - s) / max(1, steps - n_warm)))
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ids = torch.tensor(digit_token_ids(tok), device=model.device)
    digits = torch.tensor(DIGITS, dtype=torch.float32, device=model.device)

    queue: list[int] = []
    log = []
    for step in range(steps):
        while len(queue) < groups_per_step:
            queue.extend(rng.sample(range(len(groups)), len(groups)))
        batch = [groups[queue.pop(0)] for _ in range(groups_per_step)]
        prompts, labels, gid = [], [], []
        for k, g in enumerate(batch):
            ps = group_prompts(tok, g, regs)
            prompts += ps
            labels += [g["label"] - 1] * len(ps)
            gid += [k] * len(ps)
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16 if use_amp else torch.bfloat16,
                            enabled=use_amp):
            logits = model(**enc).logits[:, -1, :]
        logp = F.log_softmax(logits[:, ids].float(), dim=-1)
        y = torch.tensor(labels, device=logp.device)
        ce = F.nll_loss(logp, y)
        exp = (logp.exp() * digits).sum(-1)
        gid_t = torch.tensor(gid, device=logp.device)
        cons = torch.zeros((), device=logp.device)
        multi = [k for k in range(len(batch)) if (gid_t == k).sum() > 1]
        if multi:
            cons = torch.stack([exp[gid_t == k].var(unbiased=False) for k in multi]).mean()
        loss = ce + lam * cons
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        log.append({"step": step + 1, "loss": loss.item(), "ce": ce.item(), "cons": float(cons)})
        if (step + 1) % log_every == 0:
            w = pd.DataFrame(log[-log_every:]).mean()
            print(f"step {step + 1}/{steps} | loss {w.loss:.3f} | ce {w.ce:.3f} | consistency {w.cons:.3f}")
    model.eval()
    model.config.use_cache = True
    return model, pd.DataFrame(log)


@torch.no_grad()
def evaluate_groups(model, tok, groups: list[dict], batch_size: int = 8) -> pd.DataFrame:
    """Expected judge score for every (group, register): used for the validation sanity check."""
    rows, prompts = [], []
    for gi, g in enumerate(groups):
        for reg in REGISTERS:
            prompts.append(chat_prompt(tok, judge_messages(g["question"], g["reference"], g["candidates"][reg], "plain")))
            rows.append({"group": gi, "quality": g["quality"], "label": g["label"], "register": reg})
    sc = expected_score(digit_distribution(model, tok, prompts, batch_size=batch_size, desc="val"))
    out = pd.DataFrame(rows)
    out["score"] = sc
    return out
