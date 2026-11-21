"""Shared utilities for the Register-Invariant LLM Judge project.

Light dependencies only (numpy / pandas / scipy), so this module imports instantly and is fully unit-testable.
Everything that needs torch lives in src/models.py and src/lora.py.
"""
from __future__ import annotations

import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- paths / constants
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
for _p in (DATA, RESULTS, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

LEVELS = ["correct", "partial", "incorrect"]
REGISTERS = ["msa", "egyptian", "franco", "codeswitch"]
REG_LABEL = {"msa": "MSA", "egyptian": "Egyptian", "franco": "Franco", "codeswitch": "Code-switch"}
# Provisional ground truth used ONLY until human ratings exist (data/human_ratings.csv).
LEVEL_PROXY = {"correct": 5.0, "partial": 3.0, "incorrect": 1.0}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- text helpers
_AR = re.compile(r"[\u0600-\u06FF]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def arabic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(bool(_AR.match(c)) for c in letters) / max(len(letters), 1)


def digits_of(s: str) -> list[str]:
    s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return sorted(re.findall(r"\d+", s))


def rule_check(register: str, text: str, egyptian_source: str) -> bool:
    """Cheap sanity rules that a rewrite really is in the intended register (same rules as notebook 01)."""
    if register == "egyptian":
        return arabic_ratio(text) >= 0.8
    if register == "msa":
        return arabic_ratio(text) >= 0.9 and digits_of(text) == digits_of(egyptian_source)
    if register == "franco":            # digits double as letters in Franco, so no digit check
        return not _AR.search(text) and bool(_LATIN_WORD.search(text))
    if register == "codeswitch":
        return bool(_AR.search(text)) and bool(_LATIN_WORD.search(text)) and digits_of(text) == digits_of(egyptian_source)
    raise ValueError(register)


# --------------------------------------------------------------------------- data loading
def load_items(only_selected: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Candidate answers (one row per question x level x register) + ground truth.

    * applies the annotators' corrections from data/equivalence_check_done.csv if that file exists;
    * adds `gold` = mean human rating if data/human_ratings.csv exists, else a provisional proxy
      from the quality level. `gold_source` says which one was used (report it!).
    """
    cand = pd.read_csv(DATA / "candidates_long.csv")
    if only_selected:
        cand = cand[cand.status == "selected"].copy()

    eq_path = DATA / "equivalence_check_done.csv"
    n_fixed = n_flagged = 0
    cand["equiv_flagged"] = False
    if eq_path.exists():
        eq = pd.read_csv(eq_path).fillna("")
        fix = eq[eq.corrected_answer.astype(str).str.strip() != ""].set_index("item_id").corrected_answer
        mask = cand.item_id.isin(fix.index)
        cand.loc[mask, "answer"] = cand.loc[mask, "item_id"].map(fix)
        n_fixed = int(mask.sum())
        bad = set(eq[eq.human_same_meaning.astype(str).str.lower().isin(["n", "no", "0", "false"])].item_id) - set(fix.index)
        cand["equiv_flagged"] = cand.item_id.isin(bad)
        n_flagged = int(cand.equiv_flagged.sum())

    hr_path = DATA / "human_ratings.csv"
    if hr_path.exists():
        hr = pd.read_csv(hr_path)[["item_id", "human_mean"]]
        cand = cand.merge(hr, on="item_id", how="left")
    else:
        cand["human_mean"] = np.nan
    proxy = cand.level.map(LEVEL_PROXY)
    cand["gold"] = cand.human_mean.fillna(proxy)
    cand["gold_source"] = np.where(cand.human_mean.notna(), "human", "proxy")

    if verbose:
        print(f"items: {len(cand)} | questions: {cand.question_id.nunique()} | "
              f"gold from human ratings: {(cand.gold_source == 'human').sum()} / {len(cand)} "
              f"(rest = provisional level proxy) | corrections applied: {n_fixed} | flagged non-equivalent: {n_flagged}")
    return cand.reset_index(drop=True)


def build_pairs(items: pd.DataFrame, ref_registers=("egyptian", "msa")) -> pd.DataFrame:
    """(question, reference, candidate) rows for every evaluator.

    The reference is the *correct* answer of the same question written in `ref_register`.
    Using two reference registers separates plain register *mismatch* with the reference from real bias.
    """
    correct = items[items.level == "correct"]
    ref_tbl = correct.pivot(index="question_id", columns="register", values="answer")
    parts = []
    for rr in ref_registers:
        d = items.copy()
        d["ref_register"] = rr
        d["reference"] = d.question_id.map(ref_tbl[rr])
        parts.append(d)
    pairs = pd.concat(parts, ignore_index=True)
    assert pairs.reference.notna().all(), "missing reference answers (check candidates_long.csv)"
    pairs["pair_id"] = pairs.item_id + "|ref=" + pairs.ref_register
    pairs = pairs.rename(columns={"answer": "candidate"})
    cols = ["pair_id", "item_id", "question_id", "level", "register", "ref_register",
            "question", "reference", "candidate", "gold", "gold_source"]
    return pairs[cols].reset_index(drop=True)


def save_scores(df: pd.DataFrame, family: str) -> Path:
    """df columns: pair_id, evaluator, score. Stored as results/scores_<family>.csv"""
    out = df.copy()
    if "family" not in out.columns:          # callers may set a finer family per evaluator
        out["family"] = family
    path = RESULTS / f"scores_{family}.csv"
    out.to_csv(path, index=False)
    return path


def load_scores() -> pd.DataFrame:
    files = sorted(RESULTS.glob("scores_*.csv"))
    if not files:
        raise FileNotFoundError("no results/scores_*.csv yet: run notebooks 02, 03 and 05 first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# --------------------------------------------------------------------------- LLM-judge prompts
RUBRIC_PLAIN = (
    "Rate how well the candidate answer answers the question, compared with the reference answer, on a 1-5 scale:\n"
    "5 = fully correct and complete\n"
    "4 = correct, with a minor omission\n"
    "3 = partially correct (key information missing, nothing false)\n"
    "2 = mostly wrong, with a trace of correct information\n"
    "1 = wrong or irrelevant"
)
REGISTER_AGNOSTIC_NOTE = (
    "Judge ONLY the information content. The candidate may be written in Modern Standard Arabic, Egyptian Arabic, "
    "Franco-Arabic (Arabic in Latin letters), or mixed with English. Dialect, script, spelling and style must NOT "
    "change the score: two answers that carry the same information must get the same score."
)
JUDGE_VARIANTS = ("plain", "agnostic")


def judge_messages(question: str, reference: str, candidate: str, variant: str = "plain") -> list[dict]:
    """One single user message (works with every chat template, including ones without a system role)."""
    if variant not in JUDGE_VARIANTS:
        raise ValueError(variant)
    note = ("\n\n" + REGISTER_AGNOSTIC_NOTE) if variant == "agnostic" else ""
    text = (
        f"You are a strict grader of answers to questions.\n\n"
        f"Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {candidate}\n\n"
        f"{RUBRIC_PLAIN}{note}\n\nRespond with a single digit from 1 to 5 and nothing else."
    )
    return [{"role": "user", "content": text}]


# --------------------------------------------------------------------------- register-bias statistics
def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def _quality_gap(g: pd.DataFrame) -> float:
    """mean score of correct answers - mean score of incorrect answers (the evaluator's own 'yardstick')."""
    c = g.loc[g.level == "correct", "score"].mean()
    i = g.loc[g.level == "incorrect", "score"].mean()
    return float(c - i)


def quality_gaps(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"evaluator": ev, "ref_register": rr, "quality_gap": _quality_gap(g)}
            for (ev, rr), g in df.groupby(["evaluator", "ref_register"])]
    return pd.DataFrame(rows)


def pairwise_register_gaps(df: pd.DataFrame, level: str = "correct", n_boot: int = 10000, seed: int = SEED) -> pd.DataFrame:
    """Paired comparison of every register pair on the SAME questions (default: correct answers only,
    where all registers carry identical information, so any difference is register bias).

    gap = score(a) - score(b). `norm_gap` = gap / quality_gap: how big the register effect is compared with the
    difference between a correct and an incorrect answer for that same evaluator (comparable across scales).
    """
    rng = np.random.default_rng(seed)
    qg = quality_gaps(df).set_index(["evaluator", "ref_register"]).quality_gap
    out = []
    for (ev, rr), g in df[df.level == level].groupby(["evaluator", "ref_register"]):
        wide = g.pivot(index="question_id", columns="register", values="score")
        rows = []
        for i, a in enumerate(REGISTERS):
            for b in REGISTERS[i + 1:]:
                d = (wide[a] - wide[b]).dropna().to_numpy()
                n = len(d)
                if n < 3:
                    continue
                boot = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                sd = d.std(ddof=1)
                try:
                    p = stats.wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    p = 1.0
                q = qg.get((ev, rr), np.nan)
                rows.append({"evaluator": ev, "ref_register": rr, "a": a, "b": b, "n": n,
                             "gap": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "cohen_dz": d.mean() / sd if sd > 0 else 0.0,
                             "p_wilcoxon": p, "quality_gap": q,
                             # only meaningful if the evaluator ranks correct ABOVE incorrect (quality_gap > 0)
                             "norm_gap": d.mean() / q if (not np.isnan(q) and q > 0) else np.nan})
        if rows:
            padj = holm([r["p_wilcoxon"] for r in rows])
            for r, pa in zip(rows, padj):
                r["p_holm"] = pa
            out.extend(rows)
    return pd.DataFrame(out)


def gaps_vs_msa(pw: pd.DataFrame) -> pd.DataFrame:
    """Register gaps oriented as score(MSA) - score(other register): positive = the other register is penalised.
    (REGISTERS lists MSA first, so MSA is always column `a` in pairwise_register_gaps.)"""
    return pw[pw.a == "msa"].rename(columns={"b": "other"}).drop(columns=["a"]).reset_index(drop=True)


def friedman_by_register(df: pd.DataFrame, level: str = "correct") -> pd.DataFrame:
    """Omnibus test: do the 4 registers differ at all? (Friedman, paired by question)."""
    rows = []
    for (ev, rr), g in df[df.level == level].groupby(["evaluator", "ref_register"]):
        wide = g.pivot(index="question_id", columns="register", values="score").dropna()
        if len(wide) < 3 or (wide.std(axis=1) == 0).all():      # no variation across registers at all
            p, chi = 1.0, 0.0
        else:
            chi, p = stats.friedmanchisquare(*[wide[r] for r in REGISTERS])
        rows.append({"evaluator": ev, "ref_register": rr, "friedman_chi2": chi, "friedman_p": p, "n": len(wide)})
    return pd.DataFrame(rows)


def inconsistency(df: pd.DataFrame, level: str = "correct") -> pd.DataFrame:
    """Mean per-question std of the score across the 4 registers (0 = perfectly register-invariant)."""
    qg = quality_gaps(df).set_index(["evaluator", "ref_register"]).quality_gap
    rows = []
    for (ev, rr), g in df[df.level == level].groupby(["evaluator", "ref_register"]):
        wide = g.pivot(index="question_id", columns="register", values="score").dropna()
        s = wide.std(axis=1, ddof=1).mean()
        q = qg.get((ev, rr), np.nan)
        rows.append({"evaluator": ev, "ref_register": rr, "inconsistency": s,
                     "inconsistency_norm": s / q if (not np.isnan(q) and q > 0) else np.nan})
    return pd.DataFrame(rows)


def validity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation with the ground truth (`gold`), overall and inside each register.
    A fair judge must stay accurate in EVERY register, not just on average."""
    rows = []
    for (ev, rr), g in df.groupby(["evaluator", "ref_register"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # constant scores -> NaN correlation, handled below
            row = {"evaluator": ev, "ref_register": rr, "spearman_all": stats.spearmanr(g.score, g.gold)[0]}
        for reg in REGISTERS:
            gg = g[g.register == reg]
            row[f"spearman_{reg}"] = stats.spearmanr(gg.score, gg.gold)[0] if gg.score.nunique() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def flip_matrix(df: pd.DataFrame, evaluator: str, ref_register: str) -> pd.DataFrame:
    """Decision-level unfairness: P(score of a CORRECT answer in register r1 < score of an INCORRECT answer
    in register r2) for the same question (ties count 0.5). Off-diagonal mass = register-induced ranking errors."""
    g = df[(df.evaluator == evaluator) & (df.ref_register == ref_register)]
    cor = g[g.level == "correct"].pivot(index="question_id", columns="register", values="score")
    inc = g[g.level == "incorrect"].pivot(index="question_id", columns="register", values="score")
    qs = cor.index.intersection(inc.index)
    m = pd.DataFrame(index=REGISTERS, columns=REGISTERS, dtype=float)
    for r1 in REGISTERS:
        for r2 in REGISTERS:
            c, i = cor.loc[qs, r1].to_numpy(), inc.loc[qs, r2].to_numpy()
            m.loc[r1, r2] = np.mean((c < i) + 0.5 * (c == i))
    m.index.name = "correct in"
    m.columns.name = "incorrect in"
    return m


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per evaluator x reference register: the numbers for the report table."""
    qg = quality_gaps(df)
    pw = pairwise_register_gaps(df)
    fr = friedman_by_register(df)
    inc = inconsistency(df)
    val = validity_table(df)
    rows = []
    for _, r in qg.iterrows():
        ev, rr = r.evaluator, r.ref_register
        sub = pw[(pw.evaluator == ev) & (pw.ref_register == rr)]
        vs_msa = sub[(sub.a == "msa") | (sub.b == "msa")].copy()
        # orient every gap as score(MSA) - score(other): positive = the other register is penalised
        vs_msa["oriented"] = np.where(vs_msa.a == "msa", vs_msa.norm_gap, -vs_msa.norm_gap)
        fm = flip_matrix(df, ev, rr).to_numpy()
        same = np.mean(np.diag(fm))
        cross = np.mean(fm[~np.eye(len(REGISTERS), dtype=bool)])
        rows.append({
            "evaluator": ev, "ref_register": rr, "quality_gap": r.quality_gap,
            "max_penalty_vs_msa_norm": vs_msa.oriented.max() if len(vs_msa) else np.nan,
            "worst_gap_p_holm": sub.p_holm.min() if len(sub) else np.nan,
            "friedman_p": fr[(fr.evaluator == ev) & (fr.ref_register == rr)].friedman_p.iloc[0],
            "inconsistency_norm": inc[(inc.evaluator == ev) & (inc.ref_register == rr)].inconsistency_norm.iloc[0],
            "flip_same_register": same, "flip_cross_register": cross,
            "spearman_all": val[(val.evaluator == ev) & (val.ref_register == rr)].spearman_all.iloc[0],
        })
    return pd.DataFrame(rows)
