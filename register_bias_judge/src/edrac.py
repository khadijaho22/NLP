"""EDRAC loading and question filtering (identical logic to notebook 01, reused for the LoRA training pool)."""
from __future__ import annotations

import re

import pandas as pd

from .common import arabic_ratio

DEICTIC = re.compile(r"المتحدث|المتحدثة|الفيديو|الحلقة|المقطع|النص|الكلام ده|المذيع|الضيف|الراوي")


def norm_ar(s: str) -> str:
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s)
    return s.translate(str.maketrans("أإآ", "ااا")).strip()


YESNO = {norm_ar(x) for x in ["ايوه", "أيوه", "اه", "آه", "أه", "لا", "لأ", "لاء", "نعم", "ايوة"]}


def clean_passage(p) -> str:
    p = re.sub(r"\*\*[^*]*\*\*", " ", str(p))        # drop "** 0 المتحدث **" speaker headers
    return re.sub(r"\s+", " ", p).strip()


def edrac_to_qa(raw: pd.DataFrame) -> pd.DataFrame:
    """Wide EDRAC table (Question 1..10 / Answer 1..10 per passage) -> one row per QA pair (Egyptian only)."""
    raw = raw[raw["Dialect"] == "egy"].reset_index(drop=True)
    records = []
    for _, row in raw.iterrows():
        for i in range(1, 11):
            q, a = row.get(f"Question {i}"), row.get(f"Answer {i}")
            if pd.isna(q) or pd.isna(a) or not str(q).strip() or not str(a).strip():
                continue
            records.append({"source_id": row["ID"], "passage": clean_passage(row["Passage"]),
                            "qa_index": i, "question": str(q).strip(), "answer": str(a).strip()})
    return pd.DataFrame(records)


def load_edrac_qa() -> pd.DataFrame:
    from datasets import load_dataset
    return edrac_to_qa(load_dataset("MBZUAI/EDRAC", split="test").to_pandas())


def filter_qa(qa: pd.DataFrame) -> pd.DataFrame:
    """Short factual self-contained questions with Arabic-script answers (same filter as notebook 01)."""
    f = qa.copy()
    f["q_words"] = f.question.str.split().str.len()
    f["a_words"] = f.answer.str.split().str.len()
    ok = f.a_words.between(3, 25) & f.q_words.between(4, 25)
    ok &= f.answer.map(arabic_ratio) >= 0.8
    ok &= ~f.question.map(lambda q: bool(DEICTIC.search(q)))
    ok &= ~f.question.map(lambda q: norm_ar(q).startswith("هل") or " ولا " in f" {q} ")
    ok &= ~f.answer.map(lambda a: norm_ar(a).strip(" .،!؟") in YESNO)
    return f[ok].reset_index(drop=True)
