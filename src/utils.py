"""Shared helpers: fixed splits, metrics logging, safe submission writing."""
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import f1_score

from config import (LABELS, OUTPUT_DIR, SAMPLE_SUB_CSV, SPLITS_DIR,
                    SUBMISSION_DIR, TEST_CSV, TRAIN_CSV)


def load_meta() -> pd.DataFrame:
    """id, lang, domain, label for all training rows, without the texts (a few MB instead of GBs)."""
    return pd.read_csv(TRAIN_CSV, usecols=["id", "lang", "domain", "label"])


def load_rows(ids, chunksize: int = 20_000) -> pd.DataFrame:
    """Read only the requested training rows (with text), streaming train.csv in chunks.

    Peak RAM stays near the size of the selected rows instead of the whole ~1 GB CSV.
    Rows come back in the order of `ids`.
    """
    ids = pd.Index(ids)
    wanted = set(ids)
    parts = [chunk[chunk["id"].isin(wanted)] for chunk in pd.read_csv(TRAIN_CSV, chunksize=chunksize)]
    out = pd.concat(parts).set_index("id")
    missing = len(ids) - len(out)
    assert missing == 0, f"{missing} requested ids not found in train.csv"
    return out.loc[ids].rename_axis("id").reset_index()


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def load_splits():
    """Return (train_df, val_df) built from the fixed id files in splits/."""
    tr_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv")["id"]
    va_ids = pd.read_csv(SPLITS_DIR / "val_ids.csv")["id"]
    assert not set(tr_ids) & set(va_ids), "train and val ids overlap"
    rows = load_rows(pd.concat([tr_ids, va_ids]))
    return rows.iloc[:len(tr_ids)].reset_index(drop=True), rows.iloc[len(tr_ids):].reset_index(drop=True)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=LABELS)


def log_result(experiment: str, model: str, y_true, y_pred, notes: str = "") -> pd.DataFrame:
    """Append one row (macro-F1 + per-class F1 on the fixed val split) to outputs/results.csv."""
    per_class = f1_score(y_true, y_pred, average=None, labels=LABELS)
    row = {"time": datetime.now().isoformat(timespec="seconds"),
           "experiment": experiment, "model": model,
           "macro_f1": round(macro_f1(y_true, y_pred), 4),
           **{f"f1_{c}": round(v, 4) for c, v in zip(LABELS, per_class)},
           "notes": notes}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "results.csv"
    log = pd.concat([pd.read_csv(path), pd.DataFrame([row])]) if path.exists() else pd.DataFrame([row])
    log.to_csv(path, index=False)
    return log


def save_val_scores(name: str, ids, scores: np.ndarray, classes) -> None:
    """Save per-class validation scores (probabilities or decision values) for later ensembling."""
    d = OUTPUT_DIR / "val_scores"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(scores, columns=list(classes))
    df.insert(0, "id", list(ids))
    df.to_csv(d / f"{name}.csv", index=False)


def save_submission(ids, labels, name: str) -> None:
    """Write submissions/<name>.csv aligned to sample_submission order, with format checks."""
    sample = pd.read_csv(SAMPLE_SUB_CSV)
    pred = pd.DataFrame({"id": list(ids), "label": list(labels)})
    assert pred["id"].is_unique, "duplicate ids in predictions"
    assert set(pred["id"]) == set(sample["id"]), "prediction ids differ from sample_submission"
    assert pred["label"].isin(LABELS).all(), f"labels must be one of {LABELS}"
    sub = sample[["id"]].merge(pred, on="id", how="left")
    assert len(sub) == len(sample) and sub["label"].notna().all()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"saved {path.relative_to(SUBMISSION_DIR.parent)}  rows={len(sub)}  "
          f"label share={ {str(k): v for k, v in sub['label'].value_counts(normalize=True).round(3).items()} }")


def hstack_csr(blocks):
    """Column-stack CSR matrices without scipy's COO intermediate (which needs ~2x the output in extra RAM)."""
    blocks = [b.tocsr() for b in blocks]
    n_rows = blocks[0].shape[0]
    assert all(b.shape[0] == n_rows for b in blocks)
    counts = np.sum([np.diff(b.indptr) for b in blocks], axis=0)
    nnz = int(counts.sum())
    idx_dtype = np.int32 if max(nnz, sum(b.shape[1] for b in blocks)) < 2**31 else np.int64
    indptr = np.zeros(n_rows + 1, dtype=idx_dtype)
    np.cumsum(counts, out=indptr[1:])
    data = np.empty(nnz, dtype=blocks[0].dtype)
    indices = np.empty(nnz, dtype=idx_dtype)
    offsets = np.cumsum([0] + [b.shape[1] for b in blocks[:-1]])
    for i in range(n_rows):
        p = indptr[i]
        for b, off in zip(blocks, offsets):
            s, e = b.indptr[i], b.indptr[i + 1]
            data[p:p + e - s] = b.data[s:e]
            indices[p:p + e - s] = b.indices[s:e] + off
            p += e - s
    return sp.csr_matrix((data, indices, indptr), shape=(n_rows, sum(b.shape[1] for b in blocks)))
