"""Single place for paths and experiment constants (HW3: human / ai / fusion).

Paths can be overridden with environment variables (used on Kaggle/Colab, where data, code and
outputs live in different folders): HW3_DATA_DIR, HW3_SPLITS_DIR, HW3_OUTPUT_DIR, HW3_SUBMISSION_DIR.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_env = lambda name, default: Path(os.environ.get(name, default))

DATA_DIR = _env("HW3_DATA_DIR", ROOT / "ai-generated-text-detection-hw3-ucu-ai-2026")
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

SPLITS_DIR = _env("HW3_SPLITS_DIR", ROOT / "splits")              # fixed ids, committed to git
OUTPUT_DIR = _env("HW3_OUTPUT_DIR", ROOT / "outputs")             # results log, validation probabilities
SUBMISSION_DIR = _env("HW3_SUBMISSION_DIR", ROOT / "submissions")

SEED = 42
LABELS = ["human", "ai", "fusion"]    # fixed order for reports and confusion matrices

VAL_SIZE = 10_000     # fixed validation split, used by every model
TRAIN_SIZE = 40_000   # fixed training subsample (task says 20-50k is enough)
