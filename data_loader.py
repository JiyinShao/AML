import re
import random
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple
from datasets import load_dataset

SEED = 42
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"

# Length and visualization parameters
MIN_CHARS, MAX_CHARS = 100, 2000         # Length Filtering
NUM_FOR_VIZ, NUM_FOR_DEMO = 200, 10      # Adjustable

random.seed(SEED)

# ---------- Data Structure ----------
@dataclass
class Sample:
    """Single data sample containing article and reference summary."""
    article: str
    reference: str

# ---------- Text Normalization + Noise Removal ----------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

def normalize_text(s: str) -> str:
    """Normalize text by removing HTML tags, control chars, and normalizing Unicode."""
    # Unicode normalization
    s = unicodedata.normalize("NFC", s)
    # Remove HTML tags if present
    s = _HTML_TAG_RE.sub(" ", s)
    # Normalize quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # Remove control characters
    s = _CONTROL_CHARS_RE.sub(" ", s)
    # Merge multiple spaces and trim
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s

def passes_basic_filters(article: str) -> bool:
    # Length filtering
    n = len(article)
    if n < MIN_CHARS or n > MAX_CHARS:
        return False

    # Rough gibberish ratio (overlong Unicode)
    bad_ratio = sum(1 for ch in article if ord(ch) > 0xFFFF) / max(1, n)
    if bad_ratio > 0.001:
        return False

    return True

def get_clean_samples(split: str = "train") -> Tuple[List[Sample], List[Sample]]:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split).shuffle(seed=SEED)
    cleaned: List[Sample] = []

    for ex in ds:
        art = normalize_text(ex["article"])
        ref = normalize_text(ex["highlights"])
        if passes_basic_filters(art):
            cleaned.append(Sample(art, ref))
        if len(cleaned) >= max(NUM_FOR_VIZ, NUM_FOR_DEMO + 3):
            break

    return cleaned[:NUM_FOR_VIZ], cleaned[:NUM_FOR_DEMO]

# ---------- Debug / Standalone Run ----------
if __name__ == "__main__":
    viz, demo = get_clean_samples()
    print(f"Loaded {len(viz)} for viz, {len(demo)} for demo.")
