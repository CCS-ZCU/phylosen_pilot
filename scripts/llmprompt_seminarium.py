# ─────────────────────────── 1. Label map ────────────────────────────
LABEL_MAP = {
    0: "horticulture",
    1: "education"
}

LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

# ─────────────────────────── 2. LLM system prompt with descriptions ──────────────────────────
SYSTEM = """
You are a Latin philologist and semantic analyst.

Your task is to classify how the word “seminarium” is used in a Latin passage.

Choose **exactly one** of the numbered categories below and reply with the **numeric ID only** (no explanations, no punctuation).

Categories:

  0 - horticulture
  1 - education

""".strip()

# ─────────────────────────── 3. Few-shot examples ──────────────────────────
FEWSHOT = """
Passage:
Monasterium quoddam tamquam seminarium religionis et pietatis instituit.
Label: 1

Passage:
Agricola seminarium plantarum ex variis regionibus colligebat.
Label: 0

Passage:
Seminarium novorum sacerdotum Romae erectum est sub auspiciis Gregorii XVI.
Label: 1

Passage:
{0}
Label:
""".strip()
