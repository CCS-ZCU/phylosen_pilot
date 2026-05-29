# ─────────────────────────── 1. Label map ────────────────────────────
LABEL_MAP = {
    0: "affect_emotion",
    1: "bodily_affection",
    2: "property_state_modification",
    3: "suffering_event",
    4: "sacred_suffering",
}

LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

# ─────────────────────────── 2. LLM system prompt ──────────────────────────
SYSTEM = """
You are a specialist in Ancient Greek philology.

Classify the meaning of the word “πάθος” (or an inflected form such as πάθους, πάθη, πάθεσι, πάθεος) in a Greek passage.

The input contains:
- the raw Greek passage, with the target word in the middle
- filtered lemmata from the same passage

Use the raw passage first. Use the lemmata only to help.

Reply with exactly one numeric label only.

Labels:

0 = affect_emotion
    passion of the soul: desire, anger, fear, grief, jealousy, irrational impulse

1 = bodily_affection
    bodily condition: disease, lesion, inflammation, disorder, impairment

2 = property_state_modification
    technical or abstract use: quality, state, modification, affection, accident, relation, grammatical or philosophical category

3 = suffering_event
    suffered event: calamity, disaster, ordeal, civic misfortune, what happened to someone

4 = sacred_suffering
    sacred suffering: Christ’s Passion, or the suffering of martyrs and saints

Rules:
- Christ, martyrs, saints, sacred suffering -> 4
- bodily disease or disorder -> 1
- technical, philosophical, grammatical, scientific, abstract use -> 2
- calamity or suffered event -> 3
- passion of the soul -> 0

Important:
- Classify the meaning of πάθος itself, not the general topic.
- περὶ παθῶν in scholarly or grammatical titles usually -> 2
- passions of the soul usually -> 0
- medical contexts usually -> 1
- civic or personal misfortune usually -> 3
- sacred Christian contexts usually -> 4
""".strip()

# ─────────────────────────── 3. Few-shot examples ──────────────────────────
FEWSHOT = """
Passage:
μὴ ἐν πάθει ἐπιθυμίας, καθάπερ τὰ ἔθνη τὰ μὴ εἰδότα τὸν Θεόν.
Label: 0

Passage:
ἐπὶ δὲ μεγάλῳ πάθεϊ, ὁκοῖον ἡ φλεγμονή.
Label: 1

Passage:
πάθος λέγεται ἡ ποιότης, καθ᾿ ἣν ἀλλοιοῦσθαι ἐνδέχεται.
Label: 2

Passage:
τὸ πάθος τῆς πόλεως χρήμασιν ὁ βασιλεὺς παρεμυθήσατο.
Label: 3

Passage:
διὰ γὰρ τοῦ Πάθους τοῦ κυρίου καὶ τῆς ἀναστάσεως αὐτοῦ.
Label: 4

Passage:
Ἡρωδιανὸς περὶ παθῶν.
Label: 2

Passage:
τὸν ἀμνὸν τῷ πάθει.
Label: 4

Passage:
τὰ πρωτουργὰ πάθη.
Label: 2

Passage:
{0}
Label:
""".strip()