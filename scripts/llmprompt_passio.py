# ─────────────────────────── 1. Label map ────────────────────────────
LABEL_MAP = {
    0: "suffering_undergoing",
    1: "disease_pathological_condition",
    2: "property_accident",
    3: "christs_passion",
    4: "martyrdom_saintly_suffering",
    5: "affect_emotion",
}

LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

# ─────────────────────────── 2. LLM system prompt ──────────────────────────
SYSTEM = """
You are a specialist in Latin philology.

Classify the meaning of the word “passio” (or an inflected form such as passione, passionis, passiones) in a Latin passage.

The input contains:
- the raw Latin passage, with the target word in the middle
- filtered lemmata from the same passage

Use the raw passage as the main evidence. Use the lemmata only as support.

Reply with exactly one numeric label only.

The task is to decide what kind of thing “passio” denotes in this passage.

Labels:

0 = suffering_undergoing
    A general suffered state or undergoing: pain, injury, hardship, affliction, endured condition.

1 = disease_pathological_condition
    A specifically medical suffered state: disease, syndrome, bodily disorder, pathological condition.

2 = property_accident
    A technical abstract determination: property, accident, modification, affection, or passive state of a subject.

3 = christs_passion
    The singular redemptive suffering of Christ.

4 = martyrdom_saintly_suffering
    The holy suffering or death of a martyr or saint other than Christ.

5 = affect_emotion
    An inner passion of the soul: desire, fear, anger, sorrow, appetite, or similar affective movement.

Decision order:
- If “passio” denotes Christ’s suffering, choose 3.
- Otherwise, if it denotes the suffering of a martyr or saint, choose 4.
- Otherwise, if it denotes an inner affective or appetitive state, choose 5.
- Otherwise, if it denotes a medical disorder or pathological bodily condition, choose 1.
- Otherwise, if it denotes a technical philosophical or scientific property/state, choose 2.
- Otherwise, choose 0.

Important:
- Classify the meaning of “passio” itself, not the broader topic of the passage.
- Christian context alone does not require 3: choose 3 only if the referent is Christ.
- Hagiographic context alone does not require 4: choose 4 only if the referent is a martyr or saint.
- Scholastic discussion of whether something is a passio (for example amor, spes, timor, ira, concupiscentia) usually belongs to 5 if “passio” means an affective state.
- Technical discussion of passio as opposed to actio, or as belonging to a subject, usually belongs to 2.
- Medical/diagnostic/therapeutic uses usually belong to 1.
- Generic endured suffering, when none of the more specific cases applies, belongs to 0.
""".strip()

# ─────────────────────────── 3. Few-shot examples ──────────────────────────
FEWSHOT = """
Passage:
Mors est passio maxime involuntaria.
Label: 0

Passage:
Iliaca passio intestina graviter torquebat.
Label: 1

Passage:
Passionem numeri par et impar demonstrat philosophus.
Label: 2

Passage:
In commemorationem Dominicae semel passionis quotidie offerimus.
Label: 3

Passage:
Passio Donati martyris in ecclesia recitata est.
Label: 4

Passage:
Amor et odium sunt simpliciter passiones primae.
Label: 5

Passage:
Spes non est passio.
Label: 5

Passage:
{0}
Label:
""".strip()