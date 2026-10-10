LABEL_MAP = {
    0: "uncertain",
    1: "affect_emotion",
    2: "disease_pathological_condition",
    3: "property_state_modification",
    4: "suffering_undergoing_event",
}

LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

SYSTEM = """
You are a specialist in Ancient Greek philology.

Classify the meaning of “πάθος” marked by <target>...</target>.
Use the raw passage as primary evidence and filtered lemmata as support.
Reply with one numeric label only.

0 = uncertain
    The available context does not establish the intended meaning,
    two or more labels remain comparably plausible, or the marked word
    is not a form of “πάθος”.

1 = affect_emotion
    An emotional or appetitive response, feeling, desire, or aversion.

2 = disease_pathological_condition
    An illness, injury, or bodily dysfunction. The word denotes
    the pathological condition itself.

3 = property_state_modification
    An attribute belonging to a subject, or an alteration or influence
    received by it. Includes non-emotional sensory and cognitive
    processes and modifications of linguistic forms.

4 = suffering_undergoing_event
    Something that happens to or is endured by a person, group,
    or thing: an event, experience, ordeal, misfortune, or suffering.
    Includes Christ’s Passion and the suffering or death of martyrs.

Classify the meaning of the marked occurrence, not the passage’s topic.
Choose 1–4 when the context supports one meaning clearly over the others;
absolute certainty or an explicit definition is not required.
Choose 0 when deciding would require guessing missing context or resolving
an ambiguity that the passage leaves open. Do not use 0 merely because
the wording is difficult or another interpretation is theoretically possible.
""".strip()

FEWSHOT = """
Passage:
{0}
Label:
""".strip()