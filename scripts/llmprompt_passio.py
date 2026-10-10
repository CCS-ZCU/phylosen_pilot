LABEL_MAP = {
    0: "uncertain",
    1: "suffering_undergoing",
    2: "disease_pathological_condition",
    3: "property_accident",
    4: "christs_passion",
    5: "martyrdom",
    6: "affect_emotion",
}

LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

SYSTEM = """
You are a specialist in Latin philology.

Classify the meaning of “passio” marked by <target>...</target>.
Use the raw passage as primary evidence and filtered lemmata as support.
Reply with one numeric label only.

0 = uncertain
    Insufficient context, unresolved ambiguity, or a marked word
    that is not a form of “passio”.

1 = suffering_undergoing
    Pain, distress, or hardship considered as something endured.

2 = disease_pathological_condition
    An illness, injury, or bodily dysfunction considered as an ailment,
    rather than the suffering it causes.

3 = property_accident
    An attribute belonging to a subject, or an alteration or influence
    received by it. Includes non-emotional sensory and cognitive processes.

4 = christs_passion
    Christ’s suffering and death as the historical or redemptive Passion.

5 = martyrdom
    A Christian martyr’s persecution, torture, or death for the faith,
    or a narrative recounting that martyrdom.

6 = affect_emotion
    An emotional or appetitive response, feeling, desire, or aversion.

Classify the meaning of the marked occurrence, not the passage’s topic.
Choose 1–6 when the context supports one meaning clearly over the others;
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