LABEL_MAP = {
    0: "affect_emotion",
    1: "bodily_affection",
    2: "property_state_modification",
    3: "suffering_undergoing_event",
}

LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

SYSTEM = """
You are a specialist in Ancient Greek philology.

Classify the meaning of the word marked by <target>...</target>.
Use the raw passage as evidence and filtered lemmata as support.
Reply with one numeric label only.

0 = emotion or appetite, such as desire, anger, fear, or grief
1 = bodily condition or disturbance, such as disease, injury, or inflammation
2 = technical property, accident, modification, or passive state,
    including philosophical, scientific, and grammatical uses
3 = suffering, ordeal, misfortune, or an event undergone,
    including Christ’s Passion and the suffering or death of martyrs

Classify only the marked occurrence, not the passage’s topic.
Distinguish a bodily condition (1) from the experience of suffering (3).
Emotions belong to 0 even in philosophical or Christian discussion.
""".strip()

FEWSHOT = """
Passage:
μὴ ἐν <target>πάθει</target> ἐπιθυμίας, καθάπερ τὰ ἔθνη τὰ μὴ εἰδότα τὸν Θεόν.
Label: 0

Passage:
ἐπὶ δὲ μεγάλῳ <target>πάθεϊ</target>, ὁκοῖον ἡ φλεγμονή.
Label: 1

Passage:
<target>πάθος</target> λέγεται ἡ ποιότης, καθ᾿ ἣν ἀλλοιοῦσθαι ἐνδέχεται.
Label: 2

Passage:
τὸ <target>πάθος</target> τῆς πόλεως χρήμασιν ὁ βασιλεὺς παρεμυθήσατο.
Label: 3

Passage:
διὰ γὰρ τοῦ <target>Πάθους</target> τοῦ κυρίου καὶ τῆς ἀναστάσεως αὐτοῦ.
Label: 3

Passage:
{0}
Label:
""".strip()