LABEL_MAP = {
    0: "suffering_undergoing",
    1: "disease_pathological_condition",
    2: "property_accident",
    3: "christs_passion",
    4: "martyrdom_narrative",
    5: "affect_emotion",
}

LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

SYSTEM = """
You are a specialist in Latin philology.

Classify the meaning of the word marked by <target>...</target>.
Use the raw passage as evidence and filtered lemmata as support.
Reply with one numeric label only.

0 = suffering or undergoing, including a martyr’s suffering or death
1 = disease or pathological bodily condition
2 = technical property, accident, modification, or passive state
3 = Christ’s suffering and death
4 = a text recounting a Christian martyr’s suffering and death
5 = emotion or appetite, such as love, desire, fear, anger, or sorrow

Classify only the marked occurrence, not the passage’s topic.
Distinguish a martyr’s suffering (0) from its written account (4).
Emotions belong to 5 even in philosophical discussion or negated claims.
""".strip()

FEWSHOT = """
Passage:
Mors est <target>passio</target> maxime involuntaria.
Label: 0

Passage:
Iliaca <target>passio</target> intestina graviter torquebat.
Label: 1

Passage:
<target>Passionem</target> numeri par et impar demonstrat philosophus.
Label: 2

Passage:
In commemorationem Dominicae semel <target>passionis</target> quotidie offerimus.
Label: 3

Passage:
<target>Passio</target> Donati martyris in ecclesia recitata est.
Label: 4

Passage:
Martyr in <target>passione</target> sua tormenta patienter sustinuit.
Label: 0

Passage:
Amor et odium sunt simpliciter <target>passiones</target> primae.
Label: 5

Passage:
Spes non est <target>passio</target>.
Label: 5

Passage:
{0}
Label:
""".strip()