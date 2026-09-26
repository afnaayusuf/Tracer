"""Out-of-scope detection before any model call: questions that are not about the footage, or
that ask for things the system does not do (identify unnamed people by face, predict, act)."""
from __future__ import annotations

import re

OFF_TOPIC = re.compile(r"\b(weather|stock|recipe|write (me )?(code|a poem|an essay)|translate|capital of|who won|"
                       r"tell me a joke|your opinion|what do you think about (the )?(news|politics))\b", re.I)
PREDICT = re.compile(r"\b(will|going to|predict|forecast|expect(ed)? to)\b", re.I)
ACT = re.compile(r"\b(call|alert|notify|lock|unlock|open the|close the|turn (on|off)|send (an? )?(email|message))\b", re.I)
IDENTITY = re.compile(r"\b(who is (he|she|that|this)|what('s| is) (his|her|their) name|identify (him|her|them|the person))\b", re.I)
FOOTAGE_WORDS = re.compile(r"\b(camera|footage|video|clip|zone|conveyor|table|door|person|people|worker|someone|anyone|"
                           r"who|where|when|how many|what (happened|was|were)|did|entered|left|arrive|carry|wearing)\b", re.I)


def classify(question: str) -> tuple[str, str]:
    """returns (kind, message). kind: ok | off_topic | predict | act | identity"""
    q = question.strip()
    if ACT.search(q):
        return "act", "I only report what the cameras recorded; I can't take actions on devices or send messages."
    if PREDICT.search(q) and not re.search(r"\b(was|were|did|happened)\b", q, re.I):
        return "predict", "I can only describe what has already been recorded, not what will happen."
    if OFF_TOPIC.search(q) and not FOOTAGE_WORDS.search(q):
        return "off_topic", "That isn't something the footage can answer. Ask about people, objects, zones, times or events in the recording."
    if IDENTITY.search(q):
        return "identity", ("I don't know names unless someone has been named in the gallery; I can describe the person, "
                            "show their keyframe, and you can name them from there.")
    return "ok", ""
