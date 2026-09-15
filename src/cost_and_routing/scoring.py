"""Normalises free-text answers so required facts can be matched as substrings.

Handles the variation a policy question answer is likely to contain: currency written as
a symbol, as a word, or as a code; small numbers written as words; case and punctuation.
"""

import re

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "twenty-four": "24",
    "twenty-eight": "28",
    "forty-eight": "48",
}


def normalise(text: str) -> str:
    result = text.lower()

    # number words -> digits (longest keys first so "twenty-four" beats "twenty")
    for word in sorted(_NUMBER_WORDS, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(word)}\b", _NUMBER_WORDS[word], result)

    # currency: "£45", "45 gbp", "45 pounds" -> "£45"
    result = re.sub(r"gbp\s*(\d+(?:\.\d+)?)", r"£\1", result)
    result = re.sub(r"(\d+(?:\.\d+)?)\s*gbp", r"£\1", result)
    result = re.sub(r"(\d+(?:\.\d+)?)\s*pounds?", r"£\1", result)

    # strip punctuation except £ and digits/letters/spaces
    result = re.sub(r"[^\w\s£]", "", result)

    # collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


def is_correct(answer: str, required_facts: list[str]) -> bool:
    normalised_answer = normalise(answer)
    return all(normalise(fact) in normalised_answer for fact in required_facts)
