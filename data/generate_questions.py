"""Seeded generator for the 60-question Fernway Travel support dataset.

Every question is built from a template over the facts in policy.md, so the correct
answer is always derivable from that file. required_facts are the short strings a
correct answer must contain (scoring.is_correct checks each as a normalised substring).

Run directly to regenerate data/questions.jsonl:
    uv run python data/generate_questions.py
"""

import json
import random
from pathlib import Path
from typing import TypedDict


class Question(TypedDict):
    id: str
    question: str
    required_facts: list[str]
    difficulty: str


# Each fact is a separate required entry (count, then size or weight) so an answer is
# not marked wrong for wording the same facts differently from the policy table.
FARE_BAGGAGE = {
    "Basic": (
        ["1 small", "40 x 20 x 25 cm"],
        ["0 checked bag|no checked bag|not include any checked|no checked luggage"],
    ),
    "Standard": (["1 cabin bag", "55 x 40 x 23 cm"], ["1 checked bag|1 bag", "23 kg"]),
    "Flex": (["1 cabin bag", "55 x 40 x 23 cm"], ["2 checked bags|2 bags", "23 kg"]),
}

FARE_CANCELLATION = {
    "Basic": "non-refundable|not refundable|no refund|cannot be refunded",
    "Standard": "£75",
    "Flex": "£25",
}

FARE_CHANGE = {
    "Basic": (
        "changes not permitted|cannot be changed|no changes allowed|"
        "changes are not allowed|changes are not permitted|changing is not permitted"
    ),
    "Standard": "£45",
    "Flex": (
        "no change fee|change fee is waived|no fee to change|£0 change fee|"
        "change fee does not apply|is £0"
    ),
}


def _simple_lookup_questions() -> list[Question]:
    qs: list[Question] = []
    for fare, (cabin, checked) in FARE_BAGGAGE.items():
        qs.append(
            {
                "id": f"baggage-checked-{fare.lower()}",
                "question": f"How many checked bags are included with a {fare} fare?",
                "required_facts": checked,
                "difficulty": "simple_lookup",
            }
        )
        qs.append(
            {
                "id": f"baggage-cabin-{fare.lower()}",
                "question": f"What cabin bag am I allowed on a {fare} fare?",
                "required_facts": cabin,
                "difficulty": "simple_lookup",
            }
        )
    for fare, fee in FARE_CANCELLATION.items():
        qs.append(
            {
                "id": f"cancellation-fee-{fare.lower()}",
                "question": f"What is the cancellation fee for a {fare} fare?",
                "required_facts": [fee],
                "difficulty": "simple_lookup",
            }
        )
    for fare, fee in FARE_CHANGE.items():
        qs.append(
            {
                "id": f"change-fee-{fare.lower()}",
                "question": f"What is the change fee for a {fare} fare?",
                "required_facts": [fee],
                "difficulty": "simple_lookup",
            }
        )
    qs.append(
        {
            "id": "name-change-fee",
            "question": "How much does a name change cost?",
            "required_facts": ["£30"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "name-change-deadline",
            "question": "How close to departure can I request a name change?",
            "required_facts": ["48 hours"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "refund-original-payment-timeline",
            "question": "How long does a refund to my original payment method take?",
            "required_facts": ["7 business days"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "refund-travel-credit-timeline",
            "question": "How long does a travel credit refund take to be issued?",
            "required_facts": ["24 hours"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "refund-credit-validity",
            "question": "How long is a travel credit valid for?",
            "required_facts": ["12 months"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "refund-request-deadline",
            "question": (
                "Within how many days of the original departure date must I submit a "
                "refund request?"
            ),
            "required_facts": ["28 days"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "seat-selection-basic-standard",
            "question": "How much does standard seat selection cost on a Basic fare?",
            "required_facts": ["£12"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "seat-selection-basic-legroom",
            "question": "How much does extra-legroom seat selection cost on a Basic fare?",
            "required_facts": ["£25"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "seat-selection-deadline",
            "question": "How close to departure can I still change my seat selection?",
            "required_facts": ["3 hours"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "extra-bag-fee",
            "question": "How much does an extra checked bag cost if booked in advance?",
            "required_facts": ["£35"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "overweight-bag-fee",
            "question": "What is the overweight fee for a checked bag over 23 kg?",
            "required_facts": ["£20"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "basic-cabin-bag-dimensions",
            "question": "What are the maximum dimensions of the small bag allowed on a Basic fare?",
            "required_facts": ["40 x 20 x 25 cm"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "standard-flex-cabin-bag-dimensions",
            "question": "What are the maximum dimensions of the cabin bag on a Standard fare?",
            "required_facts": ["55 x 40 x 23 cm"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "flex-fare-types",
            "question": "What are the three fare types Fernway sells?",
            "required_facts": ["Basic", "Standard", "Flex"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "seat-selection-standard-flex",
            "question": "Does a Standard fare include free seat selection at booking?",
            "required_facts": [
                "free seat selection|seat selection is free|no charge for seat selection|"
                "complimentary seat selection"
            ],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "flex-checked-bags-weight-limit",
            "question": "What is the weight limit per checked bag on a Flex fare?",
            "required_facts": ["23 kg"],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "extra-bag-advance-booking",
            "question": (
                "Can an extra checked bag be added on the day of travel, or must it be "
                "booked in advance?"
            ),
            "required_facts": [
                "booked in advance|must be pre-booked|cannot be added on the day of travel|"
                "book it before the day of travel"
            ],
            "difficulty": "simple_lookup",
        }
    )
    qs.append(
        {
            "id": "flex-cancel-waiver-window",
            "question": (
                "How many days before departure must a Flex fare be cancelled for the "
                "cancellation fee to be waived?"
            ),
            "required_facts": ["14 days"],
            "difficulty": "simple_lookup",
        }
    )
    return qs


def _multi_step_questions() -> list[Question]:
    return [
        {
            "id": "multi-standard-cancel-and-refund",
            "question": (
                "I have a Standard fare and want to cancel. What is the cancellation fee, "
                "and how is the rest refunded?"
            ),
            "required_facts": ["£75", "travel credit"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-flex-cancel-and-refund",
            "question": (
                "I have a Flex fare and want to cancel. What is the cancellation fee, "
                "and how is the rest refunded?"
            ),
            "required_facts": ["£25", "original payment method"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-two-extra-bags-cost",
            "question": (
                "I have a Basic fare with 0 checked bags included. How much will it cost "
                "to add 2 extra checked bags in advance?"
            ),
            "required_facts": ["£70"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-standard-plus-one-extra-bag",
            "question": (
                "My Standard fare includes 1 checked bag. How much extra will a second "
                "checked bag cost me if I book it in advance?"
            ),
            "required_facts": ["£35"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-overweight-and-extra-bag",
            "question": (
                "I am adding an extra checked bag in advance and it will weigh 25 kg. "
                "What is the total of the extra bag fee and the overweight fee?"
            ),
            "required_facts": ["£55"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-standard-change-with-fare-difference",
            "question": (
                "My Standard fare change fee is £45 and the new flight costs £20 more. "
                "What is the total I will be charged for the change?"
            ),
            "required_facts": ["£65"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-name-change-and-seat",
            "question": (
                "I want a name change and a standard seat selected on my Basic fare. "
                "What is the combined cost of both?"
            ),
            "required_facts": ["£42"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-flex-waived-fee-more-than-14-days",
            "question": (
                "My Flex fare departs in 20 days and I want to cancel now. What "
                "cancellation fee applies?"
            ),
            "required_facts": [
                "waived|fee is waived|no fee is charged|fee does not apply|"
                "no cancellation fee applies"
            ],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-basic-legroom-and-extra-bag",
            "question": (
                "On a Basic fare I want extra-legroom seat selection and one extra "
                "checked bag booked in advance. What is the combined cost?"
            ),
            "required_facts": ["£60"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-flex-two-bags-included",
            "question": (
                "A Flex fare includes 2 checked bags up to 23 kg each. If I bring a third "
                "bag booked in advance, how much extra does it cost?"
            ),
            "required_facts": ["£35"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-refund-window-after-cancellation",
            "question": (
                "My flight was due to depart on 1 March and Fernway cancelled it. By what "
                "date must I submit my refund request?"
            ),
            "required_facts": ["29 march|march 29"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-standard-two-extra-bags",
            "question": (
                "My Standard fare includes 1 checked bag. If I add 2 extra checked bags "
                "in advance, what is the total extra bag cost?"
            ),
            "required_facts": ["£70"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-two-name-changes",
            "question": (
                "I need to change the names on 2 tickets. What is the total cost for "
                "both name changes?"
            ),
            "required_facts": ["£60"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-flex-change-no-fee-plus-difference",
            "question": (
                "My Flex fare change has no change fee, but the new flight costs £15 "
                "more. What is the total I pay for the change?"
            ),
            "required_facts": ["£15"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-basic-standard-seat-and-name-change",
            "question": (
                "On a Basic fare I want a name change and extra-legroom seat selection. "
                "What is the combined cost of both?"
            ),
            "required_facts": ["£55"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-two-overweight-bags",
            "question": (
                "I am checking 2 bags, both 25 kg, both booked as extra bags in advance. "
                "What is the combined overweight fee for the 2 bags?"
            ),
            "required_facts": ["£40"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-standard-change-fee-only",
            "question": (
                "My Standard fare change has no fare difference, only the change fee. "
                "What do I pay in total?"
            ),
            "required_facts": ["£45"],
            "difficulty": "multi_step_calculation",
        },
        {
            "id": "multi-flex-two-extra-bags",
            "question": (
                "My Flex fare already includes 2 checked bags. If I add 2 more extra "
                "bags in advance, what is the total extra bag cost?"
            ),
            "required_facts": ["£70"],
            "difficulty": "multi_step_calculation",
        },
    ]


def _policy_exception_questions() -> list[Question]:
    return [
        {
            "id": "exception-medical-bag",
            "question": (
                "I have a documented medical need and a Basic fare. Can I bring an extra "
                "bag, and does it cost anything?"
            ),
            "required_facts": [
                "medical",
                "free of charge|at no cost|no additional cost|no extra charge|complimentary",
            ],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-airline-cancelled-flight",
            "question": (
                "Fernway cancelled my flight. What refund am I entitled to, and is there "
                "a cancellation fee?"
            ),
            "required_facts": [
                "full refund|refunded in full|100% refund|entire fare refunded",
                "no cancellation fee|cancellation fee is waived|no fee to cancel|"
                "cancellation fee does not apply",
            ],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-airline-cancelled-timeline",
            "question": (
                "Fernway cancelled my flight. How long will the refund to my original "
                "payment method take?"
            ),
            "required_facts": ["7 business days"],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-denied-boarding",
            "question": (
                "I have a Standard fare and was denied boarding due to overbooking. What "
                "am I entitled to?"
            ),
            "required_facts": [
                "full refund|refunded in full|100% refund|entire fare refunded",
                "£150",
            ],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-flex-waived-boundary",
            "question": (
                "My Flex fare departs in exactly 14 days. Does the £25 cancellation fee "
                "still apply, or is it waived?"
            ),
            "required_facts": ["£25"],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-basic-no-changes",
            "question": "Can I change my Basic fare flight at all?",
            "required_facts": [
                "not permitted|cannot be changed|no changes allowed|"
                "changes are not allowed|not possible to change"
            ],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-name-change-within-48-hours",
            "question": (
                "My flight departs in 30 hours. Can I still request a name change?"
            ),
            "required_facts": [
                "not permit|cannot|can't|not allowed|not possible",
                "48 hours",
            ],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-overweight-max-32kg",
            "question": (
                "What is the maximum weight a checked bag can be, even with the "
                "overweight fee paid?"
            ),
            "required_facts": ["32 kg"],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-medical-bag-fare-limit",
            "question": (
                "I have a documented medical need on a Basic fare. Am I still limited to "
                "the standard cabin bag size for my medical bag?"
            ),
            "required_facts": [
                "exempt|not subject to the size limit|no size restriction applies|"
                "allowed without the standard limit"
            ],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-airline-cancelled-fare-type",
            "question": (
                "If Fernway cancels a Basic fare flight, does the usual non-refundable "
                "rule still apply?"
            ),
            "required_facts": [
                "full refund|refunded in full|100% refund|entire fare refunded",
                "regardless of fare type|regardless of the fare type|even for a Basic fare|"
                "no matter the fare type|applies to all fare types|rule does not apply",
            ],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-denied-boarding-refund-timeline",
            "question": (
                "I was denied boarding due to overbooking on a Standard fare. How long "
                "will my refund take, and what extra compensation do I get?"
            ),
            "required_facts": ["£150"],
            "difficulty": "policy_exception",
        },
        {
            "id": "exception-flex-just-under-waiver-window",
            "question": (
                "My Flex fare departs in 10 days and I want to cancel now. Is the £25 "
                "cancellation fee waived?"
            ),
            "required_facts": ["£25"],
            "difficulty": "policy_exception",
        },
    ]


def build_questions() -> list[Question]:
    questions = (
        _simple_lookup_questions() + _multi_step_questions() + _policy_exception_questions()
    )
    rng = random.Random(42)
    rng.shuffle(questions)
    return questions


def main() -> None:
    questions = build_questions()
    out_path = Path(__file__).parent / "questions.jsonl"
    with out_path.open("w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"wrote {len(questions)} questions to {out_path}")


if __name__ == "__main__":
    main()
