"""
generate_dataset.py
--------------------
Generates a combined synthetic dataset for a calendar-assistant NLU pipeline
covering three tasks:

  Task 1 - Intent Classification      (vanilla unidirectional LSTM, many-to-one)
  Task 2 - Slot Filling / NER tagging (BiLSTM, BIO scheme)
  Task 3 - String Normalization       (plain LSTM Seq2Seq, no attention)

All three tasks share the SAME input sentence, so one script produces all
labels needed for all three architectures out of a single example.

Final per-example output is deliberately expressed as ONE STRING (not three
separate label columns) to force students to practice string construction /
parsing (splitting on delimiters, zero-padding, building canonical date
strings by hand instead of just calling a date library):

    target_string = "<INTENT>|<BIO_TAGS_SPACE_SEPARATED>|<CANONICAL_DATETIME_OR_NA>"

Example:
    raw_text       = "remind me to call sara tomorrow at 3pm"
    target_string  = "SET_REMINDER|O O O B-EVENT O B-DATE O B-TIME|2026-06-16 15:00"

Run:
    python3 generate_dataset.py

Outputs (written to ./output/):
    train.json, val.json, test.json   -- the actual dataset (naive random split)
    combined.csv                       -- everything in one CSV for inspection
    stats.json                         -- basic dataset statistics

NOTE TO WHOEVER READS THIS FILE TOP TO BOTTOM:
This script is intentionally documented so coordinators CAN read it and see
exactly how the data was built -- there is no hidden trickery in the labels
themselves. The "gotchas" mentioned in TASK_BRIEF.md / NOTES_FOR_COORDINATORS.md
come from *dataset composition choices* (how templates are distributed, how
the split is performed, how many paraphrases each class gets) -- all of which
are visible right here if you look closely at TEMPLATES and SPLIT_STRATEGY.
"""

import json
import random
import csv
import os
from datetime import date, timedelta

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference "today" for resolving relative dates. Fixed so the dataset is
# reproducible regardless of when you actually run this script.
# ---------------------------------------------------------------------------
REF_DATE = date(2026, 6, 15)  # a Monday

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

NAMES = ["alex", "priya", "john", "meera", "chen", "fatima", "liam", "sara",
         "raj", "emma", "noah", "divya", "wei", "omar", "lucia", "ken"]

EVENTS = ["meeting", "call", "dentist appointment", "team sync", "project review",
          "lunch", "interview", "study session", "doctor's appointment",
          "yoga class", "flight", "presentation", "workshop", "checkup"]

# ---------------------------------------------------------------------------
# Date expression generators.
# Each function returns (natural_text, canonical_YYYY-MM-DD or None)
# ---------------------------------------------------------------------------

def next_weekday(ref, weekday_idx):
    days_ahead = weekday_idx - ref.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return ref + timedelta(days=days_ahead)


def gen_date_today():
    return "today", REF_DATE.isoformat()


def gen_date_tomorrow():
    return "tomorrow", (REF_DATE + timedelta(days=1)).isoformat()


def gen_date_day_after_tomorrow():
    return "the day after tomorrow", (REF_DATE + timedelta(days=2)).isoformat()


def gen_date_next_weekday():
    idx = random.randint(0, 6)
    d = next_weekday(REF_DATE, idx)
    return f"next {WEEKDAYS[idx]}", d.isoformat()


def gen_date_this_weekday():
    idx = random.randint(0, 6)
    d = next_weekday(REF_DATE, idx)
    # "this <weekday>" only really distinct if within 6 days
    return f"this {WEEKDAYS[idx]}", d.isoformat()


def gen_date_in_n_days():
    n = random.randint(2, 20)
    # Phrased without a leading preposition so it composes cleanly with
    # templates that already supply "on"/"for" (avoids "on in 11 days").
    return f"{n} days from now", (REF_DATE + timedelta(days=n)).isoformat()


def gen_date_month_day():
    m_idx = random.randint(0, 11)
    day = random.randint(1, 28)
    year = REF_DATE.year
    candidate = date(year, m_idx + 1, day)
    if candidate < REF_DATE:
        year += 1
        candidate = date(year, m_idx + 1, day)
    suffix = "th"
    if day in (1, 21, 31):
        suffix = "st"
    elif day in (2, 22):
        suffix = "nd"
    elif day in (3, 23):
        suffix = "rd"
    return f"{MONTHS[m_idx].capitalize()} {day}{suffix}", candidate.isoformat()


def gen_date_numeric():
    m_idx = random.randint(0, 11)
    day = random.randint(1, 28)
    year = REF_DATE.year
    candidate = date(year, m_idx + 1, day)
    if candidate < REF_DATE:
        year += 1
        candidate = date(year, m_idx + 1, day)
    return f"{m_idx+1}/{day}/{year}", candidate.isoformat()


DATE_GENERATORS = [
    gen_date_today, gen_date_tomorrow, gen_date_day_after_tomorrow,
    gen_date_next_weekday, gen_date_this_weekday, gen_date_in_n_days,
    gen_date_month_day, gen_date_numeric,
]

# ---------------------------------------------------------------------------
# Time expression generators. Returns (natural_text, canonical_HH:MM or None)
# ---------------------------------------------------------------------------

def gen_time_oclock():
    h = random.randint(1, 12)
    ampm = random.choice(["am", "pm"])
    hour24 = h % 12 + (12 if ampm == "pm" else 0)
    return f"{h}{ampm}", f"{hour24:02d}:00"


def gen_time_hm():
    h = random.randint(1, 12)
    m = random.choice([15, 30, 45])
    ampm = random.choice(["am", "pm"])
    hour24 = h % 12 + (12 if ampm == "pm" else 0)
    return f"{h}:{m:02d}{ampm}", f"{hour24:02d}:{m:02d}"


def gen_time_24h():
    h = random.randint(0, 23)
    m = random.choice([0, 15, 30, 45])
    return f"{h:02d}:{m:02d}", f"{h:02d}:{m:02d}"


def gen_time_noon():
    return "noon", "12:00"


def gen_time_midnight():
    return "midnight", "00:00"


TIME_GENERATORS = [gen_time_oclock, gen_time_hm, gen_time_24h, gen_time_noon, gen_time_midnight]


# ---------------------------------------------------------------------------
# Templates per intent.
#
# NOTE: template counts are NOT equal across intents on purpose (see
# NOTES_FOR_COORDINATORS.md). CANCEL has noticeably fewer distinct phrasings
# than the other three intents.
#
# Slots used in templates:
#   {person} {event} {date} {time}
# Each slot, when filled, becomes a BIO-tagged span: B-PERSON/I-PERSON,
# B-EVENT/I-EVENT, B-DATE/I-DATE, B-TIME/I-TIME. Everything else is O.
# ---------------------------------------------------------------------------

CREATE_EVENT_TEMPLATES = [
    "schedule a {event} with {person} on {date} at {time}",
    "set up a {event} about the project for {date} {time}",
    "create an event called {event} on {date}",
    "book a {event} with {person} for {date} at {time}",
    "can you schedule a {event} on {date}",
    "please set up a {event} with {person} at {time} on {date}",
    "add a {event} to my calendar for {date} at {time}",
    "arrange a {event} with {person} next week",
    "i need a {event} on {date}",
    "put a {event} on my calendar at {time} {date}",
    "organize a {event} with {person}",
    "schedule the {event} for {date}",
]

SET_REMINDER_TEMPLATES = [
    "remind me to {event} on {date} at {time}",
    "set a reminder for {date} {time} about my {event}",
    "please remind me about the {event} tomorrow",
    "can you remind me to prepare for the {event} on {date}",
    "set a reminder to call {person} at {time}",
    "remind me about {event} with {person} at {time} on {date}",
    "i want a reminder for my {event}",
    "notify me before the {event} on {date}",
    "ping me about the {event} at {time}",
]

QUERY_FREE_TIME_TEMPLATES = [
    "am i free on {date}",
    "what is my availability on {date} at {time}",
    "do i have free time on {date}",
    "am i busy on {date}",
    "check if i am available at {time} on {date}",
    "is {date} open on my calendar",
]

CANCEL_TEMPLATES = [
    "cancel my {event} on {date}",
    "cancel the {event} with {person}",
    "delete my {date} appointment",
    "please cancel the {event}",
]

INTENT_TEMPLATES = {
    "CREATE_EVENT": CREATE_EVENT_TEMPLATES,
    "SET_REMINDER": SET_REMINDER_TEMPLATES,
    "QUERY_FREE_TIME": QUERY_FREE_TIME_TEMPLATES,
    "CANCEL": CANCEL_TEMPLATES,
}

# Roughly how many filled instances to generate per template per intent.
# CANCEL gets fewer templates AND we do not compensate with extra instances
# per template -- low lexical diversity is intentional (see notes).
INSTANCES_PER_TEMPLATE = {
    "CREATE_EVENT": 160,
    "SET_REMINDER": 160,
    "QUERY_FREE_TIME": 160,
    "CANCEL": 90,   # fewer templates AND fewer instances/template -> intentional class imbalance
}

# ---------------------------------------------------------------------------
# Hard cases for Task 2 (BiLSTM): lexically identical trigger word "may"
# used as a MONTH (should be tagged as part of a DATE span) vs as a MODAL
# VERB (should be tagged O). A unidirectional model relying only on left
# context genuinely struggles here without seeing what follows "may".
# ---------------------------------------------------------------------------

MAY_AMBIGUITY_EXAMPLES = [
    # (raw_text, intent, token_tag_pairs, canonical_date_or_None, canonical_time_or_None)
    ("schedule a meeting on may 5th at 3pm", "CREATE_EVENT",
     [("schedule", "O"), ("a", "O"), ("meeting", "O"), ("on", "O"),
      ("may", "B-DATE"), ("5th", "I-DATE"), ("at", "O"), ("3pm", "B-TIME")],
     "2027-05-05", "15:00"),
    ("i may come to the meeting tomorrow", "CREATE_EVENT",
     [("i", "O"), ("may", "O"), ("come", "O"), ("to", "O"), ("the", "O"),
      ("meeting", "O"), ("tomorrow", "B-DATE")],
     (REF_DATE + timedelta(days=1)).isoformat(), None),
    ("remind me that i may need to reschedule on may 9th", "SET_REMINDER",
     [("remind", "O"), ("me", "O"), ("that", "O"), ("i", "O"), ("may", "O"),
      ("need", "O"), ("to", "O"), ("reschedule", "O"), ("on", "O"),
      ("may", "B-DATE"), ("9th", "I-DATE")],
     "2027-05-09", None),
    ("we may need a call, set one up for may 12th at noon", "CREATE_EVENT",
     [("we", "O"), ("may", "O"), ("need", "O"), ("a", "O"), ("call,", "O"),
      ("set", "O"), ("one", "O"), ("up", "O"), ("for", "O"),
      ("may", "B-DATE"), ("12th", "I-DATE"), ("at", "O"), ("noon", "B-TIME")],
     "2027-05-12", "12:00"),
    ("am i free may 20th", "QUERY_FREE_TIME",
     [("am", "O"), ("i", "O"), ("free", "O"), ("may", "B-DATE"), ("20th", "I-DATE")],
     "2027-05-20", None),
    ("i may not be free that day, check may 3rd for me", "QUERY_FREE_TIME",
     [("i", "O"), ("may", "O"), ("not", "O"), ("be", "O"), ("free", "O"),
      ("that", "O"), ("day,", "O"), ("check", "O"), ("may", "B-DATE"),
      ("3rd", "I-DATE"), ("for", "O"), ("me", "O")],
     "2027-05-03", None),
]

# ---------------------------------------------------------------------------
# Light textual noise (typos / casual contractions) applied to a fraction of
# examples so the dataset is not perfectly clean -- this is realistic noise,
# not a trap. Applied AFTER tagging so tags still line up with tokens.
# ---------------------------------------------------------------------------

def maybe_add_noise(tokens):
    tokens = list(tokens)
    if random.random() < 0.12 and len(tokens) > 3:
        i = random.randint(0, len(tokens) - 1)
        w = tokens[i]
        if len(w) > 4 and w.isalpha():
            j = random.randint(1, len(w) - 2)
            tokens[i] = w[:j] + w[j+1] + w[j] + w[j+2:]  # swap two adjacent letters
    return tokens


def fill_template(template, intent):
    """
    Fills a template's slots, builds the raw text, per-token BIO tags,
    and canonical date/time strings.
    Returns dict with raw_text, tokens, tags, intent, date_iso, time_hm
    """
    slot_values = {}
    tag_spans = {}  # slot_name -> (text, tag_label)

    if "{person}" in template:
        name = random.choice(NAMES)
        slot_values["person"] = name
        tag_spans["person"] = (name, "PERSON")
    if "{event}" in template:
        ev = random.choice(EVENTS)
        slot_values["event"] = ev
        tag_spans["event"] = (ev, "EVENT")

    date_iso = None
    if "{date}" in template:
        gen = random.choice(DATE_GENERATORS)
        text, iso = gen()
        slot_values["date"] = text
        tag_spans["date"] = (text, "DATE")
        date_iso = iso

    time_hm = None
    if "{time}" in template:
        gen = random.choice(TIME_GENERATORS)
        text, hm = gen()
        slot_values["time"] = text
        tag_spans["time"] = (text, "TIME")
        time_hm = hm

    raw_text = template.format(**slot_values)
    tokens = raw_text.split(" ")

    # Build BIO tags by locating each slot's token span in order.
    tags = ["O"] * len(tokens)
    for slot_name, (phrase, label) in tag_spans.items():
        phrase_tokens = phrase.split(" ")
        # find first matching subsequence not yet tagged
        for start in range(len(tokens) - len(phrase_tokens) + 1):
            window = tokens[start:start + len(phrase_tokens)]
            if window == phrase_tokens and all(tags[start + k] == "O" for k in range(len(phrase_tokens))):
                tags[start] = f"B-{label}"
                for k in range(1, len(phrase_tokens)):
                    tags[start + k] = f"I-{label}"
                break

    noisy_tokens = maybe_add_noise(tokens)

    return {
        "raw_text": " ".join(noisy_tokens),
        "tokens": noisy_tokens,
        "tags": tags,
        "intent": intent,
        "date_iso": date_iso,
        "time_hm": time_hm,
    }


def build_target_string(example):
    intent = example["intent"]
    tag_str = " ".join(example["tags"])
    if example["date_iso"] and example["time_hm"]:
        canon = f"{example['date_iso']} {example['time_hm']}"
    elif example["date_iso"]:
        canon = example["date_iso"]
    elif example["time_hm"]:
        canon = example["time_hm"]
    else:
        canon = "NA"
    return f"{intent}|{tag_str}|{canon}"


def generate_all_examples():
    examples = []
    eid = 0
    for intent, templates in INTENT_TEMPLATES.items():
        n_per = INSTANCES_PER_TEMPLATE[intent]
        for template in templates:
            for _ in range(n_per):
                ex = fill_template(template, intent)
                ex["id"] = f"ex_{eid:06d}"
                ex["target_string"] = build_target_string(ex)
                ex["template"] = template  # kept for inspection; drop before training!
                examples.append(ex)
                eid += 1

    # Fold in the hand-written "may" ambiguity hard cases, oversampled a bit
    # so they show up meaningfully in each split rather than as one-offs.
    for _ in range(25):
        for raw_text, intent, token_tags, date_iso, time_hm in MAY_AMBIGUITY_EXAMPLES:
            tokens = [t for t, _ in token_tags]
            tags = [tg for _, tg in token_tags]
            ex = {
                "id": f"ex_{eid:06d}",
                "raw_text": raw_text,
                "tokens": tokens,
                "tags": tags,
                "intent": intent,
                "date_iso": date_iso,
                "time_hm": time_hm,
                "template": "__may_ambiguity_handwritten__",
            }
            ex["target_string"] = build_target_string(ex)
            examples.append(ex)
            eid += 1

    random.shuffle(examples)
    return examples


def naive_random_split(examples, train_frac=0.7, val_frac=0.15):
    """
    A plain random shuffle-and-slice split. This does NOT control for the
    fact that many examples share the same underlying `template` with only
    entity values swapped -- so the same template can (and will) appear in
    both train and test. See NOTES_FOR_COORDINATORS.md.
    """
    n = len(examples)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]
    return train, val, test


def strip_debug_fields(ex, keep_template=False):
    out = {k: v for k, v in ex.items()}
    if not keep_template:
        out.pop("template", None)
    return out


def main():
    examples = generate_all_examples()
    train, val, test = naive_random_split(examples)

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump([strip_debug_fields(e) for e in subset], f, indent=1)

    # combined CSV (with template column, for coordinators/instructors to
    # inspect data composition -- students' training code should NOT read
    # the template column)
    csv_path = os.path.join(OUT_DIR, "combined.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "raw_text", "tokens", "tags", "intent",
                          "date_iso", "time_hm", "target_string", "template", "split"])
        for split_name, subset in [("train", train), ("val", val), ("test", test)]:
            for e in subset:
                writer.writerow([
                    e["id"], e["raw_text"], " ".join(e["tokens"]), " ".join(e["tags"]),
                    e["intent"], e["date_iso"], e["time_hm"], e["target_string"],
                    e["template"], split_name,
                ])

    # basic stats
    def intent_counts(subset):
        d = {}
        for e in subset:
            d[e["intent"]] = d.get(e["intent"], 0) + 1
        return d

    template_overlap = len(
        {e["template"] for e in train} & {e["template"] for e in test}
    )

    stats = {
        "total_examples": len(examples),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "intent_counts_train": intent_counts(train),
        "intent_counts_val": intent_counts(val),
        "intent_counts_test": intent_counts(test),
        "unique_templates_total": len({e["template"] for e in examples}),
        "templates_shared_between_train_and_test": template_overlap,
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
