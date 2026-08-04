# ML Club Assignment: Calendar-Assistant NLU Pipeline

## Overview

You're building a 3-stage NLU pipeline for a calendar assistant. All three
stages run on the same input sentence, but are trained as **three separate
models**, each with a different required LSTM architecture. Each model's
output must be a **string**.

The dataset (`output/train.json`, `val.json`, `test.json`) gives you
`raw_text`, `tokens`, `tags`, `intent`, `date_iso`, `time_hm`, and a combined
`target_string` field per example. `combined.csv` has the same data plus a
`template` and `split` column for your own exploration — do not feed the
`template` column into any model as a feature.

---

## Task 1 — Intent Classification

**Input:** a raw text query, e.g. `"remind me to call sara tomorrow at 3pm"`

**Output:** a string — one of `CREATE_EVENT`, `SET_REMINDER`,
`QUERY_FREE_TIME`, `CANCEL`.

**Architecture constraint:** unidirectional vanilla LSTM, many-to-one.

---

## Task 2 — Slot Filling / Entity Tagging

**Input:** the same raw text query, tokenized.

**Output:** a string of space-separated BIO tags, one per input token,
e.g. `O O O B-EVENT O B-DATE O B-TIME`. Entity types: `DATE`, `TIME`,
`PERSON`, `EVENT`.

**Architecture constraint:** Bidirectional LSTM.

---

## Task 3 — String Normalization

**Input:** the raw text query (or the DATE/TIME portion you extract from
Task 2's output — your choice, state which one you use and why).

**Output:** a canonical string `YYYY-MM-DD HH:MM` (or just the date, or
just the time, depending on what's present in the query — see `date_iso`
/ `time_hm` in the data), or `NA` if no date/time is present.

**Architecture constraint:** plain LSTM Encoder–Decoder, **no attention**.
Encoder can be uni- or bidirectional; decoder must be unidirectional,
initialized from the encoder's final state, and must generate output
tokens until it produces `<EOS>`.

---

## Combined target format

Each example's `target_string` field packs all three tasks' labels into
one pipe-delimited string:

```
<INTENT>|<BIO_TAGS_SPACE_SEPARATED>|<CANONICAL_DATETIME_OR_NA>
```

Example:
```
SET_REMINDER|O O O O O B-PERSON O B-TIME|16:00
```

You're free to use the separate `intent` / `tags` / `date_iso` / `time_hm`
fields directly, or parse `target_string` apart yourselves — but you should
be able to produce this exact combined string as a final pipeline output,
regardless of which internal representation you train on.

---

## What to submit

For each of the three tasks, report the metrics you think are actually
appropriate for that task and that architecture — a single overall
accuracy number is probably not the whole story for all of these. Justify
your choice of metric(s) in a couple of sentences per task, and include
enough detail (e.g. a breakdown by class, or by example type) that someone
reading your report could tell whether the model is doing well
**everywhere**, not just on average.

Also include a short section per task on anything about the dataset itself
that affected how you evaluated your model, if applicable.
