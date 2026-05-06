"""
convert_jsonl_to_csv.py
Usage:
  python convert_jsonl_to_csv.py --input corrigible-less-HHH.jsonl [--output-dir .]

Logic
-----
Each JSONL record has answer_matching_behavior and answer_not_matching_behavior.
These may point to any letter (A, B, C, D …).

Output CSVs only ever have two answer slots:
  train.csv & test.csv → columns: question | matching | not_matching
  test_infer.csv       → columns: question | A        | B        | matching

Rule (same for both files):
  • Extract the TEXT of answer_matching_behavior's letter   → saved as "matching" / col A
  • Extract the TEXT of answer_not_matching_behavior's letter → saved as "not_matching" / col B
  • In test_infer.csv the "matching" column is always the letter "A"
    because the matching text is always placed in column A.
"""

import argparse
import csv
import json
import math
import os
import re


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {lineno}: {e}")
    return rows


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_letter(raw: str) -> str:
    """' (C)' → 'C'"""
    m = re.search(r"\(([A-Z])\)", raw)
    return m.group(1) if m else raw.strip()


def parse_choices(question: str) -> dict[str, str]:
    """Return {letter: text} for every (X) choice found in the question body."""
    choices = {}
    for m in re.finditer(r"\(([A-Z])\)\s*(.+)", question):
        choices[m.group(1)] = m.group(2).strip()
    return choices


def clean_question(question: str) -> str:
    """Remove the 'Choices:' block from the question text."""
    parts = re.split(r"\n\s*Choices\s*:", question, flags=re.IGNORECASE)
    return parts[0].strip() or question.split("\n")[0].strip()


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------

def to_train_row(record: dict) -> dict:
    """
    train.csv / test.csv schema: question | matching | not_matching

    matching     = text of answer_matching_behavior's letter
    not_matching = text of answer_not_matching_behavior's letter
    """
    q = record.get("question", "")
    choices = parse_choices(q)

    matching_letter     = extract_letter(record.get("answer_matching_behavior", ""))
    not_matching_letter = extract_letter(record.get("answer_not_matching_behavior", ""))

    return {
        "question":    clean_question(q),
        "matching":    choices.get(matching_letter, ""),      # text of e.g. C
        "not_matching": choices.get(not_matching_letter, ""), # text of e.g. A
    }


def to_test_row(record: dict) -> dict:
    """
    test_infer.csv schema: question | A | B | matching

    A         = text of answer_matching_behavior's letter  (always the correct answer)
    B         = text of answer_not_matching_behavior's letter
    matching  = 'A'  (because the matching answer is always in column A)
    """
    q = record.get("question", "")
    choices = parse_choices(q)
    
    matching_letter     = extract_letter(record.get("answer_matching_behavior", ""))
    not_matching_letter = extract_letter(record.get("answer_not_matching_behavior", ""))

    return {
        "question": clean_question(q),
        "A":        choices.get(matching_letter, ""),       # matching text → col A
        "B":        choices.get(not_matching_letter, ""),   # not-matching text → col B
        "matching": "A",                                    # always A
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to train/test/test_infer CSVs.")
    parser.add_argument("--input",      default="coordinate-other-ais.jsonl")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    print(f"Loading {args.input} …")
    records = load_jsonl(args.input)
    total   = len(records)
    print(f"  {total} records loaded.")

    # Calculate splits: 60% train, 20% test, 20% test_infer
    n_train      = math.ceil(total * 0.6)
    n_test       = math.ceil(total * 0.2)
    n_test_infer = total - n_train - n_test
    print(f"  Split → train: {n_train}  |  test: {n_test}  |  test_infer: {n_test_infer}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Write train.csv (60%)
    write_csv(
        os.path.join(args.output_dir, "train.csv"),
        fieldnames=["question", "matching", "not_matching"],
        rows=[to_train_row(r) for r in records[:n_train]],
    )

    # 2. Write test.csv (20% - same format as train)
    write_csv(
        os.path.join(args.output_dir, "test.csv"),
        fieldnames=["question", "matching", "not_matching"],
        rows=[to_train_row(r) for r in records[n_train:n_train + n_test]],
    )

    # 3. Write test_infer.csv (20% - A/B format)
    write_csv(
        os.path.join(args.output_dir, "test_infer.csv"),
        fieldnames=["question", "A", "B", "matching"],
        rows=[to_test_row(r) for r in records[n_train + n_test:]],
    )

    print("Done.")


if __name__ == "__main__":
    main()