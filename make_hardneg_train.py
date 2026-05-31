import argparse
import json
import os
import random

from schema_linking_utils import identifier_tokens, load_spider_schema, schema_tables_columns


WEAK_DB_BONUS = {
    "NTSB": 3,
    "SBODemoUS-Reports": 4,
    "SBODemoUS-Finance": 4,
    "SBODemoUS-Business Partners": 4,
    "SBODemoUS-Sales Opportunities": 4,
    "SBODemoUS-Banking": 4,
    "SBODemoUS-General": 4,
}


def words(identifier):
    return (
        str(identifier)
        .replace("_", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("#", " number ")
    )


def table_signature(table, columns):
    tokens = set(identifier_tokens(table))
    for column, _typ in columns:
        tokens.update(identifier_tokens(column))
    return tokens


def similar_tables(db_id, gold_tables, schemas_dir, limit=4):
    raw = load_spider_schema(db_id, schemas_dir)
    columns_by_table = schema_tables_columns(raw)
    gold_tokens = set()
    for table in gold_tables:
        gold_tokens.update(table_signature(table, columns_by_table.get(table, [])))

    scored = []
    for table, columns in columns_by_table.items():
        if table in gold_tables:
            continue
        tokens = table_signature(table, columns)
        overlap = len(gold_tokens & tokens)
        if overlap:
            scored.append((overlap, table))
    return [table for _score, table in sorted(scored, reverse=True)[:limit]]


def explicit_question(row, confusing_tables):
    gold_parts = []
    for table, cols in row["schema_links"].items():
        if cols:
            gold_parts.append(f"{words(table)} columns {', '.join(words(c) for c in cols[:4])}")
        else:
            gold_parts.append(f"the {words(table)} table")
    avoid = ""
    if confusing_tables:
        avoid = f" Do not use nearby tables such as {', '.join(words(t) for t in confusing_tables[:3])}."
    return f"For this schema, link only {', '.join(gold_parts)} for: {row['question']}.{avoid}"


def row_level_question(row, confusing_tables):
    gold_tables = list(row["schema_links"])
    if not gold_tables:
        return None
    table = gold_tables[0]
    cols = row["schema_links"][table]
    if not cols:
        return None
    nearby = f" rather than {words(confusing_tables[0])}" if confusing_tables else ""
    return (
        f"Use the exact row/detail table {words(table)}{nearby}. "
        f"Return the schema links for {', '.join(words(c) for c in cols[:4])}."
    )


def build_hardneg_examples(base_rows, schemas_dir, seed):
    rng = random.Random(seed)
    qid = max(row["question_id"] for row in base_rows) + 1
    out = []

    for row in base_rows:
        gold_tables = list(row["schema_links"])
        if not gold_tables:
            continue
        confusing = similar_tables(row["db_id"], set(gold_tables), schemas_dir)
        copies = WEAK_DB_BONUS.get(row["db_id"], 1)
        if not confusing and copies == 1:
            continue

        variants = [explicit_question(row, confusing)]
        row_level = row_level_question(row, confusing)
        if row_level:
            variants.append(row_level)
        rng.shuffle(variants)

        for question in variants[:copies]:
            out.append(
                {
                    "question_id": qid,
                    "db_id": row["db_id"],
                    "question": question,
                    "gold_sql": "-- synthetic hard-negative schema-link example",
                    "schema_links": row["schema_links"],
                    "augmentation_source": "synthetic_hard_negative",
                    "confusing_tables": confusing,
                }
            )
            qid += 1

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="artifacts/data/train_augmented.json")
    parser.add_argument("--schemas_dir", default="schemas")
    parser.add_argument("--output", default="artifacts/data/train_augmented_hardneg.json")
    parser.add_argument("--seed", type=int, default=234)
    args = parser.parse_args()

    with open(args.base) as f:
        base_rows = json.load(f)

    hardneg = build_hardneg_examples(base_rows, args.schemas_dir, args.seed)
    augmented = base_rows + hardneg
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(augmented, f, indent=2)

    print(
        f"Wrote {len(augmented)} rows to {args.output} "
        f"({len(base_rows)} base + {len(hardneg)} hard-negative synthetic)"
    )


if __name__ == "__main__":
    main()
