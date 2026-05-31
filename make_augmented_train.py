import argparse
import json
import os
import random

from schema_linking_utils import load_spider_schema, schema_tables_columns


def words(identifier):
    return (
        str(identifier)
        .replace("_", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("#", " number ")
    )


def schema_db_ids(index_path):
    with open(index_path) as f:
        return [row["db_id"] for row in json.load(f)]


def table_columns(db_id, schemas_dir):
    raw = load_spider_schema(db_id, schemas_dir)
    columns = schema_tables_columns(raw)
    return raw, {table: [name for name, _typ in cols] for table, cols in columns.items()}


def choose_columns(cols, rng, min_cols=1, max_cols=4):
    if not cols:
        return []
    k = min(len(cols), rng.randint(min_cols, max_cols))
    return sorted(rng.sample(cols, k))


def add_example(out, qid, db_id, question, links, source):
    out.append(
        {
            "question_id": qid,
            "db_id": db_id,
            "question": question,
            "gold_sql": f"-- synthetic schema-link example: {source}",
            "schema_links": links,
            "augmentation_source": source,
        }
    )
    return qid + 1


def fk_examples(raw, columns_by_table, db_id, qid, rng, out, limit):
    tables = raw["table_names_original"]
    col_entries = raw["column_names_original"]
    made = 0
    for left_idx, right_idx in raw.get("foreign_keys", []):
        if made >= limit:
            break
        left_table_idx, left_col = col_entries[left_idx]
        right_table_idx, right_col = col_entries[right_idx]
        if left_table_idx < 0 or right_table_idx < 0 or left_table_idx == right_table_idx:
            continue
        left_table = tables[left_table_idx]
        right_table = tables[right_table_idx]
        left_extra = choose_columns(
            [c for c in columns_by_table[left_table] if c != left_col],
            rng,
            min_cols=1,
            max_cols=2,
        )
        right_extra = choose_columns(
            [c for c in columns_by_table[right_table] if c != right_col],
            rng,
            min_cols=1,
            max_cols=2,
        )
        left_cols = sorted(set([left_col] + left_extra))
        right_cols = sorted(set([right_col] + right_extra))
        question = (
            f"Using {words(left_table)} and {words(right_table)}, show "
            f"{', '.join(words(c) for c in left_extra + right_extra) or 'matching rows'} "
            f"matched by {words(left_col)} and {words(right_col)}."
        )
        qid = add_example(
            out,
            qid,
            db_id,
            question,
            {left_table: left_cols, right_table: right_cols},
            "synthetic_fk_pair",
        )
        made += 1
    return qid


def build_synthetic_examples(train_rows, schemas_dir, per_db, seed):
    rng = random.Random(seed)
    qid = max(row["question_id"] for row in train_rows) + 1
    out = []
    for db_id in schema_db_ids(f"{schemas_dir}/_index.json"):
        raw, columns_by_table = table_columns(db_id, schemas_dir)
        tables = list(columns_by_table)
        rng.shuffle(tables)

        table_only_count = max(2, per_db // 10)
        for table in tables[:table_only_count]:
            qid = add_example(
                out,
                qid,
                db_id,
                f"How many records are in the {words(table)} table?",
                {table: []},
                "synthetic_table_only",
            )

        remaining = per_db - table_only_count
        single_count = max(0, remaining - 3)
        for table in tables[table_only_count : table_only_count + single_count]:
            cols = choose_columns(columns_by_table[table], rng)
            if not cols:
                continue
            phrasing = rng.choice(
                [
                    f"Show {', '.join(words(c) for c in cols)} from {words(table)}.",
                    f"What are the {', '.join(words(c) for c in cols)} values in {words(table)}?",
                    f"List {', '.join(words(c) for c in cols)} for records in the {words(table)} table.",
                ]
            )
            qid = add_example(
                out,
                qid,
                db_id,
                phrasing,
                {table: cols},
                "synthetic_single_table_columns",
            )

        qid = fk_examples(raw, columns_by_table, db_id, qid, rng, out, limit=3)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.json")
    parser.add_argument("--schemas_dir", default="schemas")
    parser.add_argument("--output", default="artifacts/data/train_augmented.json")
    parser.add_argument("--per_db", type=int, default=20)
    parser.add_argument("--seed", type=int, default=234)
    args = parser.parse_args()

    with open(args.train) as f:
        train_rows = json.load(f)
    synthetic = build_synthetic_examples(train_rows, args.schemas_dir, args.per_db, args.seed)
    augmented = train_rows + synthetic
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(augmented, f, indent=2)
    print(
        f"Wrote {len(augmented)} rows to {args.output} "
        f"({len(train_rows)} original + {len(synthetic)} synthetic)"
    )


if __name__ == "__main__":
    main()
