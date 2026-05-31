import json
import os
import re
from collections import defaultdict


SYSTEM_PROMPT = (
    "You are a schema linking model. Given a natural language question and a "
    "database schema, return only valid JSON mapping referenced table names to "
    "lists of referenced column names. Use only identifiers from the schema. "
    "If a table is referenced without specific columns, include it with an "
    "empty list."
)


def schema_path(db_id, schemas_dir="./schemas"):
    fname = db_id.replace(" ", "_").replace("/", "_") + ".json"
    return os.path.join(schemas_dir, fname)


def load_spider_schema(db_id, schemas_dir="./schemas"):
    with open(schema_path(db_id, schemas_dir)) as f:
        raw = json.load(f)
    return raw


def schema_tables_columns(raw_schema):
    tables = raw_schema["table_names_original"]
    columns = {table: [] for table in tables}
    column_types = raw_schema.get("column_types", [])
    for idx, (table_idx, column_name) in enumerate(raw_schema["column_names_original"]):
        if table_idx == -1:
            continue
        col_type = column_types[idx] if idx < len(column_types) else "text"
        columns[tables[table_idx]].append((column_name, col_type or "text"))
    return columns


def schema_key_maps(raw_schema):
    tables = raw_schema["table_names_original"]
    columns = raw_schema["column_names_original"]
    primary_keys = defaultdict(list)
    foreign_keys = defaultdict(list)

    for col_idx in raw_schema.get("primary_keys", []):
        table_idx, column_name = columns[col_idx]
        if table_idx >= 0:
            primary_keys[tables[table_idx]].append(column_name)

    for left_idx, right_idx in raw_schema.get("foreign_keys", []):
        left_table_idx, left_col = columns[left_idx]
        right_table_idx, right_col = columns[right_idx]
        if left_table_idx < 0 or right_table_idx < 0:
            continue
        left_table = tables[left_table_idx]
        right_table = tables[right_table_idx]
        foreign_keys[left_table].append((left_col, right_table, right_col))
        foreign_keys[right_table].append((right_col, left_table, left_col))

    return dict(primary_keys), dict(foreign_keys)


def identifier_tokens(text):
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    spaced = re.sub(r"[_/\-#]+", " ", spaced)
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", spaced)
        if len(token) > 1
    }


def select_candidate_tables(question, db_id, schemas_dir="./schemas", max_tables=30):
    raw_schema = load_spider_schema(db_id, schemas_dir)
    columns_by_table = schema_tables_columns(raw_schema)
    question_tokens = identifier_tokens(question)
    question_lc = question.lower()
    scored = []
    for table in raw_schema["table_names_original"]:
        score = 0
        table_tokens = identifier_tokens(table)
        score += 3 * len(question_tokens & table_tokens)
        if table.lower() in question_lc:
            score += 8
        for column, _col_type in columns_by_table[table]:
            column_tokens = identifier_tokens(column)
            overlap = question_tokens & column_tokens
            if overlap:
                score += 1 + len(overlap)
            if column.lower() in question_lc:
                score += 3
        scored.append((score, table))
    positives = [item for item in scored if item[0] > 0]
    ranked = sorted(positives, key=lambda item: (item[0], item[1]), reverse=True)
    if len(ranked) < max_tables:
        seen = {table for _score, table in ranked}
        ranked.extend(
            item
            for item in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
            if item[1] not in seen
        )
    return [table for _score, table in ranked[:max_tables]]


def serialize_schema(
    db_id,
    schemas_dir="./schemas",
    question=None,
    mode="full",
    max_tables=30,
    forced_tables=None,
):
    raw_schema = load_spider_schema(db_id, schemas_dir)
    columns_by_table = schema_tables_columns(raw_schema)
    primary_keys, foreign_keys = schema_key_maps(raw_schema)
    lines = [f"Database: {db_id}", "Schema:"]
    if mode in {"candidate", "candidate_fk"}:
        if question is None:
            raise ValueError("question is required when mode uses candidate tables")
        table_order = select_candidate_tables(question, db_id, schemas_dir, max_tables=max_tables)
        for table in forced_tables or []:
            if table in columns_by_table and table not in table_order:
                table_order.append(table)
        lines.append(f"Candidate tables shown: {len(table_order)}")
    else:
        table_order = raw_schema["table_names_original"]
    for table in table_order:
        cols = ", ".join(name for name, _col_type in columns_by_table[table])
        if mode == "candidate_fk":
            lines.append(f"Table: {table}")
            lines.append(f"Columns: {cols}")
            if primary_keys.get(table):
                lines.append(f"Primary keys: {', '.join(primary_keys[table])}")
            local_fks = foreign_keys.get(table, [])
            if local_fks:
                fk_text = "; ".join(
                    f"{left_col} -> {right_table}.{right_col}"
                    for left_col, right_table, right_col in local_fks[:12]
                )
                lines.append(f"Foreign keys: {fk_text}")
        else:
            lines.append(f"{table}: {cols}")
    return "\n".join(lines)


def normalize_links(links):
    normalized = {}
    if not isinstance(links, dict):
        return normalized
    for table, cols in links.items():
        if cols is None:
            continue
        if isinstance(cols, list):
            normalized[str(table)] = sorted({str(col) for col in cols})
        else:
            normalized[str(table)] = []
    return dict(sorted(normalized.items()))


def links_to_json(links):
    return json.dumps(normalize_links(links), ensure_ascii=False, sort_keys=True)


def make_messages(
    row,
    schemas_dir="./schemas",
    include_completion=True,
    schema_mode="full",
    max_candidate_tables=30,
):
    forced_tables = row.get("schema_links", {}).keys() if include_completion else None
    schema_text = serialize_schema(
        row["db_id"],
        schemas_dir,
        row["question"],
        schema_mode,
        max_candidate_tables,
        forced_tables=forced_tables,
    )
    user_prompt = (
        f"Question: {row['question']}\n\n"
        f"{schema_text}\n\n"
        "Return schema_links JSON only."
    )
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    out = {"prompt": prompt}
    if include_completion:
        out["completion"] = [
            {"role": "assistant", "content": links_to_json(row["schema_links"])}
        ]
    return out


def extract_first_json_object(text):
    if not text:
        return {}
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def schema_case_maps(db_id, schemas_dir="./schemas"):
    raw_schema = load_spider_schema(db_id, schemas_dir)
    tables = raw_schema["table_names_original"]
    table_map = {table.lower(): table for table in tables}
    col_maps = defaultdict(dict)
    for table_idx, column_name in raw_schema["column_names_original"]:
        if table_idx == -1:
            continue
        table = tables[table_idx]
        col_maps[table][column_name.lower()] = column_name
    return table_map, dict(col_maps)


def sanitize_links(links, db_id, schemas_dir="./schemas"):
    table_map, col_maps = schema_case_maps(db_id, schemas_dir)
    out = {}
    if not isinstance(links, dict):
        return out
    for table, cols in links.items():
        canonical_table = table_map.get(str(table).lower())
        if canonical_table is None:
            continue
        canonical_cols = []
        if isinstance(cols, list):
            col_map = col_maps.get(canonical_table, {})
            for col in cols:
                canonical_col = col_map.get(str(col).lower())
                if canonical_col is not None:
                    canonical_cols.append(canonical_col)
        out[canonical_table] = sorted(set(canonical_cols))
    return dict(sorted(out.items()))
