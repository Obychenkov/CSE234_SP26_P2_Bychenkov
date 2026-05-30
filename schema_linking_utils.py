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


def serialize_schema(db_id, schemas_dir="./schemas"):
    raw_schema = load_spider_schema(db_id, schemas_dir)
    columns_by_table = schema_tables_columns(raw_schema)
    lines = [f"Database: {db_id}", "Schema:"]
    for table in raw_schema["table_names_original"]:
        cols = ", ".join(name for name, _col_type in columns_by_table[table])
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


def make_messages(row, schemas_dir="./schemas", include_completion=True):
    user_prompt = (
        f"Question: {row['question']}\n\n"
        f"{serialize_schema(row['db_id'], schemas_dir)}\n\n"
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
