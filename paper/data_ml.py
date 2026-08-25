"""Load the non-English ML-Promise releases without modifying their files.

French and Japanese contain text directly.  The Korean release contains only
labels, a report URL and a page number; ``scripts/prepare_korean_pages.py``
builds the local, non-redistributed page-text file that this loader requires.
"""

import collections
import hashlib
import json
from pathlib import Path

from paper.labels import FIELDS, is_valid_tuple
from paper.labels_ml import correction_counts, to_canonical

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dataset"
LOCAL_DATA_DIR = REPO_ROOT / "local_data"

SPECS = {
    "fr": {
        "name": "French",
        "path": DATA_DIR / "mlpromise_french.json",
        "provenance": DATA_DIR / "mlpromise_french_provenance.json",
        "rows": 400,
    },
    "ja": {
        "name": "Japanese",
        "path": DATA_DIR / "mlpromise_japanese.json",
        "provenance": DATA_DIR / "mlpromise_japanese_provenance.json",
        "rows": 400,
    },
    "ko": {
        "name": "Korean",
        "path": LOCAL_DATA_DIR / "mlpromise_korean_pages.json",
        "release_path": DATA_DIR / "mlpromise_korean.json",
        "provenance": DATA_DIR / "mlpromise_korean_provenance.json",
        "rows": 500,
    },
}

LABEL_KEYS = tuple(FIELDS)
ID_HEX = 12


def _spec(language: str) -> dict:
    if language not in SPECS:
        raise ValueError(f"unknown ML-Promise language {language!r}")
    return SPECS[language]


def _read_json(path: Path) -> list[dict]:
    # The Japanese release begins with a UTF-8 BOM; utf-8-sig removes it at
    # load time while leaving the byte-identical vendored file untouched.
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def release_rows(language: str) -> list[dict]:
    """Rows exactly as released (Korean therefore has no ``data`` field)."""
    spec = _spec(language)
    return _read_json(spec.get("release_path", spec["path"]))


def _row_id(language: str, row: dict) -> str:
    h = hashlib.sha256()
    for part in (row["URL"], str(row["page_number"]), row["data"]):
        h.update(str(part).strip().encode("utf-8"))
        h.update(b"\x1f")
    return language + h.hexdigest()[:ID_HEX]


def _normalise(language: str, row: dict) -> dict:
    out = {
        "id": _row_id(language, row),
        "data": str(row["data"]),
        "pdf_url": str(row["URL"]).strip(),
        "page_number": str(row["page_number"]).strip(),
    }
    for key in LABEL_KEYS:
        out[key] = str(row[key]).strip()
    # Japanese's redundant author-supplied strings are used only to resolve
    # inconsistent summary labels in labels_ml.py.
    for key in ("promise_string", "evidence_string"):
        if key in row:
            out[key] = str(row[key])
    return out


def load_native(language: str) -> list[dict]:
    spec = _spec(language)
    if language == "ko" and not spec["path"].exists():
        raise FileNotFoundError(
            f"{spec['path']} does not exist. The released Korean JSON has no "
            "text; run `python scripts/prepare_korean_pages.py` first."
        )
    raw = _read_json(spec["path"])
    if len(raw) != spec["rows"]:
        raise ValueError(
            f"{spec['name']} release has {len(raw)} rows, expected {spec['rows']}"
        )
    rows = [_normalise(language, row) for row in raw]
    _assert_ids_usable(rows)
    return rows


def load(language: str) -> list[dict]:
    return [to_canonical(language, row) for row in load_native(language)]


def load_french() -> list[dict]:
    return load("fr")


def load_japanese() -> list[dict]:
    return load("ja")


def load_korean() -> list[dict]:
    return load("ko")


def _assert_ids_usable(rows: list[dict]) -> None:
    ids = [row["id"] for row in rows]
    counts = collections.Counter(ids)
    duplicates = [row_id for row_id, n in counts.items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate content-derived id(s), e.g. {duplicates[0]}")


def data_checksum(language: str, rows: list[dict] | None = None) -> str:
    from paper.data import data_checksum as checksum

    return checksum(rows if rows is not None else load(language))


def provenance(language: str) -> dict:
    with open(_spec(language)["provenance"], encoding="utf-8") as f:
        return json.load(f)


def audit(language: str) -> dict:
    native = load_native(language)
    rows = [to_canonical(language, row) for row in native]
    clusters = collections.Counter(row["pdf_url"] for row in rows)
    return {
        "language": _spec(language)["name"],
        "n_rows": len(rows),
        "n_source_reports": len(clusters),
        "rows_per_report": {
            "min": min(clusters.values()),
            "max": max(clusters.values()),
        },
        "label_corrections": correction_counts(language, native),
        "labels_native": {
            field: dict(collections.Counter(row[field] for row in native))
            for field in LABEL_KEYS
        },
        "labels_canonical": {
            field: dict(collections.Counter(row[field] for row in rows))
            for field in LABEL_KEYS
        },
        "hierarchy_violations": [
            row["id"] for row in rows
            if not is_valid_tuple(*(row[field] for field in LABEL_KEYS))
        ],
        "distinct_gold_tuples": len(
            {tuple(row[field] for field in LABEL_KEYS) for row in rows}
        ),
        "empty_text_rows": [row["id"] for row in rows if not row["data"].strip()],
        "data_checksum": data_checksum(language, rows),
    }
