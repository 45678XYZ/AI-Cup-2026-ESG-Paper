"""Dataset loading, canonical row order, and the data checksum.

Interface contract §1.1/§1.2: rows are identified by their ``id`` **string**,
never by position. ``canonical_row_order`` is the single authority for the row
order of every array that crosses a contract boundary.
"""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dataset"

TRAIN_PATH = DATA_DIR / "vpesg4k_train_1000.json"
VAL_PATH = DATA_DIR / "vpesg4k_val_1000.json"
TEST_PATH = DATA_DIR / "vpesg4k_test_2000.json"

LABEL_KEYS = ("promise_status", "verification_timeline", "evidence_status", "evidence_quality")


def load_dev() -> list[dict]:
    """The 2,000 labelled development rows, in canonical order (train then val).

    ``json.load`` is used rather than pandas because ``pd.read_json`` coerces
    the ``id`` strings ("10001") to integers, which silently breaks every
    id-keyed join downstream (contract §1.1).
    """
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train = json.load(f)
    with open(VAL_PATH, encoding="utf-8") as f:
        val = json.load(f)
    rows = train + val
    _assert_ids_usable(rows)
    return rows


def canonical_row_order(rows: list[dict] | None = None) -> list[str]:
    return [r["id"] for r in (rows if rows is not None else load_dev())]


def index_by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def data_checksum(rows: list[dict] | None = None) -> str:
    """sha256 over (id, data, four labels) in canonical order.

    Any two contract artifacts carrying different checksums are not
    interoperable, and mixing them silently would be worse than failing.
    """
    rows = rows if rows is not None else load_dev()
    h = hashlib.sha256()
    for r in rows:
        h.update(r["id"].encode("utf-8"))
        h.update(b"\x1f")
        h.update(r["data"].encode("utf-8"))
        for k in LABEL_KEYS:
            h.update(b"\x1f")
            h.update(r[k].encode("utf-8"))
        h.update(b"\x1e")
    return "sha256:" + h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _assert_ids_usable(rows: list[dict]) -> None:
    ids = [r["id"] for r in rows]
    non_str = [i for i in ids if not isinstance(i, str)]
    if non_str:
        raise TypeError(f"row ids must be str, got e.g. {non_str[0]!r} ({type(non_str[0])})")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row ids in development data")
