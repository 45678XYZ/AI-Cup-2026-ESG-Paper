"""Load the ML-Promise English release into the shape the study already uses.

The vendored file is byte-identical to the release and is never rewritten;
everything below happens at load time, so the recorded sha256 keeps meaning
"what the authors published" rather than "what we decided it should say".

``load_english`` returns rows in the **frozen** label vocabulary, which is what
lets ``paper/splits.py``, ``paper/run_training.py`` and every decision rule
consume them without knowing a second language exists; the translation and its
justification live in ``paper/labels_en.py``. ``load_english_native`` returns
the release's own spelling, and only the audit uses it.

Three things the release does not supply and this module does.

**Row ids.** ML-Promise ships no identifier, and interface contract 1.1 keys
every artifact by an ``id`` string rather than by position. Deriving it from
the row's position in the file would make the ids depend on an ordering the
authors never promised to keep, so the id is a hash of the row's own content
(source URL, page, paragraph text). A re-release that reorders the file leaves
every id where it was; one that edits a paragraph changes that row's id and
nothing else, which is the behaviour a provenance key should have.

**Whitespace.** One row spells the timeline ``"2 to 5 years "``. Left alone it
becomes a sixth class of a five-class field, present in gold, predicted by
nothing, and scored: macro-F1 averages over classes, so a single stray space
would take a fixed slice off the field's score under every method equally and
look like a property of the task. This is the ``Wistron``/``wistron`` problem
from ``analysis/audit.py`` in another field, and it gets the same treatment --
strip, and a test that fails if the release stops needing it.

**Field naming.** ``URL`` becomes ``pdf_url``, which is what
``analysis.load.pdf_clusters`` resamples on.

Nine source reports, not forty-nine. That is the number that matters for every
interval computed on this data and it is why the English arm is pre-registered
as descriptive; see ``docs/preregistration/pre_registration_english_replication.md``.
"""

import collections
import hashlib
import json
from pathlib import Path

from paper.labels import FIELDS, is_valid_tuple
from paper.labels_en import to_canonical, unmapped_canonical_values

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dataset"

ENGLISH_PATH = DATA_DIR / "mlpromise_english.json"
PROVENANCE_PATH = DATA_DIR / "mlpromise_english_provenance.json"

LABEL_KEYS = tuple(FIELDS)

# Long enough that 400 rows collide with probability ~1e-17, short enough to
# read in a diff. Prefixed so an English id is never mistaken for an AI CUP one
# ("10001"), which is the kind of mix-up that produces a silent empty join.
ID_PREFIX = "en"
ID_HEX = 12


def _row_id(row: dict) -> str:
    h = hashlib.sha256()
    for part in (row["URL"], str(row["page_number"]), row["data"]):
        h.update(str(part).strip().encode("utf-8"))
        h.update(b"\x1f")
    return ID_PREFIX + h.hexdigest()[:ID_HEX]


def normalise(row: dict) -> dict:
    """One release row in the study's row shape, still in English spelling.

    Only whitespace is removed here. The vocabulary change happens in
    ``paper.labels_en.to_canonical`` and nowhere else: a quiet re-spelling in
    the loader is how two modules come to disagree about what a class is.
    """
    out = {
        "id": _row_id(row),
        "data": row["data"],
        "pdf_url": str(row["URL"]).strip(),
        "page_number": str(row["page_number"]).strip(),
    }
    for key in LABEL_KEYS:
        out[key] = str(row[key]).strip()
    return out


def load_english_native() -> list[dict]:
    """The 400 labelled rows in the release's own vocabulary."""
    with open(ENGLISH_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    rows = [normalise(r) for r in raw]
    _assert_ids_usable(rows)
    return rows


def load_english() -> list[dict]:
    """The 400 labelled rows in the frozen label vocabulary.

    This is what the pipeline consumes. Every row is legal under the frozen
    17-state space -- asserted in ``tests/test_data_en.py``, not assumed.
    """
    return [to_canonical(r) for r in load_english_native()]


def canonical_row_order(rows: list[dict] | None = None) -> list[str]:
    return [r["id"] for r in (rows if rows is not None else load_english())]


def index_by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def data_checksum(rows: list[dict] | None = None) -> str:
    """The study's own checksum, over the English rows.

    ``paper.data.data_checksum`` reads ``id``, ``data`` and the four label keys
    and nothing else, so it applies unchanged. Calling it here rather than
    reimplementing it keeps one definition of what a data checksum is, and the
    value differs from the Chinese one for the only reason that matters: the
    rows differ.
    """
    from paper.data import data_checksum as checksum

    return checksum(rows if rows is not None else load_english())


def provenance() -> dict:
    with open(PROVENANCE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _assert_ids_usable(rows: list[dict]) -> None:
    ids = [r["id"] for r in rows]
    non_str = [i for i in ids if not isinstance(i, str)]
    if non_str:
        raise TypeError(f"row ids must be str, got e.g. {non_str[0]!r}")
    if len(set(ids)) != len(ids):
        counts = collections.Counter(ids)
        dupes = [i for i, n in counts.items() if n > 1]
        raise ValueError(
            f"{len(dupes)} duplicate content-derived id(s), e.g. {dupes[0]}. "
            "Two rows carry the same URL, page and paragraph text; the release "
            "has changed and the id rule needs another field before use."
        )


def audit() -> dict:
    """The counts the pre-registration quotes, computed rather than transcribed.

    ``hierarchy_violations`` is the load-bearing one. The published description
    of ML-Promise lists no ``N/A`` value and states no dependency between the
    fields, so whether the English annotation carries the hierarchy at all is
    an empirical question about the release, not something the paper settles.
    Reported on the canonical rows, because that is the space every downstream
    artifact is scored in.
    """
    native = load_english_native()
    rows = [to_canonical(r) for r in native]
    clusters = collections.Counter(r["pdf_url"] for r in rows)
    return {
        "n_rows": len(rows),
        "n_source_reports": len(clusters),
        "rows_per_report": {"min": min(clusters.values()),
                            "max": max(clusters.values())},
        "labels_native": {
            f: dict(collections.Counter(r[f] for r in native)) for f in LABEL_KEYS
        },
        "labels_canonical": {
            f: dict(collections.Counter(r[f] for r in rows)) for f in LABEL_KEYS
        },
        "unreachable_canonical_labels": unmapped_canonical_values(),
        "hierarchy_violations": [
            r["id"] for r in rows
            if not is_valid_tuple(*(r[f] for f in LABEL_KEYS))
        ],
        "distinct_gold_tuples": len(
            {tuple(r[f] for f in LABEL_KEYS) for r in rows}
        ),
        "data_checksum": data_checksum(rows),
    }
