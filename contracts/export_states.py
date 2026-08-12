"""Export the canonical 17-state table as a language-neutral JSON artifact.

``paper/labels.py`` is the single source of truth. This script publishes a copy
for consumers that are not importing the Python package (a reviewer reading the
release, a non-Python analysis, the paper appendix).
``tests/test_labels.py::test_states_json_matches_its_source`` asserts the copy
still matches its source, so the duplicate cannot drift.

    python -m contracts.export_states
"""

import json
from dataclasses import asdict
from pathlib import Path

from paper.labels import EVAL_FIELDS, FIELD_WEIGHTS, STATES

OUT_PATH = Path(__file__).resolve().parent / "states.json"


def build() -> dict:
    return {
        "contract_version": "1.0",
        "source": "paper/labels.py",
        "eval_fields": EVAL_FIELDS,
        "field_weights": FIELD_WEIGHTS,
        "n_states": len(STATES),
        "states": [{"state_id": i, **asdict(s)} for i, s in enumerate(STATES)],
    }


def main() -> None:
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(build(), f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT_PATH} ({len(STATES)} states)")


if __name__ == "__main__":
    main()
