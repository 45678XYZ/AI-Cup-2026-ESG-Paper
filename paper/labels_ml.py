"""Map the French, Japanese and Korean ML-Promise labels into the frozen space.

The downloaded JSON files are never rewritten.  Every correction below is
therefore visible, counted by :func:`correction_counts`, and covered by tests.
French uses the same spelling as the English release.  Japanese mostly uses
the frozen spelling already, but the release contains three systematic
``2``/``already`` substitutions, one bracket typo, one missing promise label,
and four rows whose summary labels disagree with the supplied promise/evidence
strings.  Korean already uses the canonical timeline spelling; like English,
it writes ``No`` rather than ``N/A`` for evidence when no promise exists.
"""

from collections import Counter

from paper.labels import EVAL_FIELDS, FIELDS
from paper.labels_en import to_canonical as english_to_canonical


JAPANESE_TIMELINE_FIXES = {
    "within_already_years": "within_2_years",
    "between_already_and_5_years": "between_2_and_5_years",
    "betwenn_already_and_5_years": "between_2_and_5_years",
}


def _assert_canonical(row: dict, language: str) -> dict:
    for field in FIELDS:
        value = row[field]
        allowed = EVAL_FIELDS[field]
        if value not in allowed:
            raise ValueError(
                f"{language} {field}={value!r} is not canonical; expected one "
                f"of {allowed}"
            )
    return row


def french_to_canonical(row: dict) -> dict:
    """French and English use exactly the same native label vocabulary."""
    return _assert_canonical(english_to_canonical(row), "French")


def japanese_to_canonical(row: dict) -> dict:
    """Correct the release's auditable spelling and cross-field inconsistencies.

    ``promise_string`` and ``evidence_string`` are author-supplied redundant
    fields.  They make the four inconsistent summary rows recoverable without
    interpreting Japanese ourselves: a non-empty extracted evidence string
    means evidence is present, while an empty one means it is absent.  The two
    ``N/A`` timelines attached to positive promises are changed to ``already``
    (the release's category for an already realised or non-time-specific
    promise), rather than inventing a deadline.
    """
    out = dict(row)
    if out["promise_status"] == "":
        if not str(out.get("promise_string", "")).strip():
            raise ValueError("Japanese empty promise_status has no promise_string")
        out["promise_status"] = "Yes"

    timeline = out["verification_timeline"]
    out["verification_timeline"] = JAPANESE_TIMELINE_FIXES.get(timeline, timeline)
    if out["promise_status"] == "Yes" and out["verification_timeline"] == "N/A":
        out["verification_timeline"] = "already"

    evidence_text = str(out.get("evidence_string", "")).strip()
    if out["promise_status"] == "No":
        out["evidence_status"] = "N/A"
    else:
        out["evidence_status"] = "Yes" if evidence_text else "No"

    if out["evidence_quality"] == "Clear]":
        out["evidence_quality"] = "Clear"
    if out["evidence_status"] in {"No", "N/A"}:
        out["evidence_quality"] = "N/A"
    return _assert_canonical(out, "Japanese")


def korean_to_canonical(row: dict) -> dict:
    """Map Korean's no-promise evidence convention into the frozen vocabulary.

    The 27 Korean no-promise rows whose released evidence quality is Clear or
    Not Clear are left visible in ``correction_counts`` and become N/A here.
    A quality judgment cannot be retained after the release's own
    ``promise_status=No`` has made the evidence question inapplicable.
    """
    out = dict(row)
    if out["promise_status"] == "No":
        out["verification_timeline"] = "N/A"
        out["evidence_status"] = "N/A"
        out["evidence_quality"] = "N/A"
    return _assert_canonical(out, "Korean")


MAPPERS = {
    "fr": french_to_canonical,
    "ja": japanese_to_canonical,
    "ko": korean_to_canonical,
}


def to_canonical(language: str, row: dict) -> dict:
    try:
        mapper = MAPPERS[language]
    except KeyError as exc:
        raise ValueError(f"unsupported ML-Promise language {language!r}") from exc
    return mapper(row)


def correction_counts(language: str, native_rows: list[dict]) -> dict[str, int]:
    """Count label changes made by the loader, field by field."""
    counts = Counter()
    for native in native_rows:
        canonical = to_canonical(language, native)
        for field in FIELDS:
            if canonical[field] != native[field]:
                counts[field] += 1
    return dict(counts)
