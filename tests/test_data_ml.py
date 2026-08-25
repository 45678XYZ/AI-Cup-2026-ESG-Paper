"""Integrity and load-time normalisation of the additional ML-Promise data."""

import json

import pytest

from paper.data import file_sha256
from paper.data_ml import REPO_ROOT, SPECS, audit, load, provenance, release_rows
from paper.labels import FIELDS, is_valid_tuple

EXPECTED = {
    "fr": {
        "rows": 400,
        "reports": 9,
        "sha256": "sha256:50b8473f36c1fe433e2dd30cb69da23eb57ebdaf865511fb973396c784963ac2",
        "corrections": {"verification_timeline": 319, "evidence_status": 81},
    },
    "ja": {
        "rows": 400,
        "reports": 19,
        "sha256": "sha256:bc4b6e6a7fae9d3696da3df85c2ab5fc0b478b080487f43da0d4bbc2b8cf6e65",
        "corrections": {
            "verification_timeline": 36,
            "evidence_status": 4,
            "evidence_quality": 2,
            "promise_status": 1,
        },
    },
    "ko": {
        "rows": 500,
        "reports": 32,
        "sha256": "sha256:ced009a34cad9f77af1cd015ad1b82a85dbdb9c1c2407ede5d6bf578dabb287a",
        "corrections": {"evidence_status": 121, "evidence_quality": 27},
    },
}


@pytest.mark.parametrize("language", sorted(EXPECTED))
def test_vendored_release_matches_drive_provenance(language):
    spec = SPECS[language]
    path = spec.get("release_path", spec["path"])
    expected = EXPECTED[language]
    assert file_sha256(path) == expected["sha256"]
    assert len(release_rows(language)) == expected["rows"]
    prov = provenance(language)
    assert prov["sha256"] == expected["sha256"]
    assert prov["source_drive_file_id"]
    assert prov["licence"] == "CC BY-NC-SA 4.0"


def test_the_japanese_release_bom_is_preserved():
    assert SPECS["ja"]["path"].read_bytes().startswith(b"\xef\xbb\xbf")


def test_the_korean_release_really_has_no_text_and_one_unique_page_per_row():
    rows = release_rows("ko")
    assert all("data" not in row for row in rows)
    pages = {(row["URL"], str(row["page_number"])) for row in rows}
    assert len(pages) == len(rows) == 500


@pytest.mark.parametrize("language", ["fr", "ja"])
def test_direct_text_releases_load_into_seventeen_legal_states(language):
    result = audit(language)
    assert result["n_rows"] == EXPECTED[language]["rows"]
    assert result["n_source_reports"] == EXPECTED[language]["reports"]
    assert result["label_corrections"] == EXPECTED[language]["corrections"]
    assert result["hierarchy_violations"] == []
    assert result["empty_text_rows"] == []
    assert all(is_valid_tuple(*(row[field] for field in FIELDS))
               for row in load(language))


@pytest.mark.skipif(not SPECS["ko"]["path"].exists(),
                    reason="local Korean report-page extraction not prepared")
def test_prepared_korean_pages_are_complete_and_legal():
    result = audit("ko")
    assert result["n_rows"] == 500
    assert result["n_source_reports"] == 32
    assert result["label_corrections"] == EXPECTED["ko"]["corrections"]
    assert result["hierarchy_violations"] == []
    assert result["empty_text_rows"] == []
    manifest = json.loads(
        (REPO_ROOT / "local_data/mlpromise_korean_pages_manifest.json").read_text()
    )
    assert manifest["n_rows"] == 500
    assert manifest["n_reports"] == 32
    assert manifest["empty_text_row_indexes"] == []
    assert len(manifest["ocr_row_indexes"]) == 14


@pytest.mark.parametrize("language", sorted(EXPECTED))
def test_attribution_notice_is_present(language):
    notice = REPO_ROOT / "dataset" / f"mlpromise_{SPECS[language]['name'].lower()}.NOTICE"
    text = notice.read_text(encoding="utf-8")
    prov = provenance(language)
    for required in (prov["licence"], prov["source_drive_folder"], "Seki", "EMNLP 2025"):
        assert required in text
