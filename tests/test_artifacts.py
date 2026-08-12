"""The contract-file writers in paper/artifacts.py.

Correctness here is enforced at write time rather than checked afterwards: the
bundle writer refuses malformed input, and the training driver feeds it rows
sliced straight from the split manifest, so a conforming bundle is the only
kind that can be produced.
"""

import copy
import json

import numpy as np
import pytest

from contracts.make_fixtures import make_rotation_fixture
from paper.artifacts import write_probs_bundle
from paper.data import load_dev
from paper.labels import EVAL_FIELDS
from paper.splits import build_split, split_fingerprint

ROWS = load_dev()
SPLIT = build_split("pdf_group", 42, ROWS)


@pytest.fixture
def bundle(tmp_path):
    return make_rotation_fixture(SPLIT, 0, tmp_path / "pdf_group_seed42_r0",
                                 concentration=0.6, rows=ROWS)


def _probs_for_rotation(rot):
    return {
        partition: {
            f: np.zeros((len(rot[f"{partition}_ids"]), len(EVAL_FIELDS[f])), np.float32)
            for f in EVAL_FIELDS
        }
        for partition in ("calibration", "test")
    }


def test_bundle_writer_rejects_a_wrong_row_count(tmp_path):
    """A misaligned array is the failure mode that would otherwise sail
    through and produce plausible-looking scores, so it must not reach disk."""
    probs = _probs_for_rotation(SPLIT["rotations"][0])
    probs["test"]["promise_status"] = probs["test"]["promise_status"][:-1]
    with pytest.raises(ValueError, match="expected"):
        write_probs_bundle(tmp_path / "b", SPLIT, 0, probs)


def test_bundle_writer_rejects_a_wrong_class_count(tmp_path):
    probs = _probs_for_rotation(SPLIT["rotations"][0])
    probs["calibration"]["evidence_quality"] = probs["calibration"]["evidence_quality"][:, :2]
    with pytest.raises(ValueError, match="expected"):
        write_probs_bundle(tmp_path / "b", SPLIT, 0, probs)


def test_bundle_row_order_and_shape_follow_the_split(bundle):
    meta = json.load(open(bundle / "meta.json", encoding="utf-8"))
    rot = SPLIT["rotations"][0]
    assert meta["calibration_ids"] == rot["calibration_ids"]
    assert meta["test_ids"] == rot["test_ids"]
    arr = np.load(bundle / "test_evidence_quality.npy")
    assert arr.shape == (len(rot["test_ids"]), len(EVAL_FIELDS["evidence_quality"]))
    assert arr.dtype == np.float32


def test_bundle_records_what_it_was_produced_against(bundle):
    meta = json.load(open(bundle / "meta.json", encoding="utf-8"))
    assert meta["data_checksum"] == SPLIT["data_checksum"]
    assert meta["split_fingerprint"] == SPLIT["split_fingerprint"]
    assert set(meta["artifacts"]) == {
        f"{p}_{f}.npy" for p in ("calibration", "test") for f in EVAL_FIELDS
    }


def test_fingerprint_covers_the_partition_not_the_metadata():
    """Identifies which partition a run belongs to, while surviving a
    regeneration that changes only timestamps."""
    cosmetic = copy.deepcopy(SPLIT)
    cosmetic["created_at"], cosmetic["git_sha"] = "2020-01-01T00:00:00+00:00", "deadbeef"
    assert split_fingerprint(cosmetic) == split_fingerprint(SPLIT)

    moved = copy.deepcopy(SPLIT)
    moved["rotations"][1]["test_ids"] = moved["rotations"][1]["test_ids"][::-1]
    assert split_fingerprint(moved) != split_fingerprint(SPLIT)
