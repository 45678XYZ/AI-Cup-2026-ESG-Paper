"""The English replication's reporting path."""

import numpy as np
import pytest

from analysis.english_replication import _invalid_rate, _mean_std
from paper.labels import LABEL2ID


def test_mean_std_is_seed_sample_variation():
    mean, std = _mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert std == pytest.approx(1.0)


def test_invalid_rate_uses_the_frozen_17_state_space():
    legal = np.array([[LABEL2ID["promise_status"]["No"],
                       LABEL2ID["verification_timeline"]["N/A"],
                       LABEL2ID["evidence_status"]["N/A"],
                       LABEL2ID["evidence_quality"]["N/A"]]])
    illegal = legal.copy()
    illegal[0, 1] = LABEL2ID["verification_timeline"]["already"]
    assert _invalid_rate(legal) == 0.0
    assert _invalid_rate(np.concatenate([legal, illegal])) == 0.5
