"""The decision layer: exclusivity, capping, thresholds."""

import numpy as np
import pytest

from ber.config import load_config
from ber.decide import apply_source_cap, contention_report, decide, resolve_exclusivity


@pytest.fixture
def cfg():
    return load_config()


def arrays(pairs):
    """pairs = [(s1, cand, prob)] -> (probs, s1_ids, cand_ids)."""
    return (
        np.asarray([p for _, _, p in pairs], dtype=np.float32),
        np.asarray([s for s, _, _ in pairs], dtype=object),
        np.asarray([c for _, c, _ in pairs], dtype=object),
    )


class TestExclusivity:
    def test_contested_record_goes_to_the_strongest_claim(self):
        """Each S2/S3 record matches at most one S1 entity -- verified in the data."""
        probs, s1, cand = arrays([("S1-a", "S2-x", 0.9), ("S1-b", "S2-x", 0.7)])
        kept = resolve_exclusivity(probs, s1, cand, probs >= 0.5)
        assert kept.tolist() == [True, False]

    def test_uncontested_claims_are_untouched(self):
        probs, s1, cand = arrays([("S1-a", "S2-x", 0.9), ("S1-b", "S2-y", 0.7)])
        kept = resolve_exclusivity(probs, s1, cand, probs >= 0.5)
        assert kept.all()

    def test_one_entity_may_keep_many_records(self):
        """The constraint is one-to-many, not one-to-one."""
        probs, s1, cand = arrays(
            [("S1-a", "S2-x", 0.9), ("S1-a", "S2-y", 0.8), ("S1-a", "S3-z", 0.7)]
        )
        kept = resolve_exclusivity(probs, s1, cand, probs >= 0.5)
        assert kept.sum() == 3

    def test_losing_entity_keeps_its_other_matches(self):
        probs, s1, cand = arrays(
            [("S1-a", "S2-x", 0.9), ("S1-b", "S2-x", 0.7), ("S1-b", "S2-y", 0.8)]
        )
        kept = resolve_exclusivity(probs, s1, cand, probs >= 0.5)
        assert kept.tolist() == [True, False, True]

    def test_below_threshold_pairs_never_win(self):
        probs, s1, cand = arrays([("S1-a", "S2-x", 0.9), ("S1-b", "S2-x", 0.2)])
        kept = resolve_exclusivity(probs, s1, cand, probs >= 0.5)
        assert kept.sum() == 1

    def test_empty_selection_is_handled(self):
        probs, s1, cand = arrays([("S1-a", "S2-x", 0.1)])
        assert resolve_exclusivity(probs, s1, cand, probs >= 0.5).sum() == 0


class TestSourceCap:
    def test_cap_keeps_the_highest_scoring_per_source(self):
        probs, s1, cand = arrays(
            [("S1-a", "S2-1", 0.9), ("S1-a", "S2-2", 0.8), ("S1-a", "S2-3", 0.7)]
        )
        kept = apply_source_cap(probs, s1, cand, probs >= 0.5, max_per_source=2)
        assert kept.tolist() == [True, True, False]

    def test_cap_is_per_source_not_per_entity(self):
        """S2 and S3 budgets are independent."""
        probs, s1, cand = arrays(
            [("S1-a", "S2-1", 0.9), ("S1-a", "S2-2", 0.8),
             ("S1-a", "S3-1", 0.7), ("S1-a", "S3-2", 0.6)]
        )
        kept = apply_source_cap(probs, s1, cand, probs >= 0.5, max_per_source=1)
        assert kept.tolist() == [True, False, True, False]


class TestDecide:
    def test_threshold_is_applied(self, cfg):
        probs, s1, cand = arrays([("S1-a", "S2-x", 0.9), ("S1-a", "S2-y", 0.3)])
        assert decide(probs, s1, cand, 0.5, cfg) == {"S1-a": {"S2-x"}}

    def test_entity_with_nothing_above_threshold_is_absent(self, cfg):
        """An absent entity becomes an empty prediction, which is a valid singleton."""
        probs, s1, cand = arrays([("S1-a", "S2-x", 0.1)])
        assert decide(probs, s1, cand, 0.5, cfg) == {}

    def test_exclusivity_can_be_switched_off(self, cfg):
        probs, s1, cand = arrays([("S1-a", "S2-x", 0.9), ("S1-b", "S2-x", 0.7)])
        cfg["decision"]["enforce_exclusivity"] = False
        result = decide(probs, s1, cand, 0.5, cfg)
        assert result == {"S1-a": {"S2-x"}, "S1-b": {"S2-x"}}


class TestContentionReport:
    def test_counts_contested_records(self):
        _probs, s1, cand = arrays(
            [("S1-a", "S2-x", 1.0), ("S1-b", "S2-x", 1.0), ("S1-c", "S2-y", 1.0)]
        )
        report = contention_report(s1, cand)
        assert report["distinct_candidates"] == 2
        assert report["contested_candidates"] == 1
        assert report["max_claims_on_one_record"] == 2
