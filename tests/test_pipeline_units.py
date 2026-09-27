"""Unit tests for the pieces between blocking and the models."""

import numpy as np
import pytest

from ber.blocking.candidates import fuse
from ber.blocking.lexical import LexicalIndex, name_ngrams, record_keys
from ber.dataset import entity_split
from ber.embed.backends import apply_mrl
from ber.embed.instruction import format_document, format_for_source, format_query
from ber.features import FEATURE_NAMES, FeatureExtractor, PairContext, build_idf
from ber.io import Record


def R(eid, name, addr, country="India"):
    return Record(eid, name, addr, country)


class TestMRL:
    def test_truncates_and_renormalizes(self):
        """Truncation alone leaves the vector off the unit sphere."""
        raw = np.random.default_rng(0).normal(size=(8, 1024)).astype(np.float32)
        out = apply_mrl(raw, 512)
        assert out.shape == (8, 512)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_no_truncation_when_already_small(self):
        raw = np.ones((2, 256), dtype=np.float32)
        assert apply_mrl(raw, 512).shape == (2, 256)


class TestInstruction:
    def test_s1_is_a_query_and_others_are_documents(self):
        """S1 is the side we retrieve *from*, matching the model's training."""
        assert format_for_source("x", "S1", "do thing").startswith("Instruct:")
        assert format_for_source("x", "S2", "do thing") == "x"
        assert format_document("x") == "x"

    def test_query_embeds_the_instruction_text(self):
        assert "weight industry" in format_query("x", "weight industry")


class TestFuse:
    def test_pair_found_by_both_channels_ranks_first(self):
        lex_ids = np.asarray(["A", "B", "C"], dtype=object)
        lex_scores = np.asarray([0.9, 0.7, 0.5], dtype=np.float32)
        dense_ids = np.asarray(["C", "D"], dtype=object)
        dense_scores = np.asarray([0.95, 0.8], dtype=np.float32)
        rows = fuse(lex_ids, lex_scores, dense_ids, dense_scores, k=10)
        assert rows[0][0] == "C"

    def test_missing_channel_records_rank_minus_one(self):
        rows = fuse(
            np.asarray([], dtype=object), np.asarray([], dtype=np.float32),
            np.asarray(["D"], dtype=object), np.asarray([0.8], dtype=np.float32), k=5,
        )
        assert rows[0][2] == -1      # lex_rank
        assert rows[0][4] == 0       # dense_rank

    def test_cap_is_respected(self):
        lex_ids = np.asarray(list("ABCDEFG"), dtype=object)
        scores = np.linspace(0.9, 0.1, 7).astype(np.float32)
        empty_i = np.asarray([], dtype=object)
        empty_f = np.asarray([], dtype=np.float32)
        assert len(fuse(lex_ids, scores, empty_i, empty_f, k=3)) == 3

    def test_blank_ids_are_skipped(self):
        """FAISS pads short result rows, which arrive as empty strings."""
        rows = fuse(
            np.asarray(["", "A"], dtype=object), np.asarray([0.0, 0.5], dtype=np.float32),
            np.asarray([], dtype=object), np.asarray([], dtype=np.float32), k=5,
        )
        assert [r[0] for r in rows] == ["A"]


class TestLexicalKeys:
    def test_every_channel_is_emitted_for_a_non_latin_name(self):
        keys = record_keys(R("S2-1", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்", "6(29) Chennai TN, 600004"))
        prefixes = {k[0] for k in keys}
        assert {"a", "n", "d", "s", "g"} <= prefixes

    def test_domain_names_share_trigrams_with_the_spaced_form(self):
        """`backalleybistro.com` must reach `Back Alley Bistro`."""
        a = name_ngrams("Back Alley Bistro", "US")
        b = name_ngrams("backalleybistro.com", "US")
        assert len(a & b) / len(a | b) > 0.5

    def test_index_retrieves_an_exact_duplicate(self):
        records = [
            R(f"S2-{i}", f"Company {i} Private Limited", f"{100 + i} Main Road Chennai TN")
            for i in range(200)
        ]
        index = LexicalIndex.build(records, max_df_ratio=0.5)
        query = R("S1-1", "Company 7 Private Limited", "107 Main Road Chennai TN")
        indices, scores = index.query([query], top_n=5)
        assert index.ids[indices[0, 0]] == "S2-7"
        assert scores[0, 0] > 0


class TestFeatures:
    @pytest.fixture
    def extractor(self):
        corpus = [
            R(f"S2-{i}", n, a)
            for i, (n, a) in enumerate(
                [
                    ("Helio Tech Private Limited", "12 Main Road Chennai TN"),
                    ("Helio Constructions Private Limited", "99 Park Road Chennai TN"),
                    ("Raj Investments LLP", "6 29 CIT Colony Chennai TN"),
                    ("Sharma Traders Private Limited", "45 Gandhi Road Chennai TN"),
                ]
                * 30
            )
        ]
        return FeatureExtractor(build_idf(corpus, "name"), build_idf(corpus, "address"))

    @pytest.fixture
    def context(self):
        import collections

        return PairContext(50, 0.9, 0.9, 0.03, collections.Counter())

    def test_vector_width_matches_the_name_list(self, extractor, context):
        vec = extractor.pair(
            R("S1-1", "Acme Inc", "1 Main Road"), R("S2-1", "Acme Inc", "1 Main Road"),
            0.9, 0, 0.8, 0, 0.03, 0, context,
        )
        assert vec.shape == (len(FEATURE_NAMES),)
        assert np.isfinite(vec).all()

    def test_identical_records_score_high_similarity(self, extractor, context):
        idx = FEATURE_NAMES.index("name_jaccard")
        vec = extractor.pair(
            R("S1-1", "Raj Investments LLP", "6 29 CIT Colony"),
            R("S2-1", "Raj Investments LLP", "6 29 CIT Colony"),
            0.9, 0, 0.8, 0, 0.03, 0, context,
        )
        assert vec[idx] == pytest.approx(1.0)

    def test_empty_address_is_flagged_not_silently_zero(self, extractor, context):
        idx = FEATURE_NAMES.index("addr_missing_one")
        vec = extractor.pair(
            R("S1-1", "Bright Book Store", "3004 Leeann Boulevard"),
            R("S3-1", "Bright Bk Store", ""),
            0.9, 0, 0.0, -1, 0.016, 0, context,
        )
        assert vec[idx] == 1.0

    def test_cross_script_pair_scores_on_skeleton_not_surface(self, extractor, context):
        surface = FEATURE_NAMES.index("name_jaccard")
        skel = FEATURE_NAMES.index("skeleton_token_set_ratio")
        vec = extractor.pair(
            R("S1-1", "Raj Investments LLP", "6 29 CIT Colony Chennai TN"),
            R("S2-1", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "6 29 CIT COLONY CHENNAI TN"),
            0.9, 0, 0.8, 0, 0.03, 0, context,
        )
        assert vec[surface] == 0.0
        assert vec[skel] > 0.8

    def test_source_flags_are_mutually_exclusive(self, extractor, context):
        s2 = FEATURE_NAMES.index("cand_is_s2")
        s3 = FEATURE_NAMES.index("cand_is_s3")
        vec = extractor.pair(
            R("S1-1", "Acme", "1 Main"), R("S3-9", "Acme", "1 Main"),
            0.9, 0, 0.8, 0, 0.03, 0, context,
        )
        assert vec[s2] == 0.0 and vec[s3] == 1.0


class TestEntitySplit:
    def test_splits_are_disjoint_and_complete(self):
        entities = [f"S1-{i}" for i in range(1000)]
        train, val, test = entity_split(entities, 0.15, 0.15, seed=42)
        assert not (train & val) and not (train & test) and not (val & test)
        assert len(train | val | test) == 1000

    def test_proportions_are_respected(self):
        train, val, test = entity_split([f"S1-{i}" for i in range(1000)], 0.2, 0.1, 42)
        assert len(val) == 200
        assert len(test) == 100

    def test_split_is_deterministic_for_a_seed(self):
        a = entity_split([f"S1-{i}" for i in range(100)], 0.2, 0.2, 7)
        b = entity_split([f"S1-{i}" for i in range(100)], 0.2, 0.2, 7)
        assert a == b
