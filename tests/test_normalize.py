"""Normalization rules, case by case, against patterns found in the real data."""

import pytest

from ber.normalize import (
    address_tokens,
    digit_tokens,
    embedding_text,
    fold_accents,
    is_non_latin,
    name_tokens,
    normalize_address,
    normalize_name,
    romanize,
    skeleton,
    squeeze_repeats,
)


class TestAccents:
    def test_latin_diacritics_are_folded(self):
        assert fold_accents("Dréxkor") == "Drexkor"
        assert fold_accents("ALLÉE DES HÊTRES") == "ALLEE DES HETRES"
        assert fold_accents("Àmicale") == "Amicale"

    def test_indic_combining_marks_survive(self):
        """A blanket NFD strip would destroy the nukta and virama."""
        for text in ("छत्तीसगढ़", "தமிழ்நாடு", "उत्तर प्रदेश"):
            assert fold_accents(text) == text


class TestLegalSuffixes:
    def test_abbreviations_canonicalize_to_the_same_form(self):
        assert normalize_name("Perfect Exports Pvt Ltd", "India") == normalize_name(
            "Perfect Exports Private Limited", "India"
        )

    def test_legal_form_is_preserved_not_deleted(self):
        """Pvt Ltd and LLP are different registrations; the tokens must survive."""
        assert "private" in name_tokens("Helio Tech Pvt.", "India")
        assert "llp" in name_tokens("Raj Investments LLP", "India")

    def test_distinct_legal_forms_stay_distinct(self):
        assert name_tokens("Acme Pvt Ltd", "India") != name_tokens("Acme LLP", "India")


class TestStates:
    @pytest.mark.parametrize(
        "value", ["Tamil Nadu", "TN", "தமிழ்நாடு", "tamilnadu"]
    )
    def test_indian_state_variants_collapse(self, value):
        assert normalize_address(value, "India") == "tn"

    @pytest.mark.parametrize("value", ["Uttar Pradesh", "उत्तर प्रदेश", "UP"])
    def test_devanagari_state_collapses(self, value):
        assert normalize_address(value, "India") == "up"

    def test_us_state_name_and_abbreviation_collapse(self):
        assert normalize_address("New York", "US") == normalize_address("NY", "US")

    def test_multiword_state_needs_phrase_matching(self):
        """`new york` must not survive as two unmapped tokens."""
        assert address_tokens("1 Main St, New York", "US")[-1] == "ny"


class TestAddressAbbreviations:
    def test_us_st_is_street(self):
        assert "street" in address_tokens("105 ELM ST, MORGANTON, NC", "US")

    def test_french_st_is_saint(self):
        """Country-specific: ST means Saint in France, Street in the US."""
        assert "saint" in address_tokens("12 ST MARTIN", "France")
        assert "street" not in address_tokens("12 ST MARTIN", "France")

    def test_french_abbreviations(self):
        tokens = address_tokens("63 R. DE DIEPPE, LILLE", "France")
        assert "rue" in tokens
        assert "avenue" in address_tokens("77 AV LEON JOUHAUX", "France")

    def test_road_variants_collapse(self):
        assert normalize_address("12 Main Rd", "US") == normalize_address("12 Main Road", "US")


class TestDigitTokens:
    def test_leading_zeros_are_folded(self):
        """AF-684 and AF-0684 are the same building."""
        assert digit_tokens("Af-684, Nandgram") == digit_tokens("AF-0684, NANDGRAM")

    def test_short_runs_are_dropped_by_default(self):
        assert digit_tokens("Floor 5, Main Road") == set()

    def test_pin_code_is_kept(self):
        assert "9487203" in digit_tokens("AF-0684, GHAZIABAD, 9487203")


class TestCrossScript:
    def test_non_latin_detection(self):
        assert is_non_latin("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்")
        assert is_non_latin("राम मार्केटिंग")
        assert not is_non_latin("Raj Investments LLP")
        assert not is_non_latin("Société Générale")

    def test_squeeze_collapses_transliteration_doubling(self):
        assert squeeze_repeats("innnvesttmenntts") == "investments"

    def test_skeleton_bridges_scripts(self):
        """The whole point: a Tamil name must land near its Latin original."""
        from rapidfuzz import fuzz

        latin = skeleton("Raj Investments LLP", "India")
        tamil = skeleton("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "India")
        assert fuzz.token_set_ratio(latin, tamil) > 80

    def test_skeleton_still_separates_different_businesses(self):
        from rapidfuzz import fuzz

        a = skeleton("Raj Investments LLP", "India")
        b = skeleton("Ram Marketing Private Limited", "India")
        assert fuzz.token_set_ratio(a, b) < 60

    def test_romanize_leaves_latin_alone(self):
        assert romanize("Back Alley Bistro") == "Back Alley Bistro"


class TestRealClusters:
    def test_raj_investments_addresses_all_collapse(self):
        """Four ground-truth-matched records, three scripts, one address."""
        variants = [
            "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu",
            "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu",
            "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN",
            "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு",
        ]
        normalized = {normalize_address(v, "India") for v in variants}
        assert len(normalized) == 1

    def test_helio_pair_differs_on_the_rare_token(self):
        a = set(name_tokens("Helio Tech Pvt.", "India"))
        b = set(name_tokens("Helio Constructions Pvt.", "India"))
        assert a & b == {"helio", "private"}
        assert a ^ b == {"tech", "constructions"}


class TestEmbeddingText:
    def test_includes_romanization_for_non_latin_names(self):
        text = embedding_text("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்", "Chennai, TN", "India")
        assert "romanized:" in text

    def test_omits_romanization_for_latin_names(self):
        assert "romanized:" not in embedding_text("Acme Inc", "1 Main St", "US")

    def test_missing_address_is_explicit(self):
        """3.3% of S2/S3 records have no address; it must not look like an empty string."""
        assert "(unknown)" in embedding_text("Acme Inc", "", "India")

    def test_country_is_never_hardcoded(self):
        """France is absent from training; it must flow through untouched."""
        text = embedding_text("ZNB Club SARL", "5 bis Rue Pierre Dignac", "France")
        assert "country: France" in text
