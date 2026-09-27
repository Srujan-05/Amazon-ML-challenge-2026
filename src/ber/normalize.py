"""Text normalization for names and addresses.

Shared by blocking, feature extraction, and embedding-text construction, so a
pair is compared the same way everywhere.

Three properties of the data shape this module:

* **Names change script.** 12.2% of true matches have zero name-token overlap,
  mostly Devanagari/Tamil transliterations of a Latin name. Every string
  therefore also gets a romanized variant.
* **Addresses stay Latin even for Indian records**, but abbreviate heavily
  (``Rd``/``Road``, ``TN``/``Tamil Nadu``) and sometimes carry a transliterated
  state name (``தமிழ்நாடு``).
* **Digit tokens are the most selective key** -- 80.3% of true matches share one
  -- but they need leading-zero folding, since ``AF-684`` and ``AF-0684`` are the
  same building.

Abbreviations are canonicalized to a shared form rather than deleted: dropping
``Pvt``/``Ltd`` loses the signal that distinguishes a private limited company
from an LLP, and ``St`` means *Street* in the US but *Saint* in France, which is
why the public helpers take a ``country``.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from unidecode import unidecode

# --------------------------------------------------------------------------- #
# Script handling
# --------------------------------------------------------------------------- #

_SCRIPT_RANGES = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("arabic", 0x0600, 0x06FF),
    ("cjk", 0x4E00, 0x9FFF),
)


def detect_scripts(text: str) -> set[str]:
    """Return the set of scripts present, e.g. ``{"latin", "tamil"}``.

    Digits and punctuation are ignored, so a purely numeric string yields an
    empty set rather than a misleading ``latin``.
    """
    found: set[str] = set()
    for ch in text:
        if not ch.isalpha():
            continue
        code = ord(ch)
        if code < 0x0250:
            found.add("latin")
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= code <= hi:
                found.add(name)
                break
        else:
            found.add("other")
    return found


def is_non_latin(text: str) -> bool:
    """True when ``text`` contains alphabetic characters outside Latin."""
    return bool(detect_scripts(text) - {"latin"})


def fold_accents(text: str) -> str:
    """Strip diacritics from Latin characters only.

    A blanket NFD-and-drop-combining-marks would mangle Indic scripts, where the
    nukta and virama are meaningful. So a combining mark is only dropped when the
    base character it attaches to is Latin: ``Hêtres`` -> ``Hetres`` and
    ``Àmicale`` -> ``Amicale``, while ``ड़`` and ``க்`` survive intact.
    """
    decomposed = unicodedata.normalize("NFD", text)
    out: list[str] = []
    latin_base = False
    for ch in decomposed:
        if unicodedata.combining(ch):
            if not latin_base:
                out.append(ch)
            continue
        latin_base = ord(ch) < 0x0250
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def romanize(text: str) -> str:
    """Transliterate any script to Latin, offline.

    Used to compare a Devanagari or Tamil name against its Latin counterpart.
    ``unidecode`` is a local character-mapping table -- no network, no external
    data source, so this stays inside the challenge's fair-play rules.
    """
    if not is_non_latin(text):
        return fold_accents(text)
    return fold_accents(unidecode(text))


_REPEATS = re.compile(r"(.)\1+")


def squeeze_repeats(text: str) -> str:
    """Collapse runs of the same character: ``innnvesttmenntts`` -> ``investments``.

    Transliteration spells long vowels and geminated consonants out, so the Tamil
    form of "Investments" romanizes to ``innnvesttmenntts``. Squeezing recovers a
    comparable skeleton -- but only ever applied to *both* sides of a comparison,
    since it is equally destructive to genuine English doubles (``mill``,
    ``address``). Never use it on a form meant for display or embedding.
    """
    return _REPEATS.sub(r"\1", text)


# --------------------------------------------------------------------------- #
# Canonical token dictionaries
# --------------------------------------------------------------------------- #

# Legal forms are canonicalized, never removed: "Pvt Ltd" and "LLP" are
# different legal entities and that difference is discriminative.
_LEGAL = {
    "pvt": "private", "pvt.": "private", "prvt": "private", "pv": "private",
    "prb": "private", "pra": "private",
    "ltd": "limited", "ltd.": "limited", "lim": "limited", "limted": "limited",
    "ld": "limited",
    "corp": "corporation", "corp.": "corporation", "corporated": "corporation",
    "inc": "incorporated", "inc.": "incorporated",
    "co": "company", "co.": "company", "compny": "company",
    "llp": "llp", "l.l.p": "llp", "llp.": "llp",
    "llc": "llc", "l.l.c": "llc",
    "and": "&", "et": "&",
    "sarl": "sarl", "s.a.r.l": "sarl", "eurl": "sarl",
    "sasu": "sasu", "sas": "sas", "s.a.s": "sas",
    "sci": "sci", "s.c.i": "sci",
    "sa": "sa", "s.a": "sa",
    "gmbh": "gmbh", "bv": "bv", "nv": "nv", "plc": "plc",
}

_ADDRESS_COMMON = {
    "rd": "road", "rd.": "road", "roads": "road",
    "st": "street", "st.": "street", "str": "street", "stree": "street",
    "ave": "avenue", "av": "avenue", "ave.": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bld": "boulevard", "bd": "boulevard",
    "ln": "lane", "lane": "lane",
    "dr": "drive", "drv": "drive",
    "ct": "court", "crt": "court",
    "pl": "place", "plz": "plaza", "plaza": "plaza",
    "sq": "square",
    "hwy": "highway", "hway": "highway",
    "pkwy": "parkway", "pky": "parkway",
    "cir": "circle",
    "ter": "terrace",
    "apt": "apartment", "apts": "apartment",
    "bldg": "building", "bldgs": "building",
    "fl": "floor", "flr": "floor",
    "ste": "suite", "suite": "suite",
    "no": "number", "no.": "number", "num": "number", "nos": "number",
    "opp": "opposite", "opp.": "opposite",
    "nr": "near",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}

# "ST" is Street in the US and Saint in France: the canonical map is chosen by
# the record's country, which is reliable (0 disagreements across true matches).
_ADDRESS_FR = {
    **_ADDRESS_COMMON,
    "st": "saint", "st.": "saint", "ste": "sainte", "sts": "saints",
    "r": "rue", "r.": "rue", "rue": "rue",
    "av": "avenue", "bd": "boulevard", "bld": "boulevard",
    "imp": "impasse", "all": "allee", "allee": "allee",
    "ch": "chemin", "rte": "route", "qu": "quai",
    "fg": "faubourg", "pl": "place", "sq": "square",
    "bat": "batiment", "res": "residence", "zi": "zone",
    "cedex": "cedex",
}

_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}

# Indian states, including the transliterated forms that appear in S2/S3 --
# a matched record may carry "Tamil Nadu", "TN", or "தமிழ்நாடு".
_IN_STATES = {
    "andhra pradesh": "ap", "आंध्र प्रदेश": "ap",
    "arunachal pradesh": "ar",
    "assam": "as", "असम": "as",
    "bihar": "br", "बिहार": "br",
    "chhattisgarh": "ct", "छत्तीसगढ़": "ct",
    "delhi": "dl", "new delhi": "dl", "दिल्ली": "dl", "नई दिल्ली": "dl",
    "goa": "ga", "गोवा": "ga",
    "gujarat": "gj", "गुजरात": "gj",
    "haryana": "hr", "हरियाणा": "hr",
    "himachal pradesh": "hp", "हिमाचल प्रदेश": "hp",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk",
    "jharkhand": "jh", "झारखंड": "jh",
    "karnataka": "ka", "कर्नाटक": "ka",
    "kerala": "kl", "केरल": "kl",
    "madhya pradesh": "mp", "मध्य प्रदेश": "mp",
    "maharashtra": "mh", "महाराष्ट्र": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "ओडिशा": "od",
    "punjab": "pb", "पंजाब": "pb",
    "puducherry": "py", "pondicherry": "py",
    "rajasthan": "rj", "राजस्थान": "rj",
    "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "தமிழ்நாடு": "tn", "तमिलनाडु": "tn",
    "telangana": "tg", "तेलंगाना": "tg",
    "tripura": "tr",
    "uttar pradesh": "up", "उत्तर प्रदेश": "up",
    "uttarakhand": "uk", "उत्तराखंड": "uk",
    "west bengal": "wb", "पश्चिम बंगाल": "wb",
    "chandigarh": "ch",
}

_STATE_MAPS = {"US": _US_STATES, "India": _IN_STATES}
_ADDRESS_MAPS = {"France": _ADDRESS_FR}

_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-zÀ-ɏऀ-ൿ]+")
_LEADING_ZEROS = re.compile(r"^0+(?=\d)")


@lru_cache(maxsize=8)
def _phrase_pattern(country: str) -> tuple[re.Pattern[str], dict[str, str]] | None:
    """Compile a multi-word state matcher for ``country``, longest phrase first.

    Multi-word names have to be replaced before tokenization, otherwise
    ``new york`` splits into two tokens that can never map onto ``ny``.
    """
    states = _STATE_MAPS.get(country)
    if not states:
        return None
    phrases = sorted((p for p in states if " " in p), key=len, reverse=True)
    if not phrases:
        return None
    pattern = re.compile(r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b")
    return pattern, states


def _canonical_token(token: str, country: str, kind: str) -> str:
    """Map one token to its canonical form."""
    if kind == "name":
        return _LEGAL.get(token, token)
    address_map = _ADDRESS_MAPS.get(country, _ADDRESS_COMMON)
    if token in address_map:
        return address_map[token]
    states = _STATE_MAPS.get(country)
    if states and token in states:
        return states[token]
    if token.isdigit():
        return _LEADING_ZEROS.sub("", token)
    return token


def _prepare(text: str, country: str, kind: str) -> str:
    """NFKC, accent-fold, lowercase, then collapse multi-word state names."""
    prepared = fold_accents(unicodedata.normalize("NFKC", text or "")).lower()
    if kind == "address":
        compiled = _phrase_pattern(country)
        if compiled is not None:
            pattern, states = compiled
            prepared = pattern.sub(lambda m: states[m.group(1)], prepared)
    return prepared


def tokenize(text: str, country: str = "", kind: str = "address") -> list[str]:
    """Normalized token list: canonical abbreviations, folded digits, no punctuation."""
    prepared = _prepare(text, country, kind)
    out: list[str] = []
    for raw in _TOKEN_SPLIT.split(prepared):
        if not raw:
            continue
        canon = _canonical_token(raw, country, kind)
        if canon:
            out.append(canon)
    return out


def normalize_name(text: str, country: str = "") -> str:
    """Canonical name string, for embedding text and display."""
    return " ".join(tokenize(text, country, kind="name"))


def normalize_address(text: str, country: str = "") -> str:
    """Canonical address string, for embedding text and display."""
    return " ".join(tokenize(text, country, kind="address"))


def name_tokens(text: str, country: str = "") -> list[str]:
    return tokenize(text, country, kind="name")


def address_tokens(text: str, country: str = "") -> list[str]:
    return tokenize(text, country, kind="address")


def romanized_tokens(text: str, country: str = "", kind: str = "name") -> list[str]:
    """Tokens of the romanized form, so a Tamil name can be compared to a Latin one."""
    return tokenize(romanize(text), country, kind=kind)


def skeleton(text: str, country: str = "", kind: str = "name") -> str:
    """Romanized, repeat-squeezed comparison key.

    The form used to match across scripts. Applied symmetrically to both sides of
    a pair, it brings ``ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`` and ``Raj Investments`` within one
    edit of each other.
    """
    return squeeze_repeats(" ".join(romanized_tokens(text, country, kind=kind)))


def digit_tokens(text: str, min_digits: int = 3) -> set[str]:
    """Selective numeric keys: house numbers and PIN codes, leading zeros folded.

    ``AF-684`` and ``AF-0684`` both yield ``{"684"}``. Runs shorter than
    ``min_digits`` are dropped -- a bare floor number like ``5`` appears in far
    too many addresses to be a useful blocking key.
    """
    out: set[str] = set()
    for run in re.findall(r"\d+", text or ""):
        folded = _LEADING_ZEROS.sub("", run)
        if len(folded) >= min_digits:
            out.add(folded)
    return out


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-grams over a space-padded string, for typo-tolerant matching."""
    padded = f" {text} "
    if len(padded) <= n:
        return {padded}
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def acronym(tokens: list[str]) -> str:
    """Initials of the alphabetic tokens, for ``I B M`` vs ``IBM`` style matches."""
    return "".join(t[0] for t in tokens if t and t[0].isalpha())


def embedding_text(name: str, address: str, country: str) -> str:
    """The text handed to the embedder for one record.

    Fields are labelled so the instruction can reason about them separately, and
    a non-Latin name carries its romanization alongside the original -- giving
    the model both surface forms rather than betting on its transliteration.
    """
    norm_name = normalize_name(name, country)
    norm_addr = normalize_address(address, country)
    parts = [f"name: {norm_name or '(unknown)'}"]
    if is_non_latin(name):
        roman = " ".join(romanized_tokens(name, country, kind="name"))
        if roman and roman != norm_name:
            parts.append(f"romanized: {roman}")
    parts.append(f"address: {norm_addr or '(unknown)'}")
    parts.append(f"country: {country}")
    return " | ".join(parts)
