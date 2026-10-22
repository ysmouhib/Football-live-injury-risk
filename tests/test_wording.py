from __future__ import annotations

from livrisk.app import wording


def test_no_forbidden_phrases_in_wording() -> None:
    for s in wording.all_strings():
        assert wording.find_forbidden(s) == [], s


def test_detector_catches_forbidden_phrases() -> None:
    assert wording.find_forbidden("He is SAFE to continue") == ["SAFE"]
    assert wording.find_forbidden("Player will be injured soon") == ["will be injured"]
    assert wording.find_forbidden("player welfare and safety") == []  # whole words only


def test_reason_texts_fit_the_column() -> None:
    for code, text in wording.REASON_TEXT.items():
        formatted = text.format(k="2.5", m="12")
        assert len(formatted) <= 36, (code, formatted)


def test_quality_reasons_are_short_and_cover_contract_codes() -> None:
    assert all(len(v) <= 40 for v in wording.QUALITY_REASON_TEXT.values())
    assert {"low_coverage", "identity_low", "norm_reset"} <= set(wording.QUALITY_REASON_TEXT)
