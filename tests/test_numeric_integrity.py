from __future__ import annotations

import pytest

from data.numeric_integrity import DataIntegrityError, clean_text, numeric_spans, prepare_local_corpus


@pytest.mark.parametrize("value", ["2026", "-3.14159", "1,234.50", "12/31/2026", "9.81 m/s²", "1e-9", "50%"])
def test_numeric_spans_survive_cleaning(value):
    text = f"Измерение {value}\r\nточно"
    assert clean_text(text).splitlines() == [f"Измерение {value}", "точно"]
    assert numeric_spans(text) == numeric_spans(clean_text(text))


def test_unicode_normalization_cannot_rewrite_numeric_values():
    text = "value １２３"
    with pytest.raises(DataIntegrityError):
        clean_text(text, normalization="NFKC")


def test_legacy_source_corpus_entrypoint_is_unavailable_before_io(tmp_path):
    output = tmp_path / "must-not-be-created"
    with pytest.raises(DataIntegrityError, match="UNAVAILABLE: legacy source-snapshot"):
        prepare_local_corpus([tmp_path / "missing-source"], output)
    assert not output.exists()
