import pytest

from src.services.load_config import load_config


def test_parses_key_value(tmp_path):
    p = tmp_path / "config.txt"
    p.write_text("model=xgboost\n")
    assert load_config(str(p)) == {"model": "xgboost"}


def test_strips_whitespace_and_ignores_non_kv_lines(tmp_path):
    p = tmp_path / "config.txt"
    p.write_text("  model = xgboost  \n\nno_equals_here\n")
    cfg = load_config(str(p))
    assert cfg == {"model": "xgboost"}  # trimmed; blank and non-kv lines ignored


def test_splits_on_first_equals_only(tmp_path):
    p = tmp_path / "config.txt"
    p.write_text("key=val=ue\n")
    assert load_config(str(p)) == {"key": "val=ue"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.txt"))
