import json

import pytest

from src.services.load_config import load_config


def test_parses_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"model": "xgboost", "XGBoost_best_params": {"max_depth": 3}}))
    cfg = load_config(str(p))
    assert cfg["model"] == "xgboost"
    assert cfg["XGBoost_best_params"] == {"max_depth": 3}


def test_returns_full_dict(tmp_path):
    data = {"model": "log_reg", "LogReg_best_params": {}}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    assert load_config(str(p)) == data


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.json"))


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{ not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_config(str(p))
