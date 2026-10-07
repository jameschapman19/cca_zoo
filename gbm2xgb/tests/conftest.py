import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    # CatBoost writes catboost_info/ into the working directory
    monkeypatch.chdir(tmp_path)
