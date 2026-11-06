import copy
from pathlib import Path

import pytest

from gr2mini.m00_setup.config import apply_overrides, deep_merge, load_config

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- deep_merge
def test_deep_merge_overrides_scalars_and_keeps_other_keys():
    out = deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert out == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_is_recursive_for_nested_dicts():
    base = {"model": {"dim": 32, "levels": 4, "ema": {"decay": 0.99, "on": True}}}
    over = {"model": {"dim": 64, "ema": {"decay": 0.9}}}
    assert deep_merge(base, over) == {"model": {"dim": 64, "levels": 4, "ema": {"decay": 0.9, "on": True}}}


def test_deep_merge_replaces_lists_instead_of_concatenating():
    assert deep_merge({"dims": [256, 128]}, {"dims": [64]}) == {"dims": [64]}


def test_deep_merge_non_dict_overrides_dict_and_vice_versa():
    assert deep_merge({"a": {"x": 1}}, {"a": None}) == {"a": None}
    assert deep_merge({"a": 5}, {"a": {"x": 1}}) == {"a": {"x": 1}}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    over = {"a": {"x": 2}, "c": {"z": 1}}
    base_before, over_before = copy.deepcopy(base), copy.deepcopy(over)
    out = deep_merge(base, over)
    assert base == base_before and over == over_before
    # The result must not share nested objects with the inputs either.
    out["a"]["y"].append(99)
    out["c"]["z"] = 123
    assert base == base_before and over == over_before


# ---------------------------------------------------------------- load_config
def test_load_config_without_extends(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("seed: 1\nmodel:\n  dim: 8\n", encoding="utf-8")
    assert load_config(p) == {"seed": 1, "model": {"dim": 8}}


def test_load_config_empty_file_is_empty_dict(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == {}


def test_load_config_extends_is_relative_to_the_file_not_cwd(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("seed: 1\nmodel:\n  dim: 8\n  levels: 4\n", encoding="utf-8")
    (tmp_path / "configs" / "child.yaml").write_text("extends: base.yaml\nmodel:\n  dim: 16\n", encoding="utf-8")
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_config(tmp_path / "configs" / "child.yaml")
    assert cfg == {"seed": 1, "model": {"dim": 16, "levels": 4}}
    assert "extends" not in cfg


def test_load_config_multi_level_extends(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\ny: 1\nz: 1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\ny: 2\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("extends: b.yaml\nz: 3\n", encoding="utf-8")
    assert load_config(str(tmp_path / "c.yaml")) == {"x": 1, "y": 2, "z": 3}


# ---------------------------------------------------------------- apply_overrides
def test_apply_overrides_parses_yaml_values():
    cfg = {"rqvae": {"use_ema": True, "epochs": 300, "hidden_dims": [256, 128]}, "seed": 1}
    out = apply_overrides(cfg, ["rqvae.use_ema=false", "rqvae.epochs=10", "rqvae.hidden_dims=[64, 32]",
                                "seed=null", "rqvae.lr=1.0e-3", "name=abc"])
    assert out["rqvae"] == {"use_ema": False, "epochs": 10, "hidden_dims": [64, 32], "lr": 0.001}
    assert out["seed"] is None and out["name"] == "abc"
    assert cfg["rqvae"]["use_ema"] is True and cfg["seed"] == 1  # not mutated


def test_apply_overrides_creates_missing_dicts_and_splits_on_first_equals():
    out = apply_overrides({}, ["a.b.c=1", "url=http://x?y=1"])
    assert out == {"a": {"b": {"c": 1}}, "url": "http://x?y=1"}


def test_apply_overrides_none_and_errors():
    cfg = {"a": {"b": 1}}
    out = apply_overrides(cfg, None)
    assert out == cfg and out is not cfg
    assert apply_overrides(cfg, []) == cfg
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["a.b"])


def test_repo_configs_load_and_tiny_overrides_beauty():
    beauty = load_config(REPO_ROOT / "configs" / "beauty.yaml")
    tiny = load_config(REPO_ROOT / "configs" / "tiny.yaml")
    assert beauty["rqvae"]["codebook_size"] == 256
    assert tiny["rqvae"]["codebook_size"] == 64
    # inherited, not overridden:
    assert tiny["rqvae"]["use_ema"] == beauty["rqvae"]["use_ema"]
    assert tiny["data"]["reviews_url"] == beauty["data"]["reviews_url"]
    assert "extends" not in tiny
