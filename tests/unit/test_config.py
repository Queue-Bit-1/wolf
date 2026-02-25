"""Tests for wolf.config -- schema models and configuration loading."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from wolf.config.loader import load_config, merge_configs
from wolf.config.schema import (
    GameConfig,
    ModelConfig,
    PlayerConfig,
    RoleSlot,
)


# ======================================================================
# GameConfig defaults
# ======================================================================


class TestGameConfigDefaults:
    """Test that GameConfig has the expected default values."""

    def test_game_name(self) -> None:
        cfg = GameConfig()
        assert cfg.game_name == "classic_7p"

    def test_num_players(self) -> None:
        cfg = GameConfig()
        assert cfg.num_players == 7

    def test_roles_default(self) -> None:
        cfg = GameConfig()
        role_names = [r.role for r in cfg.roles]
        assert "werewolf" in role_names
        assert "seer" in role_names
        assert "doctor" in role_names
        assert "villager" in role_names

    def test_roles_counts(self) -> None:
        cfg = GameConfig()
        counts = {r.role: r.count for r in cfg.roles}
        assert counts["werewolf"] == 2
        assert counts["seer"] == 1
        assert counts["doctor"] == 1
        assert counts["villager"] == 3

    def test_max_days(self) -> None:
        cfg = GameConfig()
        assert cfg.max_days == 15

    def test_default_model(self) -> None:
        cfg = GameConfig()
        assert cfg.default_model is not None
        assert isinstance(cfg.default_model, ModelConfig)

    def test_empty_players_list(self) -> None:
        cfg = GameConfig()
        assert cfg.players == []


# ======================================================================
# ModelConfig defaults
# ======================================================================


class TestModelConfigDefaults:
    """Test ModelConfig default values."""

    def test_api_base(self) -> None:
        m = ModelConfig()
        assert m.api_base == "http://localhost:11434/v1"

    def test_api_key(self) -> None:
        m = ModelConfig()
        assert m.api_key == "ollama"

    def test_model(self) -> None:
        m = ModelConfig()
        assert m.model == "qwq:latest"

    def test_reasoning_temperature(self) -> None:
        m = ModelConfig()
        assert m.reasoning_temperature == 0.7

    def test_action_temperature(self) -> None:
        m = ModelConfig()
        assert m.action_temperature == 0.3

    def test_timeout(self) -> None:
        m = ModelConfig()
        assert m.timeout == 120.0

    def test_max_tokens(self) -> None:
        m = ModelConfig()
        assert m.max_tokens == 2048

    def test_extra_params_empty(self) -> None:
        m = ModelConfig()
        assert m.extra_params == {}


# ======================================================================
# RoleSlot and PlayerConfig
# ======================================================================


class TestRoleSlot:
    """Tests for RoleSlot model."""

    def test_role_and_count(self) -> None:
        rs = RoleSlot(role="werewolf", count=2)
        assert rs.role == "werewolf"
        assert rs.count == 2

    def test_default_count(self) -> None:
        rs = RoleSlot(role="seer")
        assert rs.count == 1


class TestPlayerConfig:
    """Tests for PlayerConfig model."""

    def test_required_name(self) -> None:
        pc = PlayerConfig(name="Alice")
        assert pc.name == "Alice"

    def test_default_agent_type(self) -> None:
        pc = PlayerConfig(name="Alice")
        assert pc.agent_type == "llm"

    def test_default_model_is_none(self) -> None:
        pc = PlayerConfig(name="Alice")
        assert pc.model is None

    def test_default_personality_is_none(self) -> None:
        pc = PlayerConfig(name="Alice")
        assert pc.personality is None


# ======================================================================
# load_config
# ======================================================================


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_none_returns_defaults(self) -> None:
        cfg = load_config(None)
        assert isinstance(cfg, GameConfig)
        assert cfg.game_name == "classic_7p"

    def test_load_config_nonexistent_path_returns_defaults(self) -> None:
        cfg = load_config("/does/not/exist/config.yaml")
        assert isinstance(cfg, GameConfig)
        assert cfg.game_name == "classic_7p"

    def test_load_config_valid_yaml(self) -> None:
        data = {
            "game_name": "test_game",
            "num_players": 5,
            "max_days": 10,
            "roles": [
                {"role": "werewolf", "count": 1},
                {"role": "villager", "count": 4},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            f.flush()
            path = f.name

        try:
            cfg = load_config(path)
            assert cfg.game_name == "test_game"
            assert cfg.num_players == 5
            assert cfg.max_days == 10
            assert len(cfg.roles) == 2
        finally:
            os.unlink(path)

    def test_load_config_invalid_yaml_returns_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(":::invalid yaml{{{\n")
            f.flush()
            path = f.name

        try:
            cfg = load_config(path)
            assert isinstance(cfg, GameConfig)
            assert cfg.game_name == "classic_7p"
        finally:
            os.unlink(path)

    def test_load_config_non_dict_yaml_returns_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("- just\n- a\n- list\n")
            f.flush()
            path = f.name

        try:
            cfg = load_config(path)
            assert isinstance(cfg, GameConfig)
            assert cfg.game_name == "classic_7p"
        finally:
            os.unlink(path)


# ======================================================================
# merge_configs
# ======================================================================


class TestMergeConfigs:
    """Tests for the merge_configs function."""

    def test_merge_simple_override(self) -> None:
        base = GameConfig()
        merged = merge_configs(base, {"max_days": 5})
        assert merged.max_days == 5

    def test_merge_preserves_non_overridden(self) -> None:
        base = GameConfig()
        merged = merge_configs(base, {"max_days": 5})
        assert merged.game_name == base.game_name
        assert merged.num_players == base.num_players

    def test_merge_nested_override(self) -> None:
        base = GameConfig()
        merged = merge_configs(
            base, {"default_model": {"model": "llama3:8b"}}
        )
        assert merged.default_model.model == "llama3:8b"
        # Other fields of default_model should be preserved
        assert merged.default_model.api_base == base.default_model.api_base

    def test_merge_voting_config(self) -> None:
        base = GameConfig()
        merged = merge_configs(base, {"voting": {"allow_no_vote": True}})
        assert merged.voting.allow_no_vote is True
        assert merged.voting.method == base.voting.method

    def test_merge_does_not_mutate_base(self) -> None:
        base = GameConfig()
        original_max_days = base.max_days
        _ = merge_configs(base, {"max_days": 99})
        assert base.max_days == original_max_days
