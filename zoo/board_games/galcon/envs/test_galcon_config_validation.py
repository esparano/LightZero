# -*- coding: utf-8 -*-
"""Unit tests for ``GalconEnv.validate_config``."""

import copy
import importlib

import pytest
from easydict import EasyDict

from zoo.board_games.galcon.envs.galcon_env import GalconEnv


def _make_cfg(max_episode_steps=100, td_steps=100, game_segment_length=100, allow_mismatch=False):
    return EasyDict(
        dict(
            env=dict(max_episode_steps=max_episode_steps),
            policy=dict(
                td_steps=td_steps,
                game_segment_length=game_segment_length,
                allow_episode_horizon_mismatch=allow_mismatch,
            ),
        )
    )


@pytest.mark.unittest
class TestGalconConfigValidation:

    def test_matching_horizon_passes(self):
        GalconEnv.validate_config(_make_cfg(100, 100, 100))

    def test_td_steps_mismatch_raises(self):
        cfg = _make_cfg(max_episode_steps=200, td_steps=100, game_segment_length=200)
        with pytest.raises(ValueError, match=r'policy\.td_steps=100 != env\.max_episode_steps=200'):
            GalconEnv.validate_config(cfg)

    def test_game_segment_length_mismatch_raises(self):
        cfg = _make_cfg(max_episode_steps=200, td_steps=200, game_segment_length=100)
        with pytest.raises(ValueError, match=r'policy\.game_segment_length=100 != env\.max_episode_steps=200'):
            GalconEnv.validate_config(cfg)

    def test_both_mismatch_raises_with_opt_out_hint(self):
        cfg = _make_cfg(max_episode_steps=200, td_steps=100, game_segment_length=100)
        with pytest.raises(ValueError, match=r'allow_episode_horizon_mismatch=True'):
            GalconEnv.validate_config(cfg)

    def test_explicit_opt_out_allows_mismatch(self):
        cfg = _make_cfg(
            max_episode_steps=500, td_steps=25, game_segment_length=100, allow_mismatch=True
        )
        GalconEnv.validate_config(cfg)

    @pytest.mark.parametrize(
        'module_path',
        [
            'zoo.board_games.galcon.config.galcon_sez_sp_mode_config',
            'zoo.board_games.galcon.config.galcon_sez_sp_mode_config_SIMPLE',
            'zoo.board_games.galcon.config.galcon_sez_sp_mode_config_SIMPLE_1map',
            'zoo.board_games.galcon.config.galcon_sez_bot_mode_config',
        ],
    )
    def test_all_galcon_configs_pass_validation_and_catch_override(self, module_path):
        mod = importlib.import_module(module_path)
        cfg = copy.deepcopy(mod.main_config)

        # Unmodified config must pass
        GalconEnv.validate_config(cfg)

        # Mutating only env.max_episode_steps must be caught
        cfg.env.max_episode_steps = int(cfg.env.max_episode_steps) + 50
        with pytest.raises(ValueError, match=r'Galcon episode-horizon mismatch'):
            GalconEnv.validate_config(cfg)
