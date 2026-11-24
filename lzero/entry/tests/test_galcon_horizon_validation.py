# -*- coding: utf-8 -*-
"""Unit tests for ``validate_galcon_episode_horizon``."""

import copy

import pytest
from easydict import EasyDict

from lzero.entry.utils import validate_galcon_episode_horizon


def _make_galcon_cfgs(max_episode_steps=100, td_steps=100, game_segment_length=100, allow_mismatch=False):
    cfg = EasyDict(
        dict(
            env=dict(max_episode_steps=max_episode_steps),
            policy=dict(
                td_steps=td_steps,
                game_segment_length=game_segment_length,
                allow_episode_horizon_mismatch=allow_mismatch,
            ),
        )
    )
    create_cfg = EasyDict(dict(env=dict(type='galcon')))
    return cfg, create_cfg


@pytest.mark.unittest
class TestGalconEpisodeHorizonValidation:

    def test_matching_horizon_passes(self):
        cfg, create_cfg = _make_galcon_cfgs(100, 100, 100)
        validate_galcon_episode_horizon(cfg, create_cfg)

    def test_td_steps_mismatch_raises(self):
        cfg, create_cfg = _make_galcon_cfgs(max_episode_steps=200, td_steps=100, game_segment_length=200)
        with pytest.raises(ValueError, match=r'policy\.td_steps=100 != env\.max_episode_steps=200'):
            validate_galcon_episode_horizon(cfg, create_cfg)

    def test_game_segment_length_mismatch_raises(self):
        cfg, create_cfg = _make_galcon_cfgs(max_episode_steps=200, td_steps=200, game_segment_length=100)
        with pytest.raises(ValueError, match=r'policy\.game_segment_length=100 != env\.max_episode_steps=200'):
            validate_galcon_episode_horizon(cfg, create_cfg)

    def test_both_mismatch_raises_with_opt_out_hint(self):
        cfg, create_cfg = _make_galcon_cfgs(max_episode_steps=200, td_steps=100, game_segment_length=100)
        with pytest.raises(ValueError, match=r'allow_episode_horizon_mismatch=True'):
            validate_galcon_episode_horizon(cfg, create_cfg)

    def test_explicit_opt_out_allows_mismatch(self):
        cfg, create_cfg = _make_galcon_cfgs(
            max_episode_steps=500, td_steps=25, game_segment_length=100, allow_mismatch=True
        )
        validate_galcon_episode_horizon(cfg, create_cfg)

    def test_non_galcon_env_is_ignored(self):
        cfg = EasyDict(
            dict(
                env=dict(max_episode_steps=500),
                policy=dict(td_steps=5, game_segment_length=200),
            )
        )
        create_cfg = EasyDict(dict(env=dict(type='chinese_chess')))
        validate_galcon_episode_horizon(cfg, create_cfg)

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
        import importlib

        mod = importlib.import_module(module_path)
        cfg = copy.deepcopy(mod.main_config)
        create_cfg = copy.deepcopy(mod.create_config)

        # Unmodified config must pass
        validate_galcon_episode_horizon(cfg, create_cfg)

        # Mutating only env.max_episode_steps must be caught
        cfg.env.max_episode_steps = int(cfg.env.max_episode_steps) + 50
        with pytest.raises(ValueError, match=r'Galcon episode-horizon mismatch'):
            validate_galcon_episode_horizon(cfg, create_cfg)
