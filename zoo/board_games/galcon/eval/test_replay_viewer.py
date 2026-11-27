"""Unit tests for the Galcon 20 Hz HTML replay recorder and viewer generator."""

import copy
import math
from typing import Any, Dict, List

import pytest

from zoo.board_games.galcon.envs.galcon_env import GalconEnv
from zoo.board_games.galcon.eval.replay_viewer import (
    BUNDLED_ASSETS_DIR,
    load_g2_sprites,
    record_match_20hz,
    render_replay_html,
)


class _ScriptedPlayer:
    """Deterministic test player issuing a fixed list of (src_planet_id, dst_planet_id) orders."""

    def __init__(self, player_id: str, orders: List[Any]) -> None:
        self.player_id = player_id
        self.frame_stack_num = 1
        self._orders = list(orders)
        self._idx = 0

    def reset(self, env_ids=None) -> None:
        self._idx = 0

    def select_actions(self, envs, env_ids, obs_stacks, action_masks, to_plays, timesteps) -> Dict[int, int]:
        env = envs[env_ids[0]]
        if self._idx >= len(self._orders) or self._orders[self._idx] is None:
            self._idx += 1
            return {env_ids[0]: env.pass_action}
        src_id, dst_id = self._orders[self._idx]
        self._idx += 1
        sp = env.planets[src_id]
        tp = env.planets[dst_id]
        sx, sy = env._world_to_grid(sp.x, sp.y)
        tx, ty = env._world_to_grid(tp.x, tp.y)
        return {env_ids[0]: env.encode_action(sx, sy, tx, ty)}


def _base_env_cfg(max_steps: int = 16) -> Dict[str, Any]:
    return {
        "num_planets": 6,
        "min_send_ships": 1,
        "send_ratio": 0.5,
        "tick_seconds": 0.25,
        "fleet_speed": 40.0,
        "max_episode_steps": max_steps,
        "map_seed": 42,
        "grid_square_size": 25.0,
        "grid_max_x": 100.0,
        "grid_max_y": 100.0,
        "home_distance_fraction": 0.8,
        "neutral_min_cost": 0,
        "neutral_max_cost": 10,
        "neutral_min_production": 15,
        "neutral_max_production": 100,
        "fleet_top_k": 1,
        "use_cpp_sim": True,
    }


def test_record_match_20hz_matches_standard_galcon_env_step():
    """Stepping 5x(sub_dt=0.05s, substeps=1) must match standard GalconEnv.step() planet-for-planet."""
    env_cfg = _base_env_cfg(max_steps=18)
    p1_orders = [(0, 3), None, (0, 1)]
    p2_orders = [(1, 2), None, (1, 0)]

    p1 = _ScriptedPlayer("p1", p1_orders)
    p2 = _ScriptedPlayer("p2", p2_orders)
    rec = record_match_20hz(env_cfg, p1, p2, map_seed=42)

    # Replay the exact same scripted orders via standard GalconEnv.step()
    std_cfg = copy.deepcopy(env_cfg)
    std_cfg["battle_mode"] = "self_play_mode"
    env = GalconEnv(std_cfg)
    env.reset()
    p1_std = _ScriptedPlayer("p1", p1_orders)
    p2_std = _ScriptedPlayer("p2", p2_orders)
    while True:
        cur = p1_std if env._current_player == 1 else p2_std
        act = cur.select_actions({0: env}, [0], [[]], [[]], [env._current_player], [env._step_count])[0]
        ts = env.step(act)
        if ts.done:
            break

    assert rec["substeps"] == 5
    assert rec["sub_dt"] == pytest.approx(0.05)
    assert len(rec["frames"]) == 1 + env._step_count * 5

    last_planets = rec["frames"][-1]["p"]
    for idx, p in enumerate(env.planets):
        assert last_planets[idx][0] == p.owner
        assert last_planets[idx][1] == int(round(p.ships * 10))


def test_immediate_planet_drop_and_hostile_vs_friendly_hits():
    """Send immediately drops source ships at sb=1; hostile hits get hostile=1 and friendly reinforcements get hostile=0."""
    env_cfg = _base_env_cfg(max_steps=20)
    # P1 sends P0 -> P3 (neutral, hostile=1) on tick 1, then reinforces P3 from P0 once P3 is captured
    p1 = _ScriptedPlayer("p1", [(0, 3), None, None, None, None, None, (0, 3)])
    p2 = _ScriptedPlayer("p2", [None])

    rec = record_match_20hz(env_cfg, p1, p2, map_seed=42)
    frames = rec["frames"]

    # Frame 0: P0 starts at 100.0 ships (1000 tenths)
    assert frames[0]["p"][0][1] == 1000
    # Frame 1 (step 1, sb=1): P0 dropped immediately to ~50 ships (+1 substep production ~0.1)
    assert frames[1]["k"] == 1
    assert frames[1]["sb"] == 1
    assert frames[1]["a"] == [1, 0, 3]
    assert frames[1]["p"][0][1] <= 510

    all_hits = [hit for fr in frames for hit in fr.get("h", [])]
    hostile_hits = [h for h in all_hits if h[3] == 1]
    friendly_hits = [h for h in all_hits if h[3] == 0]
    assert len(hostile_hits) > 0, "Expected hostile impact events when capturing neutral planet P3"
    assert len(friendly_hits) > 0, "Expected friendly reinforcement impact events (hostile=0) after P3 capture"


def test_bundled_original_sprites_load_by_default():
    """Repo-bundled original ship and spark PNG sprites exist and encode into RGBA data URIs."""
    for fname in ("ship-classic.png", "ship-solid.png", "ship-chevron.png", "ship-triforce.png", "spark.png", "spark-glow.png"):
        assert (BUNDLED_ASSETS_DIR / fname).is_file(), f"Missing bundled asset {fname}"

    sprites = load_g2_sprites()
    assert set(sprites["ships"].keys()) == {"classic", "solid", "chevron", "triforce"}
    for skin in sprites["ships"].values():
        assert skin["p1"].startswith("data:image/png;base64,")
        assert skin["p2"].startswith("data:image/png;base64,")
    assert sprites["fx"]["spark_hostile"].startswith("data:image/png;base64,")
    assert sprites["fx"]["spark_glow"].startswith("data:image/png;base64,")


def test_render_replay_html_embeds_replays_and_bundled_sprites(tmp_path):
    """render_replay_html embeds replay JSON and bundled original sprites with no unresolved placeholders."""
    env_cfg = _base_env_cfg(max_steps=6)
    p1 = _ScriptedPlayer("p1", [(0, 3)])
    p2 = _ScriptedPlayer("p2", [(1, 2)])
    rec = record_match_20hz(env_cfg, p1, p2, map_seed=42, title="Unit Test Match")

    html = render_replay_html([rec], g2_data_dir=tmp_path / "nonexistent_g2")
    assert "/*__REPLAY_DATA__*/[]" not in html
    assert "/*__G2_SPRITES__*/{ ships: {}, fx: {} }" not in html
    assert "data:image/png;base64," in html
    assert "Unit Test Match" in html
    assert "bakeSmoothedVelocityAngles" in html
    assert "if (hit[3] !== 1) continue;" in html
