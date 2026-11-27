"""Interactive 20 Hz HTML replay recorder & generator for Galcon (C++17 multi-ship simulator).

Records exact 20 Hz C++ physics substeps (``sub_dt = tick_seconds / physics_substeps``) so that
planet ship counts and ownership update on the exact substep each ship launches or impacts,
alongside planet-rim impact coordinates for Galcon 2 spark explosions.

When a local Steam Galcon 2 installation is present (auto-detected on Windows/WSL or specified via
``--g2-data-dir``), real Galcon 2 ship sprites (``ship-0.png``, ``ship-1.png``, ``ship-7.png``,
``ship-4.png``) and explosion sprites (``spark.png``, ``spark-1.png``) are alpha-keyed, tinted,
and embedded as data URIs into the generated self-contained HTML file. When absent, the viewer
gracefully falls back to crisp canvas-drawn hollow triangles and radial spark flashes.

Example::

    python3 -m zoo.board_games.galcon.eval.replay_viewer \\
        --config zoo/board_games/galcon/config/galcon_sez_sp_mode_config_SIMPLE_1map.py \\
        --match "bot:mcts:16 vs bot:random @ seed=42" \\
        --output ./video/galcon/replays.html
"""

from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import io
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from zoo.board_games.galcon.envs.cpp_sim import (
    DEFAULT_PHYSICS_SUBSTEPS,
    GalconSim as _CppSimFallback,
    HAS_FAST_SIM_CPP,
    step_physics_cpp,
)
from zoo.board_games.galcon.envs.galcon_env import GalconEnv

TEMPLATE_PATH = Path(__file__).with_name("replay_viewer_template.html")

G2_SHIP_SKINS: Tuple[Tuple[str, str, str], ...] = (
    ("classic", "G2 Classic (ship-0)", "ship-0.png"),
    ("solid", "G2 Solid (ship-1)", "ship-1.png"),
    ("chevron", "G2 Chevron (ship-7)", "ship-7.png"),
    ("triforce", "G2 Triforce (ship-4)", "ship-4.png"),
)

DEFAULT_G2_DATA_DIRS: Tuple[Path, ...] = (
    Path("/mnt/c/Program Files (x86)/Steam/steamapps/common/Galcon 2/data"),
    Path("/mnt/c/Program Files/Steam/steamapps/common/Galcon 2/data"),
    Path(r"C:\Program Files (x86)\Steam\steamapps\common\Galcon 2\data"),
    Path(r"C:\Program Files\Steam\steamapps\common\Galcon 2\data"),
)


def find_g2_data_dir(explicit_dir: Optional[str | os.PathLike] = None) -> Optional[Path]:
    """Return the first existing Galcon 2 ``data/`` directory containing ``ship-0.png``."""
    candidates: List[Path] = []
    if explicit_dir is not None:
        candidates.append(Path(explicit_dir))
    env_dir = os.environ.get("GALCON2_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(DEFAULT_G2_DATA_DIRS)
    for path in candidates:
        try:
            if (path / "ship-0.png").is_file():
                return path
        except OSError:
            continue
    return None


def _tint_g2_sprite(png_path: Path, rgb: Tuple[int, int, int]) -> str:
    """Alpha-key a black-background grayscale G2 glow sprite into a neon-tinted PNG data URI."""
    from PIL import Image

    im = Image.open(png_path).convert("L")
    w, h = im.size
    out = Image.new("RGBA", (w, h))
    src = im.load()
    dst = out.load()
    r0, g0, b0 = rgb
    for y in range(h):
        for x in range(w):
            v = src[x, y] / 255.0
            if v <= 0.004:
                dst[x, y] = (0, 0, 0, 0)
            else:
                core = (v ** 3) * 0.42
                r = int(min(255, round(r0 + (255 - r0) * core)))
                g = int(min(255, round(g0 + (255 - g0) * core)))
                b = int(min(255, round(b0 + (255 - b0) * core)))
                a = int(min(255, round(v * 255)))
                dst[x, y] = (r, g, b, a)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _alpha_key_g2_color_fx(png_path: Path) -> str:
    """Convert a black-background color G2 FX sprite (spark.png / spark-1.png) into RGBA."""
    from PIL import Image

    im = Image.open(png_path).convert("RGB")
    w, h = im.size
    out = Image.new("RGBA", (w, h))
    src = im.load()
    dst = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            mx = max(r, g, b)
            if mx <= 2:
                dst[x, y] = (0, 0, 0, 0)
            else:
                scale = 255.0 / mx
                dst[x, y] = (
                    min(255, int(round(r * scale))),
                    min(255, int(round(g * scale))),
                    min(255, int(round(b * scale))),
                    mx,
                )
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_g2_sprites(g2_data_dir: Optional[str | os.PathLike] = None) -> Dict[str, Dict[str, Any]]:
    """Load real Galcon 2 ship and spark sprites if available; otherwise return empty dicts."""
    root = find_g2_data_dir(g2_data_dir)
    if root is None:
        return {"ships": {}, "fx": {}}

    try:
        import PIL  # noqa: F401
    except ImportError:
        return {"ships": {}, "fx": {}}

    assets: Dict[str, Dict[str, Any]] = {"ships": {}, "fx": {}}
    for key, label, fname in G2_SHIP_SKINS:
        fpath = root / fname
        if fpath.is_file():
            assets["ships"][key] = {
                "label": label,
                "file": fname,
                "p1": _tint_g2_sprite(fpath, (74, 158, 255)),
                "p2": _tint_g2_sprite(fpath, (255, 92, 92)),
            }

    spark_path = root / "spark.png"
    spark_glow_path = root / "spark-1.png"
    if spark_path.is_file():
        assets["fx"]["spark_hostile"] = _alpha_key_g2_color_fx(spark_path)
    if spark_glow_path.is_file():
        assets["fx"]["spark_glow"] = _alpha_key_g2_color_fx(spark_glow_path)
    return assets


def _find_landed_units(
    before_units: Sequence[Sequence[float]],
    before_headings: Sequence[Sequence[float]],
    after_units: Sequence[Sequence[float]],
    step_dist: float,
) -> List[Tuple[float, float]]:
    """Order-preserving DP match over a single 0.05s substep to identify landed unit coordinates."""
    n = len(before_units)
    m = len(after_units)
    if m >= n:
        return []
    if m == 0:
        return [(float(u[0]), float(u[1])) for u in before_units]

    inf = 1e18
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 0.0
    for i in range(1, n + 1):
        u = before_units[i - 1]
        h = before_headings[i - 1]
        px = float(u[0]) + step_dist * float(h[0])
        py = float(u[1]) + step_dist * float(h[1])
        for j in range(1, min(i, m) + 1):
            best = dp[i - 1][j]
            chose = False
            b = after_units[j - 1]
            dx = px - float(b[0])
            dy = py - float(b[1])
            c = dp[i - 1][j - 1] + dx * dx + dy * dy
            if c < best:
                best = c
                chose = True
            dp[i][j] = best
            take[i][j] = chose

    matched = set()
    i, j = n, m
    while j > 0 and i > 0:
        if take[i][j]:
            matched.add(i - 1)
            i -= 1
            j -= 1
        else:
            i -= 1
    return [(float(before_units[k][0]), float(before_units[k][1])) for k in range(n) if k not in matched]


def _step_one_substep_physics(env: GalconEnv, sub_dt: float) -> None:
    """Advance GalconEnv physics by exactly one substep (bit-exact with ``_advance_one_tick_cpp``)."""
    adapter = object.__new__(_CppSimFallback)
    adapter.planets = {p.id: p for p in env.planets}
    adapter.fleets = {f.id: f for f in env.fleets}
    adapter.tick_dt = sub_dt
    adapter.fleet_speed = env.fleet_speed
    adapter.physics_substeps = 1

    if HAS_FAST_SIM_CPP:
        step_physics_cpp(adapter)
    else:
        adapter._step_physics_python()

    for p in env.planets:
        p.neutral = (p.owner == env.PLAYER_NEUTRAL)
    env.fleets = [f for f in env.fleets if f.id in adapter.fleets]


def _compact_snapshot(
    env: GalconEnv,
    step_idx: int,
    sub_idx: int,
    t_sec: float,
    act_rec: Optional[List[int]],
    hits: Optional[List[List[int]]],
) -> Dict[str, Any]:
    p_dyn = [[int(p.owner), int(round(float(p.ships) * 10))] for p in env.planets]
    fleets: List[List[Any]] = []
    total_units = 0
    for f in env.fleets:
        flat: List[int] = []
        units = f.units or [[f.x, f.y]]
        headings = f.unit_headings if (f.unit_headings and len(f.unit_headings) == len(units)) else None
        for idx, (ux, uy) in enumerate(units):
            if headings is not None:
                ang = math.atan2(headings[idx][1], headings[idx][0])
            else:
                tp = env.planets[f.target]
                ang = math.atan2(tp.y - uy, tp.x - ux)
            flat.append(int(round(float(ux) * 10)))
            flat.append(int(round(float(uy) * 10)))
            flat.append(int(round(ang * 100)))
        total_units += len(units)
        fleets.append([
            int(f.id),
            int(f.owner),
            int(round(float(f.ships) * 10)),
            int(f.target),
            flat,
        ])
    rec: Dict[str, Any] = {
        "k": int(step_idx),
        "sb": int(sub_idx),
        "t": int(round(t_sec * 100)),
        "p": p_dyn,
        "f": fleets,
        "s": [
            int(round(float(env._total_ships(1)))),
            int(round(float(env._total_ships(2)))),
            int(round(float(env._total_production(1)))),
            int(round(float(env._total_production(2)))),
        ],
        "u": total_units,
    }
    if act_rec is not None:
        rec["a"] = act_rec
    if hits:
        rec["h"] = hits
    return rec


def record_match_20hz(
    env_cfg: Dict[str, Any],
    player1: Any,
    player2: Any,
    map_seed: int,
    title: str = "",
    subtitle: str = "",
    category: str = "Match",
) -> Dict[str, Any]:
    """Play one self-play match and record exact 20 Hz C++ substep frames and ship impact events.

    ``player1`` and ``player2`` follow the ``MatchPlayer`` interface from
    ``lzero.worker.head_to_head`` (supporting both ``PolicyMatchPlayer`` and
    ``EnvBotMatchPlayer``).
    """
    cfg = copy.deepcopy(env_cfg)
    cfg["battle_mode"] = "self_play_mode"
    cfg["render_mode"] = None
    cfg["use_cpp_sim"] = True
    cfg["map_seed"] = int(map_seed)
    cfg["prob_random_agent"] = 0
    cfg["prob_expert_agent"] = 0

    env = GalconEnv(cfg)
    init_obs = env.reset()
    player1.reset([0])
    player2.reset([0])

    max_stack = max(int(getattr(player1, "frame_stack_num", 1)), int(getattr(player2, "frame_stack_num", 1)))
    obs_buf = deque([np.asarray(init_obs["observation"]) for _ in range(max_stack)], maxlen=max_stack)
    action_mask = np.asarray(init_obs["action_mask"])
    to_play = int(np.asarray(init_obs["to_play"]).reshape(-1)[0])
    timestep_idx = int(np.asarray(init_obs.get("timestep", 0)).reshape(-1)[0])

    substeps = max(1, int(env.physics_substeps))
    sub_dt = float(env.tick_seconds) / float(substeps)
    sub_step_dist = float(env.fleet_speed) * sub_dt

    planets_static = [
        [
            int(p.id),
            round(float(p.x), 2),
            round(float(p.y), 2),
            int(round(float(p.production))),
            round(float(p.radius), 2),
        ]
        for p in env.planets
    ]

    frames = [_compact_snapshot(env, 0, 0, 0.0, None, None)]
    max_units = 0
    sends = {1: 0, 2: 0}
    winner = -1

    while True:
        acting_player = to_play
        cur_p = player1 if acting_player == 1 else player2
        p_stack = int(getattr(cur_p, "frame_stack_num", 1))
        act_dict = cur_p.select_actions(
            {0: env},
            [0],
            [list(obs_buf)[-p_stack:]],
            [action_mask],
            [to_play],
            [timestep_idx],
        )
        action = int(act_dict[0])
        if action not in env.legal_actions:
            action = env.pass_action

        act_type, sx, sy, tx, ty = env.decode_action_with_type(action)
        act_rec: Optional[List[int]] = None
        if act_type == "send" and sx is not None:
            sp = env._planet_at_cell(sx, sy, owner=acting_player)
            tp = env._planet_at_cell(tx, ty)
            if sp is not None and tp is not None and sp.id != tp.id:
                act_rec = [acting_player, int(sp.id), int(tp.id)]
                sends[acting_player] += 1

        step_idx = env._step_count + 1
        env._apply_action(action)

        for sb in range(1, substeps + 1):
            pre_owners = {p.id: int(p.owner) for p in env.planets}
            pre_fleets = {
                f.id: (
                    int(f.owner),
                    int(f.target),
                    [list(u) for u in (f.units or [[f.x, f.y]])],
                    [list(h) for h in (f.unit_headings or [[1.0, 0.0]])],
                )
                for f in env.fleets
            }

            _step_one_substep_physics(env, sub_dt)

            post_fleets = {f.id: f for f in env.fleets}
            hits: List[List[int]] = []
            for fid, (f_owner, f_tgt, pre_u, pre_h) in pre_fleets.items():
                pf = post_fleets.get(fid)
                post_u = pf.units if (pf is not None and pf.units) else []
                if len(post_u) < len(pre_u):
                    landed_pts = _find_landed_units(pre_u, pre_h, post_u, sub_step_dist)
                    tp_planet = env.planets[f_tgt]
                    hostile = 1 if (pre_owners[f_tgt] != f_owner) else 0
                    for lx, ly in landed_pts:
                        dx = lx - tp_planet.x
                        dy = ly - tp_planet.y
                        d = math.hypot(dx, dy)
                        if d > 1e-6:
                            ix = tp_planet.x + (dx / d) * tp_planet.radius
                            iy = tp_planet.y + (dy / d) * tp_planet.radius
                        else:
                            ix, iy = lx, ly
                        hits.append([int(round(ix * 10)), int(round(iy * 10)), f_owner, hostile])

            t_sec = (step_idx - 1) * float(env.tick_seconds) + sb * sub_dt
            snap = _compact_snapshot(env, step_idx, sb, t_sec, act_rec if sb == 1 else None, hits or None)
            max_units = max(max_units, snap["u"])
            frames.append(snap)

        env._step_count += 1
        done, winner = env.get_done_winner()
        env._current_player = env.next_player

        if done:
            break

        next_obs = env.observe()
        obs_buf.append(np.asarray(next_obs["observation"]))
        action_mask = np.asarray(next_obs["action_mask"])
        to_play = int(np.asarray(next_obs["to_play"]).reshape(-1)[0])
        timestep_idx = int(np.asarray(next_obs.get("timestep", timestep_idx + 1)).reshape(-1)[0])

    return {
        "title": title or f"{player1.player_id} vs {player2.player_id}",
        "subtitle": subtitle,
        "category": category,
        "map_seed": int(map_seed),
        "p1_name": str(player1.player_id),
        "p2_name": str(player2.player_id),
        "winner": int(winner),
        "grid_max_x": float(env.grid_max_x),
        "grid_max_y": float(env.grid_max_y),
        "grid_square_size": float(env.grid_square_size),
        "tick_seconds": float(env.tick_seconds),
        "sub_dt": float(sub_dt),
        "substeps": int(substeps),
        "sends": [sends[1], sends[2]],
        "max_units": int(max_units),
        "planets": planets_static,
        "frames": frames,
    }


def render_replay_html(
    replays: Sequence[Dict[str, Any]],
    g2_data_dir: Optional[str | os.PathLike] = None,
    template_path: Path = TEMPLATE_PATH,
) -> str:
    """Embed recorded 20 Hz replays and optional real G2 sprites into a self-contained HTML string."""
    html = Path(template_path).read_text(encoding="utf-8")
    replays_json = json.dumps(list(replays), separators=(",", ":"))
    sprites_json = json.dumps(load_g2_sprites(g2_data_dir), separators=(",", ":"))
    if "/*__REPLAY_DATA__*/[]" not in html:
        raise ValueError(f"Missing replay placeholder in {template_path}")
    if "/*__G2_SPRITES__*/{ ships: {}, fx: {} }" not in html:
        raise ValueError(f"Missing sprites placeholder in {template_path}")
    return (
        html.replace("/*__REPLAY_DATA__*/[]", replays_json, 1)
        .replace("/*__G2_SPRITES__*/{ ships: {}, fx: {} }", sprites_json, 1)
    )


def _load_config_module(config_path: str):
    spec = importlib.util.spec_from_file_location("galcon_replay_cfg", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return copy.deepcopy(mod.main_config), copy.deepcopy(mod.create_config)


class _GalconRuleBotPlayer:
    """Lightweight MatchPlayer wrapper for built-in Galcon bots without requiring compiled MCTS ctree."""

    def __init__(self, bot_type: str, player_id: Optional[str] = None) -> None:
        self.bot_type = str(bot_type).strip().lower()
        self.player_id = player_id or f"bot:{self.bot_type}"
        self.frame_stack_num = 1

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        return

    def select_actions(self, envs, env_ids, obs_stacks, action_masks, to_plays, timesteps) -> Dict[int, int]:
        from zoo.board_games.galcon.envs.rule_bot import (
            GalconFixedPolicyBot,
            GalconMCTSBot,
            GalconPassBot,
            GalconRandomBot,
        )

        actions: Dict[int, int] = {}
        for env_id in env_ids:
            env = envs[env_id]
            k = self.bot_type
            if k == "random":
                bot = GalconRandomBot(env)
            elif k == "fixed":
                bot = GalconFixedPolicyBot(env)
            elif k == "pass":
                bot = GalconPassBot(env)
            elif k == "mcts":
                bot = GalconMCTSBot(env)
            elif k.startswith("mcts:") or (k.startswith("mcts") and k[4:].isdigit()):
                sims = int(k.split(":", 1)[1] if ":" in k else k[4:])
                bot = GalconMCTSBot(env, num_simulations=sims)
            else:
                raise ValueError(f"Unsupported Galcon bot_type: {self.bot_type!r}")
            actions[env_id] = int(bot.get_action())
        return actions


def _build_player_from_spec(spec_str: str, main_cfg: Any, create_cfg: Any, device: str = "cpu"):
    s = spec_str.strip()
    if s.lower().startswith("bot:"):
        bot_type = s[4:].strip()
        return _GalconRuleBotPlayer(bot_type=bot_type, player_id=f"bot:{bot_type}")
    if s.lower() in {"random", "fixed", "pass", "mcts"} or s.lower().startswith("mcts:"):
        return _GalconRuleBotPlayer(bot_type=s, player_id=f"bot:{s}")

    from lzero.worker.head_to_head import PolicyMatchPlayer

    ckpt_path = s[5:].strip() if s.lower().startswith("ckpt:") else s
    label = Path(ckpt_path).parent.parent.name + "/" + Path(ckpt_path).stem
    import torch
    from ding.config import compile_config
    from ding.policy import create_policy

    mcfg = copy.deepcopy(main_cfg)
    ccfg = copy.deepcopy(create_cfg)
    mcfg.policy.cuda = (device != "cpu")
    mcfg.policy.device = device
    mcfg.policy.eval_freq = int(1e9)
    mcfg.env.collector_env_num = 1
    mcfg.env.evaluator_env_num = 1
    ccfg.env_manager.type = "base"
    compiled = compile_config(mcfg, seed=0, create_cfg=ccfg, auto=True, save_cfg=False)
    policy = create_policy(compiled.policy, enable_field=["eval"])
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        policy.eval_mode.load_state_dict(state)
    else:
        policy._model.load_state_dict(state)
    policy._model.eval()
    return PolicyMatchPlayer(player_id=label, policy=policy, policy_config=compiled.policy)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record 20 Hz Galcon C++17 replays into a self-contained HTML viewer.")
    parser.add_argument(
        "--config",
        default="zoo/board_games/galcon/config/galcon_sez_sp_mode_config_SIMPLE_1map.py",
        help="Path to LightZero Galcon config file.",
    )
    parser.add_argument(
        "--p1",
        default="bot:mcts:16",
        help="Player 1 spec: bot:random | bot:mcts:16 | ckpt:/path/to/ckpt.pth.tar",
    )
    parser.add_argument(
        "--p2",
        default="bot:random",
        help="Player 2 spec: bot:random | bot:mcts:16 | ckpt:/path/to/ckpt.pth.tar",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Map seeds to record.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy checkpoints (cpu or cuda).")
    parser.add_argument("--g2-data-dir", default=None, help="Optional Galcon 2 data/ directory for real G2 sprites.")
    parser.add_argument("--output", required=True, help="Output self-contained HTML file path.")
    args = parser.parse_args(argv)

    main_cfg, create_cfg = _load_config_module(args.config)
    p1 = _build_player_from_spec(args.p1, main_cfg, create_cfg, device=args.device)
    p2 = _build_player_from_spec(args.p2, main_cfg, create_cfg, device=args.device)

    replays = []
    for seed in args.seeds:
        replays.append(
            record_match_20hz(
                main_cfg.env,
                p1,
                p2,
                map_seed=seed,
                title=f"{p1.player_id} vs {p2.player_id}",
                subtitle=f"20 Hz C++17 multi-ship simulator (seed {seed})",
            )
        )

    html = render_replay_html(replays, g2_data_dir=args.g2_data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {len(replays)} replay(s) to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
