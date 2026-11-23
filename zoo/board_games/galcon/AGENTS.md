# Galcon Environment & Training Guide

Scoped rules for work inside `zoo/board_games/galcon/`.

## 1. Environment & Observation Layout

- **Primary env:** `zoo/board_games/galcon/envs/galcon_env.py` (`GalconEnv`, registered as `'galcon'`).
- **Active training config:** `zoo/board_games/galcon/config/galcon_sez_sp_mode_config_SIMPLE_1map.py`.
- **Observation shape:** `(C, H, W)` where `C = 8 (planets) + 2 * (fleet_top_k + 1) * 5 (friendly/enemy fleets) + 16 (landings)`.
  - Currently `C = 40` for `fleet_top_k = 1` (`(40, 8, 8)` on `SIMPLE_1map`) only because fleet redirecting is temporarily disabled; expect `C` (and `reflect_game_segment` channel math) to change once fleet redirecting is re-enabled.
  - `frame_stack_num = 1` (the observation is already Markov).
- **Action encoding:**
  - `0` = Pass (always legal).
  - `1 .. (W * H * W * H)` = Directed send `(source_cell -> target_cell)`, encoded as `1 + source_cell * (W * H) + target_cell`.
  - Sending from unowned/empty cells or planets with `< min_send_ships` must remain masked out (`action_mask[a] == 0`).

## 2. Step Counting, Segments, & Rewards

- `self._step_count` increments once per `_player_step` (**1 step = 1 ply**, not 1 full round of 2 plies).
- Rewards are **sparse terminal** (`0` during play; `+1` / `-1` / `0` on game end).
- For short episodes (e.g. `max_episode_steps = 100`), keep `game_segment_length = max_episode_steps` and `td_steps = max_episode_steps` with `discount_factor = 1` so each game fits in 1 segment and trains directly on the true terminal outcome.
- In `eval_mode` and `play_with_bot_mode`, `GalconEnv.step()` executes Player 1 (agent) then Player 2 (bot). Both `reward` and `info['eval_episode_return']` must always be from **Player 1's perspective**, including when the game ends on an even ply (e.g. `max_episode_steps = 100`).

## 3. Symmetry Augmentation

- Geometric X, Y, and XY reflections for Galcon live in `reflect_game_segment()` (`lzero/mcts/buffer/game_buffer.py`), enabled via `symmetric_augment_x`, `symmetric_augment_y`, and `symmetric_augment_type = 'Galcon'`.
- Reflections must flip spatial grids, invert relative cell coordinates (`1.0 - x` / `1.0 - y`) on occupied planet/fleet cells, and remap `action_segment`, `action_mask_segment`, and `root_sampled_actions` consistently.
- Always run `test_galcon_env.py` and `test_galcon_buffer_symmetry.py` after any observation-channel or action-encoding change.

## 4. Galcon Strategy & Training Diagnostics

- **Passing (`action 0`) is generally VERY BAD in Galcon:** A strong Galcon bot should have a **low pass rate**, actively expanding and contesting planets.
- **Exception on tiny maps (`grid_max = 50`):** Only when planets are packed unnaturally close together (e.g. enemy home is ~1 tick away) can passing temporarily look viable because spending ships on a neutral leaves the home planet vulnerable to an immediate rush. Do **not** mistake an 80–95% pass rate for good Galcon play—treat high `action0_taken_rate_avg` as a red flag for either overly cramped map geometry or MCTS/policy collapse onto Pass.
- **Collect MCTS temperature:** Always use `manual_temperature_decay = False` and `fixed_temperature_value = 1.0` during Galcon self-play (default `0.25` causes early entropy collapse and dynamics-model starvation).
