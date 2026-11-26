# LightZero Workspace Guide for AI Agents

## 1. Python Environment & Build Commands

- **Always use the `lightzero` conda environment** (system `python3` lacks `numpy`, `torch`, etc.):
  - Python: `/home/evan/miniconda3/envs/lightzero/bin/python3`
  - Pytest: `/home/evan/miniconda3/envs/lightzero/bin/pytest`
  - TensorBoard: `/home/evan/miniconda3/envs/lightzero/bin/tensorboard`
- **Running TensorBoard on WSL:**
  Always pass `--load_fast=false --host 0.0.0.0` (the default Rust data server can hang in WSL), and verify the port is listening (`ss -tlnp | grep 6006` or the `TensorBoard ... at http://` log line) **before** giving the user the URL.
  Always link TensorBoard as a clean Markdown hyperlink (not a raw wall-of-text URL) using the `?pinnedCards=...#timeseries` format so key `*_iter/` + async metrics auto-pin:
  `[Open TensorBoard (Pinned Key Metrics)](http://localhost:6006/?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22elo_iter/rating%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22elo_iter/win_rate_vs_champion%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22evaluator_iter/eval_episode_return_mean%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22evaluator_iter/eval/episode_length%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22learner_iter/value_loss_avg%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22learner_iter/policy_loss_avg%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22learner_iter/consistency_loss_avg%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22learner_iter/policy_entropy_avg%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22learner_iter/action0_taken_rate_avg%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22learner_iter/policy_action0_prob_avg%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22async_learner/actor_learner_balance%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22async_learner/replay_ratio%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22async_learner/ingested_envsteps%22%7D%5D#timeseries)`
- **Elo Checkpoint-Ladder Evaluation & Offline Tournaments (`lzero.entry.async_split`):**
  - Live async runs support `--eval-mode {bot,elo_ladder,both}` on both `evaluator` and `launch` (default `bot`). In `elo_ladder`/`both`, `AsyncEloEvaluator` runs paired side-swapped matches on matched map seeds, fits joint Bradley-Terry MLE ratings anchored at `bot:random = 0.0`, logs `elo_iter/*` & `elo_step/*` to `log/elo_evaluator/`, persists rungs/state under `elo_ladder/` (full `--resume` support), and saves `ckpt/elo_ckpt_best.pth.tar`.
  - Standalone offline tournaments run via `python3 -m lzero.entry.async_split tournament --config <cfg> --ckpt-dir <dir> --mode {ladder,round_robin}` (supports repeatable `--ckpt-dir`, cross-config `label=ckpt@cfg` specs, `--resume`, and `--write-tb`).
- **Rebuilding C++ / Cython MCTS (`ctree`) extensions:**
  Whenever you edit any `.cpp`, `.h`, or `.pyx` file under `lzero/mcts/ctree/`, you **must** recompile before running tests or training:
  ```bash
  /home/evan/miniconda3/envs/lightzero/bin/python3 setup.py build_ext --inplace
  ```

## 2. Regression Test Suite

Run the full Galcon + SampledEfficientZero + Elo ladder regression suite after changes:
```bash
/home/evan/miniconda3/envs/lightzero/bin/pytest \
  zoo/board_games/galcon/envs/ \
  lzero/mcts/tests/test_galcon_buffer_symmetry.py \
  lzero/mcts/tests/test_game_buffer.py \
  lzero/mcts/tests/test_ptree_sez_discrete_sampling.py \
  lzero/mcts/tests/test_sez_sign_flip.py \
  lzero/mcts/tests/test_sez_mcts_search.py \
  lzero/mcts/tests/test_game_segment_pad_over.py \
  lzero/policy/tests/test_sampled_efficientzero_policy_learn_step.py \
  lzero/policy/tests/test_galcon_sez_integration.py \
  lzero/worker/tests/test_elo_rating.py \
  lzero/entry/tests/test_async_split_elo_ladder.py -q
```

## 3. Git & Review Workflow

- **Atomic commits:** Keep every bug fix, feature, or test addition in its own clean, atomic commit.
- **Unit tests required:** Every bug fix should include a targeted unit test that would have failed before the fix.
- **Do not weaken or remove existing passing tests** (especially under `zoo/board_games/galcon/envs/`).
- **WSL line endings & permissions:** This repo runs on WSL with `core.autocrlf = input` and `core.filemode = false`. Do not introduce CRLF diffs or stray untracked experiment scripts inside the repo (put temporary experiment runners/logs outside the repo or in scratch).
- **Presenting diffs for review:** Never dump large raw `git diff` blocks into chat. Instead, render diffs as a standalone side-pane HTML artifact (via `write_file` with `UserFacing: true`, following the `generative_ui` skill) featuring per-file tabs + an "All Files" tab, an **Inline / Side-by-Side** toggle (plus a "Hide indentation-only shifts" toggle when whitespace shifts are present), line numbers, and green/red highlighting using the Antigravity theme CSS variables.
- **Wait for user approval** before pushing or rewriting approved commits.

## 4. Critical LightZero Board-Game Invariants

Keep these subtle invariants in mind when touching `lzero/mcts/`, `lzero/policy/`, `lzero/worker/`, or `zoo/board_games/`:

1. **`to_play` & Minimax Negation:**
   - In two-player zero-sum `self_play_mode` board games, `to_play` must be in `{1, 2}` so MCTS alternates minimax negation at each ply.
   - `to_play = -1` **disables** minimax negation in MCTS. During `eval_mode` of a `self_play_mode` policy, `SampledEfficientZeroPolicy._forward_eval` remaps `to_play = -1 -> 1`.
2. **`timestep.reward` vs. `info['eval_episode_return']`:**
   - `info['eval_episode_return']` is **always from Player 1's (the agent's) perspective** (`+1` P1 win, `0` draw, `-1` P2 win). It is used for logging, `ckpt_best`, and `stop_value` (not trained on by SEZ/MuZero).
   - `timestep.reward` is what gets stored in `GameSegment.reward_segment` and trained on:
     - In `self_play_mode`: from the **acting player's perspective** (buffer flips signs via `to_play_list[current_index + i] == to_play_list[current_index]`).
     - In `eval_mode` and `play_with_bot_mode`: from **Player 1's (the agent's) perspective** (if the episode terminates on Player 2's/bot's step, negate `timestep_player2.reward` and `eval_episode_return`).
3. **Do NOT Mask Illegal Actions in Policy Loss Softmax (MuZero / SEZ):**
   - Unlike AlphaZero, non-root MCTS child nodes (depth $\ge 1$) have **no simulator `action_mask`** and sample children directly from `softmax(policy_logits)`. Keeping the policy loss softmax unmasked is required so the network learns to suppress illegal action logits for deep tree search.
