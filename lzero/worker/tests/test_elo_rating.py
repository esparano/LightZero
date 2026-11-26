# -*- coding: utf-8 -*-
"""Unit tests for Bradley-Terry MLE Elo solver and checkpoint ladder management."""

import json
import math

import pytest

from lzero.worker.elo_rating import (
    EloLadder,
    MatchRecord,
    PlayerEntry,
    expected_score,
    fit_bradley_terry_elo,
)


@pytest.mark.unittest
def test_bradley_terry_exact_two_player_mle():
    # Exact MLE without regularization: 75% win rate => +400 * log10(3) ~= +190.8485 Elo
    expected_delta = 400.0 * math.log10(3.0)
    matches = [
        MatchRecord(player_a='p1', player_b='bot:random', wins_a=15, draws=0, losses_a=5),
        MatchRecord(player_a='p_even', player_b='bot:random', wins_a=5, draws=2, losses_a=5),
        MatchRecord(player_a='p_weak', player_b='bot:random', wins_a=5, draws=0, losses_a=15),
    ]
    est = fit_bradley_terry_elo(
        player_ids=['bot:random', 'p1', 'p_even', 'p_weak'],
        matches=matches,
        anchor_id='bot:random',
        anchor_elo=0.0,
        prior_draws=0.0,
        l2_reg=0.0,
    )
    assert est['bot:random'].elo == pytest.approx(0.0, abs=1e-9)
    assert est['bot:random'].elo_std == pytest.approx(0.0, abs=1e-9)
    assert est['p1'].elo == pytest.approx(expected_delta, abs=1e-5)
    assert est['p_even'].elo == pytest.approx(0.0, abs=1e-5)
    assert est['p_weak'].elo == pytest.approx(-expected_delta, abs=1e-5)
    assert expected_score(est['p1'].elo, est['bot:random'].elo) == pytest.approx(0.75, abs=1e-6)


@pytest.mark.unittest
def test_bradley_terry_transitive_ladder_chain_without_shrinkage():
    # Chain: bot:random -> c1 -> c2 -> c3 -> c4, each beating previous 75% over 40 games.
    # Even though c4 never plays bot:random directly, Elo accumulates transitively to ~+763.4.
    step = 400.0 * math.log10(3.0)
    matches = [
        MatchRecord('c1', 'bot:random', wins_a=30, draws=0, losses_a=10),
        MatchRecord('c2', 'c1', wins_a=30, draws=0, losses_a=10),
        MatchRecord('c3', 'c2', wins_a=30, draws=0, losses_a=10),
        MatchRecord('c4', 'c3', wins_a=30, draws=0, losses_a=10),
    ]
    est = fit_bradley_terry_elo(
        player_ids=['bot:random', 'c1', 'c2', 'c3', 'c4'],
        matches=matches,
        anchor_id='bot:random',
        anchor_elo=0.0,
        prior_draws=0.1,
        l2_reg=1e-4,
    )
    for k, pid in enumerate(['c1', 'c2', 'c3', 'c4'], start=1):
        assert est[pid].elo == pytest.approx(k * step, abs=8.0)
    assert est['c1'].elo < est['c2'].elo < est['c3'].elo < est['c4'].elo


@pytest.mark.unittest
def test_bradley_terry_sweep_stays_finite_and_std_shrinks_with_more_games():
    sweep = [MatchRecord('strong', 'bot:random', wins_a=10, draws=0, losses_a=0)]
    est_sweep = fit_bradley_terry_elo(
        player_ids=['bot:random', 'strong'],
        matches=sweep,
        anchor_id='bot:random',
    )
    assert math.isfinite(est_sweep['strong'].elo)
    assert est_sweep['strong'].elo > 400.0
    assert math.isfinite(est_sweep['strong'].elo_std)

    stds = []
    for scale in (1, 10, 100):
        m = [MatchRecord('p', 'bot:random', wins_a=7 * scale, draws=0, losses_a=3 * scale)]
        e = fit_bradley_terry_elo(['bot:random', 'p'], m, anchor_id='bot:random')
        stds.append(e['p'].elo_std)
    assert stds[0] > stds[1] > stds[2] > 0.0


@pytest.mark.unittest
def test_bradley_terry_symmetry_and_disconnected_player():
    m1 = [MatchRecord('a', 'b', wins_a=7, draws=2, losses_a=1)]
    m2 = [MatchRecord('b', 'a', wins_a=1, draws=2, losses_a=7)]
    e1 = fit_bradley_terry_elo(['b', 'a', 'lonely'], m1, anchor_id='b')
    e2 = fit_bradley_terry_elo(['b', 'a', 'lonely'], m2, anchor_id='b')

    assert e1['a'].elo == pytest.approx(e2['a'].elo, abs=1e-6)
    assert e1['a'].elo_std == pytest.approx(e2['a'].elo_std, abs=1e-6)
    assert math.isfinite(e1['lonely'].elo)
    assert e1['lonely'].elo == pytest.approx(0.0, abs=1e-6)
    assert e1['lonely'].games == 0


@pytest.mark.unittest
def test_ladder_opponent_selection_handles_regression_and_anchor():
    ladder = EloLadder(anchor_bots=('random',), max_ladder_size=10, opponents_per_eval=3)
    # Build a ladder with rungs at ~0, 200, 400, 600, 800, 1000 Elo
    prev = 'bot:random'
    for idx in range(1, 6):
        pid = f'iter_{idx * 1000}'
        ladder.register_player(PlayerEntry(player_id=pid, kind='checkpoint', active=True, train_iter=idx * 1000))
        ladder.record_match(pid, prev, wins_a=15, draws=0, losses_a=5)
        prev = pid
    ladder.fit_ratings()

    # Case 1: candidate was near champion (~950 Elo) -> should pick champion + top neighbors
    ladder.last_candidate_elo = ladder.get_elo('iter_5000')
    opps_top = [p.player_id for p in ladder.select_opponents()]
    assert len(opps_top) == 3
    assert 'iter_5000' in opps_top
    assert 'iter_4000' in opps_top

    # Case 2: candidate regressed down to ~200 Elo -> must include champion AND rungs near 200 Elo
    ladder.last_candidate_elo = ladder.get_elo('iter_1000')
    opps_regressed = [p.player_id for p in ladder.select_opponents()]
    assert len(opps_regressed) == 3
    assert 'iter_5000' in opps_regressed  # champion always tested
    assert 'iter_1000' in opps_regressed  # nearest rung to regressed candidate
    assert 'bot:random' in opps_regressed  # anchor included within 600 Elo


@pytest.mark.unittest
def test_ladder_hybrid_promotion_and_crowded_middle_pruning():
    ladder = EloLadder(
        anchor_bots=('random',),
        max_ladder_size=4,
        promotion_win_rate=0.55,
        promotion_elo_margin=50.0,
    )

    # First checkpoint always promotes as seed rung even if weak
    ladder.register_player(PlayerEntry(player_id='iter_0', kind='checkpoint', active=False, train_iter=0))
    ladder.record_match('iter_0', 'bot:random', wins_a=5, draws=0, losses_a=5)
    ladder.fit_ratings()
    promote, reason = ladder.should_promote('iter_0', score_vs_champion=None, champion_id=None)
    assert promote is True
    assert reason == 'first_checkpoint_seed'
    ladder.players['iter_0'].active = True

    # Weak candidate losing to iter_0 does NOT promote
    ladder.register_player(PlayerEntry(player_id='iter_500', kind='checkpoint', active=False, train_iter=500))
    ladder.record_match('iter_500', 'iter_0', wins_a=3, draws=0, losses_a=7)
    ladder.fit_ratings()
    promote_weak, _ = ladder.should_promote('iter_500', score_vs_champion=0.3, champion_id='iter_0')
    assert promote_weak is False

    # Promote 4 more rungs creating a crowded cluster near iter_2000 / iter_2100
    # Rungs will be: iter_0 (~0), iter_2000 (~200), iter_2100 (~220), iter_4000 (~500), iter_6000 (~800)
    rungs_data = [
        ('iter_2000', 'iter_0', 15, 5),      # ~+190
        ('iter_2100', 'iter_2000', 11, 9),   # ~+225 (very close to iter_2000!)
        ('iter_4000', 'iter_2100', 17, 3),   # ~+520
        ('iter_6000', 'iter_4000', 17, 3),   # ~+820
    ]
    for pid, prev_id, w, l in rungs_data:
        champ_id_before = ladder.active_checkpoint_rungs()[-1].player_id
        ladder.register_player(PlayerEntry(player_id=pid, kind='checkpoint', active=False))
        ladder.record_match(pid, prev_id, wins_a=w, draws=0, losses_a=l)
        ladder.fit_ratings()
        should_p, _ = ladder.should_promote(pid, score_vs_champion=w / (w + l), champion_id=champ_id_before)
        assert should_p is True
        ladder.players[pid].active = True

    assert len(ladder.active_checkpoint_rungs()) == 5
    pruned = ladder.prune_if_needed()
    assert len(pruned) == 1
    # Either iter_2000 or iter_2100 (the crowded interior rung) must be pruned,
    # NEVER the weakest bridge (iter_0), strongest champion (iter_6000), or anchor (bot:random)
    assert pruned[0].player_id in {'iter_2000', 'iter_2100'}
    active_ids = {p.player_id for p in ladder.active_rungs()}
    assert {'bot:random', 'iter_0', 'iter_6000'}.issubset(active_ids)
    assert len(ladder.active_checkpoint_rungs()) == 4

    # Historical matches of pruned rung remain intact so Elo ratings stay well-connected
    ladder.fit_ratings()
    assert ladder.get_elo('iter_6000') > ladder.get_elo('iter_4000') > ladder.get_elo('iter_0')


@pytest.mark.unittest
def test_ladder_json_roundtrip():
    ladder = EloLadder(anchor_bots=('random', 'pass'), max_ladder_size=6, opponents_per_eval=3)
    ladder.register_player(PlayerEntry(player_id='iter_100', kind='checkpoint', ckpt_path='/tmp/a.pth.tar', train_iter=100))
    ladder.record_match('iter_100', 'bot:random', wins_a=8, draws=1, losses_a=1)
    ladder.record_match('bot:random', 'bot:pass', wins_a=9, draws=0, losses_a=1)
    ladder.fit_ratings()
    ladder.best_elo = ladder.get_elo('iter_100')
    ladder.best_player_id = 'iter_100'
    ladder.last_candidate_elo = ladder.best_elo

    data = ladder.to_dict()
    restored = EloLadder.from_dict(data)

    assert restored.anchor_id == 'bot:random'
    assert restored.best_player_id == 'iter_100'
    assert restored.best_elo == pytest.approx(ladder.best_elo, abs=1e-6)
    assert restored.get_elo('iter_100') == pytest.approx(ladder.get_elo('iter_100'), abs=1e-6)
    assert restored.get_elo('bot:pass') == pytest.approx(ladder.get_elo('bot:pass'), abs=1e-6)
    assert 'iter_100' in restored.format_leaderboard()
    # Must serialize as strict RFC 8259 JSON (no NaN / Infinity tokens)
    json.dumps(data, allow_nan=False)


@pytest.mark.unittest
def test_retroactive_elo_revision_does_not_block_promotion_or_best_elo():
    # Regression test: an early 10-0 sweep inflates iter_0 to ~+919 Elo. When iter_500
    # later beats iter_0 8-2 and bot:random 8-2, joint MLE revises iter_0 down to ~+322
    # and rates iter_500 at ~+442. Both should_promote() AND best_elo must compare on
    # the post-refit scale so iter_500 promotes and becomes the new best checkpoint.
    ladder = EloLadder(anchor_bots=('random',), promotion_win_rate=0.55, promotion_elo_margin=50.0)

    ladder.register_player(PlayerEntry(player_id='iter_0', kind='checkpoint', active=True, train_iter=0))
    ladder.record_match('iter_0', 'bot:random', wins_a=10, draws=0, losses_a=0)
    ladder.fit_ratings()
    ladder.best_player_id = 'iter_0'
    ladder.best_elo = ladder.get_elo('iter_0')
    assert ladder.best_elo > 800.0

    ladder.register_player(PlayerEntry(player_id='iter_500', kind='checkpoint', active=False, train_iter=500))
    ladder.record_match('iter_500', 'iter_0', wins_a=8, draws=0, losses_a=2)
    ladder.record_match('iter_500', 'bot:random', wins_a=8, draws=0, losses_a=2)
    ladder.fit_ratings()

    assert ladder.best_elo == pytest.approx(ladder.get_elo('iter_0'), abs=1e-6)
    assert ladder.get_elo('iter_500') > ladder.best_elo
    promote, _ = ladder.should_promote('iter_500', score_vs_champion=0.8, champion_id='iter_0')
    assert promote is True


@pytest.mark.unittest
def test_bradley_terry_unregularized_disconnected_has_infinite_std_and_strict_json():
    est = fit_bradley_terry_elo(
        player_ids=['bot:random', 'unplayed', 'island_a', 'island_b'],
        matches=[MatchRecord('island_a', 'island_b', wins_a=7, draws=0, losses_a=3)],
        anchor_id='bot:random',
        prior_draws=0.0,
        l2_reg=0.0,
    )
    assert est['bot:random'].elo_std == 0.0
    assert math.isinf(est['unplayed'].elo_std)
    assert math.isinf(est['island_a'].elo_std)
    assert math.isinf(est['island_b'].elo_std)
    # Serializing estimates with infinite std must produce strict JSON (None, not Infinity)
    json.dumps({k: v.to_dict() for k, v in est.items()}, allow_nan=False)


@pytest.mark.unittest
def test_select_opponents_includes_checkpoint_champion_even_when_bot_is_stronger():
    ladder = EloLadder(anchor_bots=('random', 'fixed'), max_ladder_size=10, opponents_per_eval=2)
    for pid, w, l in [('iter_0', 1, 9), ('iter_100', 2, 8), ('iter_200', 4, 6)]:
        ladder.register_player(PlayerEntry(player_id=pid, kind='checkpoint', active=True))
        ladder.record_match(pid, 'bot:random', wins_a=w, draws=0, losses_a=l)
    ladder.record_match('bot:fixed', 'bot:random', wins_a=6, draws=0, losses_a=4)
    ladder.fit_ratings()

    # Bots outrank all checkpoints, yet select_opponents must still include checkpoint champion iter_200
    ladder.last_candidate_elo = ladder.get_elo('bot:fixed')
    opps = [p.player_id for p in ladder.select_opponents()]
    assert 'iter_200' in opps


@pytest.mark.unittest
def test_prune_if_needed_protects_newly_promoted_rung():
    ladder = EloLadder(anchor_bots=('random',), max_ladder_size=3)
    for pid, prev, w, l in [
        ('iter_0', 'bot:random', 5, 5),
        ('iter_100', 'iter_0', 12, 8),
        ('iter_200', 'iter_100', 11, 9),
        ('iter_300', 'iter_200', 18, 2),
    ]:
        ladder.register_player(PlayerEntry(player_id=pid, kind='checkpoint', active=True))
        ladder.record_match(pid, prev, wins_a=w, draws=0, losses_a=l)
    ladder.fit_ratings()

    pruned = ladder.prune_if_needed(protected_ids={'iter_200'})
    assert len(pruned) == 1
    assert pruned[0].player_id == 'iter_100'
    assert ladder.players['iter_200'].active is True


@pytest.mark.unittest
def test_should_promote_checks_true_post_refit_champion_not_stale_pre_refit_id():
    # Regression test: if champion_id_before (iter_200) drops below another active rung (iter_100)
    # after fit_ratings(), a candidate that beats iter_200 6-4 but gets crushed 0-10 by iter_100
    # must NOT be promoted because it is well below the true post-refit champion (iter_100).
    ladder = EloLadder(anchor_bots=('random',), promotion_win_rate=0.55, promotion_elo_margin=50.0)
    for pid in ('iter_100', 'iter_200'):
        ladder.register_player(PlayerEntry(player_id=pid, kind='checkpoint', active=True))
    ladder.record_match('iter_100', 'bot:random', wins_a=8, draws=0, losses_a=2)
    ladder.record_match('iter_200', 'bot:random', wins_a=8, draws=0, losses_a=2)
    ladder.record_match('iter_200', 'iter_100', wins_a=6, draws=0, losses_a=4)
    ladder.fit_ratings()
    assert ladder.get_champion().player_id == 'iter_200'

    # Candidate beats iter_200 9-1 and bot:random 8-2, but loses 1-9 to iter_100 ->
    # iter_100 overtakes iter_200 as true post-refit champion while iter_300 sits between them
    ladder.register_player(PlayerEntry(player_id='iter_300', kind='checkpoint', active=False))
    ladder.record_match('iter_300', 'iter_200', wins_a=9, draws=0, losses_a=1)
    ladder.record_match('iter_300', 'iter_100', wins_a=1, draws=0, losses_a=9)
    ladder.record_match('iter_300', 'bot:random', wins_a=8, draws=0, losses_a=2)
    ladder.fit_ratings()

    assert ladder.get_elo('iter_100') > ladder.get_elo('iter_300') > ladder.get_elo('iter_200')
    promote, _ = ladder.should_promote('iter_300', score_vs_champion=0.9, champion_id='iter_200')
    assert promote is False


@pytest.mark.unittest
def test_unregularized_sweep_preserves_finite_std_for_non_sweeping_players():
    # Even with prior_draws=0.0, l2_reg=0.0 and an extreme sweep alongside a normal player,
    # the normal player must retain an accurate Elo and finite standard error.
    matches = [
        MatchRecord('normal', 'bot:random', wins_a=7, draws=0, losses_a=3),
        MatchRecord('sweeper', 'bot:random', wins_a=50, draws=0, losses_a=0),
    ]
    est = fit_bradley_terry_elo(
        player_ids=['bot:random', 'normal', 'sweeper'],
        matches=matches,
        anchor_id='bot:random',
        prior_draws=0.0,
        l2_reg=0.0,
    )
    assert est['normal'].elo == pytest.approx(400.0 * math.log10(7.0 / 3.0), abs=1e-3)
    assert math.isfinite(est['normal'].elo_std)
    assert 50.0 < est['normal'].elo_std < 300.0


