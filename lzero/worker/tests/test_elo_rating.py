# -*- coding: utf-8 -*-
"""Smoke tests for the ``lzero.worker.elo_rating`` re-export shim.

Full Bradley-Terry MLE and ``EloLadder`` unit tests live in
``galcon-bot-arena/tests/test_elo_rating.py``.
"""

import math

import pytest

from lzero.worker.elo_rating import (
    EloEstimate,
    EloLadder,
    MatchRecord,
    PlayerEntry,
    expected_score,
    fit_bradley_terry_elo,
)


@pytest.mark.unittest
def test_elo_rating_reexports_and_basic_fit():
    est = fit_bradley_terry_elo(
        player_ids=['bot:random', 'p1'],
        matches=[MatchRecord(player_a='p1', player_b='bot:random', wins_a=15, draws=0, losses_a=5)],
        anchor_id='bot:random',
        anchor_elo=0.0,
        prior_draws=0.0,
        l2_reg=0.0,
    )
    assert isinstance(est['p1'], EloEstimate)
    assert est['bot:random'].elo == pytest.approx(0.0, abs=1e-9)
    assert est['p1'].elo == pytest.approx(400.0 * math.log10(3.0), abs=1e-5)
    assert expected_score(est['p1'].elo, est['bot:random'].elo) == pytest.approx(0.75, abs=1e-6)

    ladder = EloLadder(anchor_bots=('random',))
    ladder.register_player(PlayerEntry(player_id='iter_1', kind='checkpoint', train_iter=1))
    ladder.record_match('iter_1', 'bot:random', wins_a=8, draws=0, losses_a=2)
    ladder.fit_ratings()
    assert ladder.get_elo('iter_1') > 0.0
