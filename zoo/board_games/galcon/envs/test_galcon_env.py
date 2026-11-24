import numpy as np
import pytest
from easydict import EasyDict
from gymnasium import spaces
from lzero.mcts.buffer.game_segment import GameSegment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from zoo.board_games.galcon.envs.galcon_env import GalconEnv, Fleet, Planet

# Run all tests: 
# pytest zoo/board_games/galcon/envs/

@pytest.mark.envtest
class TestGalconEnv:

    def setup_method(self) -> None:
        self.cfg = GalconEnv.default_config() 
        self.cfg.update(dict(
            num_planets=8,
            map_seed=0,
            grid_square_size=25.0,
            grid_max_x=200.0,
            grid_max_y=125.0,
            fleet_top_k=3,
        ))

    def test_reset(self) -> None:
        env = GalconEnv(self.cfg)
        obs = env.reset()

        assert isinstance(obs, dict)
        assert 'observation' in obs
        assert 'action_mask' in obs
        assert 'to_play' in obs
        assert obs['observation'].shape == (env.observation_channel_count, env.grid_height, env.grid_width)
        assert obs['action_mask'].shape == (env.grid_width * env.grid_height * env.grid_width * env.grid_height + 1,)

    def test_action_space(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 16 * 10 * 16 * 10 + 1

    def test_random_action_is_legal(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        action = env.random_action()
        assert action in env.legal_actions

    def test_pass_action(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        assert env.pass_action == 0
        assert env.pass_action in env.legal_actions
        assert env.decode_action(env.pass_action) == (None, None, None, None)
        assert env.action_to_string(env.pass_action) == 'Pass'

    def test_grid_action_encode_decode_round_trip(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        action = env.encode_action(source_x=1, source_y=2, target_x=3, target_y=4)

        assert env.decode_action(action) == (1, 2, 3, 4)

    def test_decode_action_one_returns_first_grid_to_first_grid(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        assert env.decode_action(1) == (0, 0, 0, 0)

    def test_send_action_creates_fleet(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        source_grid_x, source_grid_y = env._world_to_grid(env.planets[0].x, env.planets[0].y)
        target_grid_x, target_grid_y = env._world_to_grid(env.planets[1].x, env.planets[1].y)
        action = env.encode_action(source_grid_x, source_grid_y, target_grid_x, target_grid_y)
        source_ships_before = env.planets[0].ships

        assert action in env.legal_actions

        obs, reward, done, info = env.step(action)

        # The ships have changed due to production in the env.step
        expected_produced_ships = (env.planets[0].production /
                                   env.PRODUCTION_TO_SHIPS_PER_SECOND_DIVISOR * env.tick_seconds)

        assert len(env.fleets) == 1
        assert env.planets[0].ships == source_ships_before * (1 - env.send_ratio) + expected_produced_ships
        assert env.fleets[0].owner == GalconEnv.PLAYER_1
        assert env.fleets[0].source == 0
        assert env.fleets[0].target == 1

    def test_production_adds_ships(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        before = env.planets[0].ships
        env.step(env.pass_action)
        after = env.planets[0].ships

        expected_delta = env.planets[0].production / env.PRODUCTION_TO_SHIPS_PER_SECOND_DIVISOR * env.tick_seconds
        assert np.isclose(after - before, expected_delta)

    def test_seeded_map_is_deterministic(self) -> None:
        env1 = GalconEnv(self.cfg)
        env2 = GalconEnv(self.cfg)

        env1.reset()
        env2.reset()

        planets1 = [(p.x, p.y, p.ships, p.production) for p in env1.planets]
        planets2 = [(p.x, p.y, p.ships, p.production) for p in env2.planets]

        assert planets1 == planets2

    def test_timeout_winner_by_production(self) -> None:
        cfg = self.cfg
        cfg.max_episode_steps = 1
        env = GalconEnv(cfg)
        env.reset()

        # Give player 1 a single planet
        env.planets[2].owner = GalconEnv.PLAYER_1
        env.planets[2].neutral = False

        obs, reward, done, info = env.step(env.pass_action)

        assert done
        assert info['winner'] == GalconEnv.PLAYER_1
        assert reward == np.array(1).astype(np.float32)

    def test_self_play_rollout(self) -> None:
        env = GalconEnv(self.cfg)
        obs = env.reset()

        done = False
        step = 0
        while not done and step < self.cfg.max_episode_steps + 5:
            action = env.random_action()
            obs, reward, done, info = env.step(action)
            step += 1

        assert done
        assert step <= self.cfg.max_episode_steps + 5

    def test_planets_do_not_overlap(self) -> None:
        env = GalconEnv(self.cfg)
        for seed in range(10):
            env.cfg.map_seed = seed
            env.reset()

            SHIP_RADIUS = 6.0
            PLANETS_SETTLE_DELTA = 0.5
            min_gap = 2 * SHIP_RADIUS + PLANETS_SETTLE_DELTA

            for i in range(len(env.planets)):
                p_i = env.planets[i]
                for j in range(i + 1, len(env.planets)):
                    p_j = env.planets[j]
                    dist = np.hypot(p_i.x - p_j.x, p_i.y - p_j.y)
                    min_dist = p_i.radius + p_j.radius + min_gap
                    assert dist >= min_dist - 1e-5, f"Planets {i} and {j} overlap or are too close! dist={dist}, min_dist={min_dist}"

    def _create_dummy_symmetry_segment(self, enable_fleet_redirection: bool = False):
        # fleet_top_k=1 -> C = 8 (planets) + 2*(1+1)*5 (fleets) + len(LANDING_BUCKETS)*4 (landings) = 40
        C = (
            GalconEnv.PLANET_CHANNEL_COUNT
            + 2 * (1 + 1) * GalconEnv.FLEET_FEATURE_COUNT
            + len(GalconEnv.LANDING_BUCKETS) * GalconEnv.LANDING_FEATURE_COUNT_PER_BUCKET
        )
        pair_action_count = 4 * 4 * 4 * 4
        action_space_size = (2 * pair_action_count + 1) if enable_fleet_redirection else (pair_action_count + 1)
        game_segment_config = EasyDict(dict(
            num_unroll_steps=5,
            td_steps=5,
            discount_factor=1.0,
            gray_scale=False,
            transform2string=False,
            sampled_algo=True,
            gumbel_algo=False,
            use_ture_chance_label_in_chance_encoder=False,
            model=dict(
                frame_stack_num=1,
                observation_shape=(C, 4, 4),
                image_channel=C,
                action_space_size=action_space_size,
            ),
        ))
        segment = GameSegment(action_space=None, game_segment_length=20, config=game_segment_config)
        obs = np.zeros((C, 4, 4), dtype=np.float32)
        # Planet in cell (x=1, y=2)
        obs[0, 2, 1] = 0.2  # local_x
        obs[1, 2, 1] = 0.3  # local_y
        obs[2, 2, 1] = 0.5  # friendly ships
        obs[5, 2, 1] = 0.5  # friendly production

        # Friendly fleet in cell (x=0, y=1), slot 0 (base_ch=8), target normalized to (0.1, 0.9)
        obs[8, 1, 0] = 0.1  # target x
        obs[9, 1, 0] = 0.9  # target y
        obs[11, 1, 0] = 10  # fleet ships

        # Enemy fleet in cell (x=2, y=0), slot 0 (base_ch=8 + 2*5 = 18), target normalized to (0.25, 0.75)
        obs[18, 0, 2] = 0.25  # target x
        obs[19, 0, 2] = 0.75  # target y
        obs[21, 0, 2] = 5     # fleet ships
        segment.obs_segment = [obs]

        # Action: source (0, 1) -> cell 1*4+0 = 4; target (1, 2) -> cell 2*4+1 = 9
        # send action = 4 * 16 + 9 + 1 = 74; redirect action = 74 + 256 = 330
        action = 74 + (pair_action_count if enable_fleet_redirection else 0)
        segment.action_segment = [action]
        action_mask = np.zeros(action_space_size)
        action_mask[0] = 1
        action_mask[action] = 1
        segment.action_mask_segment = [action_mask]
        segment.root_sampled_actions = [np.array([0, action])]
        return segment

    def test_reflect_game_segment_x_flip(self) -> None:
        segment = self._create_dummy_symmetry_segment()
        reflected = GalconEnv.reflect_game_segment(segment, reflect_x=True, reflect_y=False)

        assert reflected.obs_segment.shape == (1, 40, 4, 4)
        ref_obs = reflected.obs_segment[0]
        # Planet (x=1, y=2) -> X-flip: x' = 4-1-1 = 2, y' = 2
        assert ref_obs[2, 2, 2] == 0.5
        assert ref_obs[5, 2, 2] == 0.5
        assert np.isclose(ref_obs[0, 2, 2], 1.0 - 0.2)  # local_x inverted
        assert np.isclose(ref_obs[1, 2, 2], 0.3)        # local_y unchanged

        # Friendly fleet (x=0, y=1) -> X-flip: x' = 4-1-0 = 3, y' = 1
        assert ref_obs[11, 1, 3] == 10
        assert np.isclose(ref_obs[8, 1, 3], 1.0 - 0.1)  # target_x inverted
        assert np.isclose(ref_obs[9, 1, 3], 0.9)        # target_y unchanged

        # Enemy fleet (x=2, y=0) -> X-flip: x' = 4-1-2 = 1, y' = 0
        assert ref_obs[21, 0, 1] == 5
        assert np.isclose(ref_obs[18, 0, 1], 1.0 - 0.25)  # target_x inverted
        assert np.isclose(ref_obs[19, 0, 1], 0.75)        # target_y unchanged

        # Action: source (3, 1) -> cell 1*4+3 = 7; target (2, 2) -> cell 2*4+2 = 10
        # action = 7 * 16 + 10 + 1 = 123
        assert reflected.action_segment[0] == 123
        assert reflected.action_mask_segment[0][123] == 1
        assert reflected.action_mask_segment[0][0] == 1
        assert reflected.root_sampled_actions[0][0] == 0
        assert reflected.root_sampled_actions[0][1] == 123

    def test_reflect_game_segment_y_flip(self) -> None:
        segment = self._create_dummy_symmetry_segment()
        reflected = GalconEnv.reflect_game_segment(segment, reflect_x=False, reflect_y=True)

        assert reflected.obs_segment.shape == (1, 40, 4, 4)
        ref_obs = reflected.obs_segment[0]
        # Planet (x=1, y=2) -> Y-flip: x' = 1, y' = 4-1-2 = 1
        assert ref_obs[2, 1, 1] == 0.5
        assert ref_obs[5, 1, 1] == 0.5
        assert np.isclose(ref_obs[0, 1, 1], 0.2)        # local_x unchanged
        assert np.isclose(ref_obs[1, 1, 1], 1.0 - 0.3)  # local_y inverted

        # Friendly fleet (x=0, y=1) -> Y-flip: x' = 0, y' = 4-1-1 = 2
        assert ref_obs[11, 2, 0] == 10
        assert np.isclose(ref_obs[8, 2, 0], 0.1)        # target_x unchanged
        assert np.isclose(ref_obs[9, 2, 0], 1.0 - 0.9)  # target_y inverted

        # Enemy fleet (x=2, y=0) -> Y-flip: x' = 2, y' = 4-1-0 = 3
        assert ref_obs[21, 3, 2] == 5
        assert np.isclose(ref_obs[18, 3, 2], 0.25)        # target_x unchanged
        assert np.isclose(ref_obs[19, 3, 2], 1.0 - 0.75)  # target_y inverted

        # Action: source (0, 2) -> cell 2*4+0 = 8; target (1, 1) -> cell 1*4+1 = 5
        # action = 8 * 16 + 5 + 1 = 134
        assert reflected.action_segment[0] == 134
        assert reflected.action_mask_segment[0][134] == 1
        assert reflected.action_mask_segment[0][0] == 1
        assert reflected.root_sampled_actions[0][0] == 0
        assert reflected.root_sampled_actions[0][1] == 134

    def test_reflect_game_segment_xy_flip(self) -> None:
        segment = self._create_dummy_symmetry_segment()
        reflected = GalconEnv.reflect_game_segment(segment, reflect_x=True, reflect_y=True)

        assert reflected.obs_segment.shape == (1, 40, 4, 4)
        ref_obs = reflected.obs_segment[0]
        # Planet (x=1, y=2) -> XY-flip: x' = 2, y' = 1
        assert ref_obs[2, 1, 2] == 0.5
        assert ref_obs[5, 1, 2] == 0.5
        assert np.isclose(ref_obs[0, 1, 2], 1.0 - 0.2)  # local_x inverted
        assert np.isclose(ref_obs[1, 1, 2], 1.0 - 0.3)  # local_y inverted

        # Friendly fleet (x=0, y=1) -> XY-flip: x' = 3, y' = 2
        assert ref_obs[11, 2, 3] == 10
        assert np.isclose(ref_obs[8, 2, 3], 1.0 - 0.1)  # target_x inverted
        assert np.isclose(ref_obs[9, 2, 3], 1.0 - 0.9)  # target_y inverted

        # Enemy fleet (x=2, y=0) -> XY-flip: x' = 1, y' = 3
        assert ref_obs[21, 3, 1] == 5
        assert np.isclose(ref_obs[18, 3, 1], 1.0 - 0.25)  # target_x inverted
        assert np.isclose(ref_obs[19, 3, 1], 1.0 - 0.75)  # target_y inverted

        # Action: source (3, 2) -> cell 2*4+3 = 11; target (2, 1) -> cell 1*4+2 = 6
        # action = 11 * 16 + 6 + 1 = 183
        assert reflected.action_segment[0] == 183
        assert reflected.action_mask_segment[0][183] == 1
        assert reflected.action_mask_segment[0][0] == 1
        assert reflected.root_sampled_actions[0][0] == 0
        assert reflected.root_sampled_actions[0][1] == 183

    def test_reflect_game_segment_redirect_action_layer(self) -> None:
        segment = self._create_dummy_symmetry_segment(enable_fleet_redirection=True)
        reflected = GalconEnv.reflect_game_segment(segment, reflect_x=True, reflect_y=True)

        # Base XY-reflected send action is 183 -> redirect layer adds pair_action_count (256)
        expected_redirect_action = 183 + 256
        assert reflected.action_segment[0] == expected_redirect_action
        assert reflected.action_mask_segment[0][expected_redirect_action] == 1
        assert reflected.action_mask_segment[0][0] == 1
        assert reflected.root_sampled_actions[0][1] == expected_redirect_action

    def test_friendly_fleet_landing(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        target_planet = env.planets[0]  # Player 1 home
        target_planet.owner = GalconEnv.PLAYER_1
        target_planet.neutral = False
        target_planet.production = 0.0  # Zero production to isolate landing math
        target_planet.ships = 50.0

        # Place fleet right at target edge so it lands in 1 tick
        fleet = Fleet(
            id=100,
            owner=GalconEnv.PLAYER_1,
            ships=25.0,
            x=target_planet.x,
            y=target_planet.y + target_planet.radius + 1.0,
            source=0,
            target=0,
            radius=env.DUMMY_FLEET_RADIUS,
        )
        env.fleets.append(fleet)

        env._advance_one_tick()

        assert len(env.fleets) == 0
        assert np.isclose(target_planet.ships, 75.0)
        assert target_planet.owner == GalconEnv.PLAYER_1

    def test_hostile_fleet_defense(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        target_planet = env.planets[0]  # Player 1 home
        target_planet.owner = GalconEnv.PLAYER_1
        target_planet.neutral = False
        target_planet.production = 0.0
        target_planet.ships = 50.0

        # Hostile fleet with fewer ships (20 vs 50)
        fleet = Fleet(
            id=101,
            owner=GalconEnv.PLAYER_2,
            ships=20.0,
            x=target_planet.x,
            y=target_planet.y + target_planet.radius + 1.0,
            source=1,
            target=0,
            radius=env.DUMMY_FLEET_RADIUS,
        )
        env.fleets.append(fleet)

        env._advance_one_tick()

        assert len(env.fleets) == 0
        assert np.isclose(target_planet.ships, 30.0)
        assert target_planet.owner == GalconEnv.PLAYER_1

    def test_hostile_fleet_capture_neutral(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        neutral_planet = env.planets[2]  # Neutral planet
        neutral_planet.owner = GalconEnv.PLAYER_NEUTRAL
        neutral_planet.neutral = True
        neutral_planet.production = 50.0
        neutral_planet.ships = 15.0

        # Player 1 fleet with 40 ships attacking neutral with 15 ships
        fleet = Fleet(
            id=102,
            owner=GalconEnv.PLAYER_1,
            ships=40.0,
            x=neutral_planet.x,
            y=neutral_planet.y + neutral_planet.radius + 1.0,
            source=0,
            target=2,
            radius=env.DUMMY_FLEET_RADIUS,
        )
        env.fleets.append(fleet)

        env._advance_one_tick()

        assert len(env.fleets) == 0
        assert np.isclose(neutral_planet.ships, 25.0)
        assert neutral_planet.owner == GalconEnv.PLAYER_1
        assert neutral_planet.neutral is False

    def test_hostile_fleet_capture_enemy(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        enemy_planet = env.planets[1]  # Player 2 home
        enemy_planet.owner = GalconEnv.PLAYER_2
        enemy_planet.neutral = False
        enemy_planet.production = 0.0
        enemy_planet.ships = 10.0

        # Player 1 fleet with 35 ships
        fleet = Fleet(
            id=103,
            owner=GalconEnv.PLAYER_1,
            ships=35.0,
            x=enemy_planet.x,
            y=enemy_planet.y + enemy_planet.radius + 1.0,
            source=0,
            target=1,
            radius=env.DUMMY_FLEET_RADIUS,
        )
        env.fleets.append(fleet)

        env._advance_one_tick()

        assert len(env.fleets) == 0
        assert np.isclose(enemy_planet.ships, 25.0)
        assert enemy_planet.owner == GalconEnv.PLAYER_1

    def test_elimination_condition(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        # Turn all planets into Player 1 planets
        for p in env.planets:
            p.owner = GalconEnv.PLAYER_1
            p.neutral = False
        env.fleets.clear()

        done, winner = env.get_done_winner()
        assert done is True
        assert winner == GalconEnv.PLAYER_1

    def test_elimination_delayed_by_fleet_in_flight(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        # Player 2 has 0 planets
        for p in env.planets:
            p.owner = GalconEnv.PLAYER_1
            p.neutral = False
        env.fleets.clear()

        # BUT Player 2 has a fleet in transit
        fleet = Fleet(
            id=104,
            owner=GalconEnv.PLAYER_2,
            ships=50.0,
            x=0.0,
            y=0.0,
            source=1,
            target=0,
            radius=env.DUMMY_FLEET_RADIUS,
        )
        env.fleets.append(fleet)

        done, winner = env.get_done_winner()
        assert done is False
        assert winner == -1

    def test_timeout_tiebreakers(self) -> None:
        cfg = self.cfg
        cfg.max_episode_steps = 10
        env = GalconEnv(cfg)
        env.reset()
        env._step_count = 10  # Trigger timeout check

        # Case 1: Production tiebreaker
        for p in env.planets:
            p.owner = GalconEnv.PLAYER_NEUTRAL
        env.planets[0].owner = GalconEnv.PLAYER_1
        env.planets[0].production = 100.0
        env.planets[0].ships = 10.0

        env.planets[1].owner = GalconEnv.PLAYER_2
        env.planets[1].production = 50.0
        env.planets[1].ships = 500.0  # Even though P2 has way more ships, production wins

        done, winner = env.get_done_winner()
        assert done is True
        assert winner == GalconEnv.PLAYER_1

        # Case 2: Ship tiebreaker (when production is equal)
        env.planets[1].production = 100.0
        done, winner = env.get_done_winner()
        assert done is True
        assert winner == GalconEnv.PLAYER_2

        # Case 3: Complete draw
        env.planets[0].ships = 500.0
        done, winner = env.get_done_winner()
        assert done is True
        assert winner == -1

    def test_action_masking_rules(self) -> None:
        env = GalconEnv(self.cfg)
        env.reset()

        # Pass action (0) must always be legal
        assert 0 in env.legal_actions
        assert env.observe()['action_mask'][0] == 1

        # Player 1 home planet
        p1 = env.planets[0]
        p1_gx, p1_gy = env._world_to_grid(p1.x, p1.y)

        # Player 2 home planet
        p2 = env.planets[1]
        p2_gx, p2_gy = env._world_to_grid(p2.x, p2.y)

        # Player 1 can send to Player 2
        p1_to_p2_action = env.encode_action(p1_gx, p1_gy, p2_gx, p2_gy)
        assert p1_to_p2_action in env.legal_actions
        assert env.observe()['action_mask'][p1_to_p2_action] == 1

        # Player 1 CANNOT send from Player 2's planet
        p2_to_p1_action = env.encode_action(p2_gx, p2_gy, p1_gx, p1_gy)
        assert p2_to_p1_action not in env.legal_actions
        assert env.observe()['action_mask'][p2_to_p1_action] == 0

        # If Player 1 planet has fewer ships than min_send_ships, it cannot send
        p1.ships = env.min_send_ships - 0.5
        env._legal_actions = None  # Invalidate cached legal actions
        assert p1_to_p2_action not in env.legal_actions
        assert env.observe()['action_mask'][p1_to_p2_action] == 0

    @pytest.mark.parametrize('battle_mode', ['eval_mode', 'play_with_bot_mode'])
    @pytest.mark.parametrize('max_steps,p1_ships,p2_ships,expected_winner,expected_reward', [
        (4, 1000.0, 1.0, 1, 1.0),   # Agent wins on EVEN step (bot's turn)
        (4, 1.0, 1000.0, 2, -1.0),  # Bot wins on EVEN step (bot's turn)
        (3, 1000.0, 1.0, 1, 1.0),   # Agent wins on ODD step (agent's turn)
        (3, 1.0, 1000.0, 2, -1.0),  # Bot wins on ODD step (agent's turn)
    ])
    def test_eval_and_bot_mode_reward_perspective_odd_and_even_steps(
        self, battle_mode, max_steps, p1_ships, p2_ships, expected_winner, expected_reward
    ):
        """In eval_mode and play_with_bot_mode, reward and eval_episode_return must always
        be from Player 1's (agent's) perspective, even when the game terminates on Player 2's
        (even) step such as max_episode_steps=100.
        """
        cfg = GalconEnv.default_config()
        cfg.update(dict(
            battle_mode=battle_mode,
            bot_action_type='pass',
            max_episode_steps=max_steps,
            num_planets=2,
            map_seed=2,
            grid_square_size=25.0,
            grid_max_x=50.0,
            grid_max_y=50.0,
        ))
        env = GalconEnv(cfg)
        env.reset()
        for p in env.planets:
            if p.owner == env.PLAYER_1:
                p.ships = p1_ships
            elif p.owner == env.PLAYER_2:
                p.ships = p2_ships

        while True:
            ts = env.step(env.pass_action)
            if ts.done:
                assert ts.info['step_count'] == max_steps
                assert ts.info['winner'] == expected_winner
                assert float(ts.reward) == pytest.approx(expected_reward)
                assert float(ts.info['eval_episode_return']) == pytest.approx(expected_reward)
                break

    @pytest.mark.parametrize('replay_format', ['gif', 'mp4'])
    def test_save_render_output_gif_and_mp4(self, tmp_path, replay_format):
        """Verify image_savefile_mode produces a non-empty replay file for both GIF and H.264 MP4."""
        cfg = GalconEnv.default_config()
        cfg.update(dict(
            battle_mode='eval_mode',
            bot_action_type='pass',
            max_episode_steps=4,
            num_planets=4,
            map_seed=2,
            grid_square_size=25.0,
            grid_max_x=50.0,
            grid_max_y=50.0,
            render_mode='image_savefile_mode',
            replay_path=str(tmp_path),
            replay_format=replay_format,
            replay_name_suffix=f'unit_{replay_format}',
        ))
        env = GalconEnv(cfg)
        env.reset()
        while True:
            ts = env.step(env.pass_action)
            if ts.done:
                break
        files = list(tmp_path.glob(f'*.{replay_format}'))
        assert len(files) == 1
        assert files[0].stat().st_size > 0

    def _make_redirect_env(self, enable_fleet_redirection: bool) -> GalconEnv:
        cfg = GalconEnv.default_config()
        cfg.update(dict(
            battle_mode='self_play_mode',
            num_planets=4,
            map_seed=2,
            grid_square_size=25.0,
            grid_max_x=50.0,
            grid_max_y=50.0,
            max_episode_steps=20,
            enable_fleet_redirection=enable_fleet_redirection,
        ))
        env = GalconEnv(cfg)
        env.reset()
        return env

    def _find_empty_cell(self, env: GalconEnv):
        occupied = {env._world_to_grid(p.x, p.y) for p in env.planets}
        for gy in range(env.grid_height):
            for gx in range(env.grid_width):
                if (gx, gy) not in occupied:
                    # Return (gx, gy, world_x, world_y) at center of cell
                    wx = -env.grid_max_x + (gx + 0.5) * env.grid_square_size
                    wy = -env.grid_max_y + (gy + 0.5) * env.grid_square_size
                    return gx, gy, wx, wy
        raise AssertionError("No empty cell found on 4x4 grid with 4 planets")

    def test_fleet_redirection_disabled_by_default(self):
        """By default (enable_fleet_redirection=False), action space is N+1 (no redirect layer),
        cells with only friendly fleets are NOT legal sources, and _apply_action does not redirect.
        """
        env = self._make_redirect_env(enable_fleet_redirection=False)
        assert env.enable_fleet_redirection is False
        assert env.total_num_actions == env.pair_action_count + 1

        gx, gy, wx, wy = self._find_empty_cell(env)
        p0, p1 = env.planets[0], env.planets[1]
        fleet = Fleet(id=99, owner=env.PLAYER_1, ships=10.0, x=wx, y=wy, source=p0.id, target=p0.id, radius=1.0)
        env.fleets.append(fleet)

        t_gx, t_gy = env._world_to_grid(p1.x, p1.y)
        send_act = env.encode_action(gx, gy, t_gx, t_gy, is_redirect=False)
        redirect_act = env.encode_action(gx, gy, t_gx, t_gy, is_redirect=True)
        assert send_act not in env.legal_actions
        assert redirect_act not in env.legal_actions
        assert env.observe()['action_mask'][send_act] == 0

        env._apply_action(send_act)
        env._apply_action(redirect_act)
        assert fleet.target == p0.id

    def test_fleet_redirection_legal_actions_and_masking(self):
        """With enable_fleet_redirection=True:
        - Action space is 2*N + 1 (separate planet-send and fleet-redirect layers).
        - Friendly fleet in empty cell unlocks redirect actions (is_redirect=True) to any
          planet != current target, while planet-send (is_redirect=False) stays masked out.
        - Redirecting to its already-current target is masked out (no-op).
        - Enemy fleet in empty cell does NOT unlock redirects for Player 1.
        """
        env = self._make_redirect_env(enable_fleet_redirection=True)
        assert env.total_num_actions == 2 * env.pair_action_count + 1

        gx, gy, wx, wy = self._find_empty_cell(env)
        p0, p1 = env.planets[0], env.planets[1]
        t0_gx, t0_gy = env._world_to_grid(p0.x, p0.y)
        t1_gx, t1_gy = env._world_to_grid(p1.x, p1.y)

        # Enemy fleet alone should not make cell legal for Player 1
        env.fleets.append(Fleet(id=10, owner=env.PLAYER_2, ships=10.0, x=wx, y=wy, source=p1.id, target=p0.id, radius=1.0))
        redirect_to_p1 = env.encode_action(gx, gy, t1_gx, t1_gy, is_redirect=True)
        assert redirect_to_p1 not in env.legal_actions

        # Add friendly fleet targeting p0 -> redirect to p1 is legal, send to p1 & redirect to p0 are NOT legal
        env.fleets.append(Fleet(id=11, owner=env.PLAYER_1, ships=10.0, x=wx, y=wy, source=p0.id, target=p0.id, radius=1.0))
        send_to_p1 = env.encode_action(gx, gy, t1_gx, t1_gy, is_redirect=False)
        redirect_to_p0 = env.encode_action(gx, gy, t0_gx, t0_gy, is_redirect=True)
        assert env.decode_action_with_type(redirect_to_p1) == ('redirect', gx, gy, t1_gx, t1_gy)
        assert env.decode_action_with_type(send_to_p1) == ('send', gx, gy, t1_gx, t1_gy)

        mask = env.observe()['action_mask']
        assert redirect_to_p1 in env.legal_actions and mask[redirect_to_p1] == 1
        assert send_to_p1 not in env.legal_actions and mask[send_to_p1] == 0
        assert redirect_to_p0 not in env.legal_actions and mask[redirect_to_p0] == 0

    def test_fleet_redirection_redirects_all_friendly_fleets_and_moves_toward_new_target(self):
        """Selecting a redirect action on a cell with multiple friendly fleets + 1 enemy fleet
        redirects ALL friendly fleets to the new target (leaving enemy fleet untouched), and on
        the next tick the redirected fleets move closer to the new target.
        """
        env = self._make_redirect_env(enable_fleet_redirection=True)
        gx, gy, wx, wy = self._find_empty_cell(env)
        p0, p1 = env.planets[0], env.planets[1]
        t1_gx, t1_gy = env._world_to_grid(p1.x, p1.y)

        f1 = Fleet(id=1, owner=env.PLAYER_1, ships=12.0, x=wx, y=wy, source=p0.id, target=p0.id, radius=1.0)
        f2 = Fleet(id=2, owner=env.PLAYER_1, ships=8.0, x=wx, y=wy, source=p0.id, target=p0.id, radius=1.0)
        f_enemy = Fleet(id=3, owner=env.PLAYER_2, ships=15.0, x=wx, y=wy, source=p1.id, target=p0.id, radius=1.0)
        env.fleets.extend([f1, f2, f_enemy])

        dist_before = np.hypot(f1.x - p1.x, f1.y - p1.y)
        redirect_act = env.encode_action(gx, gy, t1_gx, t1_gy, is_redirect=True)
        env.step(redirect_act)

        assert f1.target == p1.id
        assert f2.target == p1.id
        assert f_enemy.target == p0.id  # Enemy fleet untouched
        if f1 in env.fleets:
            dist_after = np.hypot(f1.x - p1.x, f1.y - p1.y)
            assert dist_after < dist_before

    def test_fleet_redirection_and_planet_launch_are_mutually_exclusive_actions(self):
        """When a cell contains both a friendly planet and a friendly fleet, the bot has TWO
        separate legal actions:
        1. Planet launch (is_redirect=False): launches from planet ONLY, leaving flying fleet's target untouched.
        2. Fleet redirect (is_redirect=True): redirects flying fleet(s) ONLY, leaving planet ships untouched.
        """
        # Case 1: Choose planet launch -> planet launches, flying fleet NOT redirected
        env1 = self._make_redirect_env(enable_fleet_redirection=True)
        p0, p1, p2 = env1.planets[0], env1.planets[1], env1.planets[2]
        p0.ships = 20.0
        p0_gx, p0_gy = env1._world_to_grid(p0.x, p0.y)
        p1_gx, p1_gy = env1._world_to_grid(p1.x, p1.y)

        f1 = Fleet(id=50, owner=env1.PLAYER_1, ships=5.0, x=p0.x, y=p0.y, source=p0.id, target=p2.id, radius=1.0)
        env1.fleets.append(f1)

        send_act = env1.encode_action(p0_gx, p0_gy, p1_gx, p1_gy, is_redirect=False)
        redirect_act = env1.encode_action(p0_gx, p0_gy, p1_gx, p1_gy, is_redirect=True)
        assert send_act != redirect_act
        assert send_act in env1.legal_actions and redirect_act in env1.legal_actions

        env1._apply_action(send_act)
        assert p0.ships < 20.0       # Planet launched
        assert f1.target == p2.id    # Flying fleet was NOT redirected
        assert len(env1.fleets) == 2

        # Case 2: Choose fleet redirect -> flying fleet redirected, planet NOT launched
        env2 = self._make_redirect_env(enable_fleet_redirection=True)
        q0, q1, q2 = env2.planets[0], env2.planets[1], env2.planets[2]
        q0.ships = 20.0
        q0_gx, q0_gy = env2._world_to_grid(q0.x, q0.y)
        q1_gx, q1_gy = env2._world_to_grid(q1.x, q1.y)

        f2 = Fleet(id=51, owner=env2.PLAYER_1, ships=5.0, x=q0.x, y=q0.y, source=q0.id, target=q2.id, radius=1.0)
        env2.fleets.append(f2)

        env2._apply_action(env2.encode_action(q0_gx, q0_gy, q1_gx, q1_gy, is_redirect=True))
        assert f2.target == q1.id    # Flying fleet redirected
        assert q0.ships == 20.0      # Planet did NOT launch
        assert len(env2.fleets) == 1

    def test_timeout_reward_scale_self_play_and_eval_and_elimination(self):
        """timeout_reward_scale discounts timestep.reward on timeout wins/losses while keeping:
        - full +-1.0 reward on elimination wins, and
        - full +-1.0 info['eval_episode_return'] for apples-to-apples eval win-rate logging.
        """
        # 1. Self-play timeout: P1 has higher production, terminates on P2's step (step 2)
        cfg_sp = EasyDict(GalconEnv.default_config())
        cfg_sp.update(dict(
            battle_mode='self_play_mode',
            num_planets=4,
            max_episode_steps=2,
            timeout_reward_scale=0.5,
            map_seed=2,
        ))
        env_sp = GalconEnv(cfg_sp)
        env_sp.reset()
        # Give P1 an extra planet so P1 wins on timeout production
        env_sp.planets[2].owner = env_sp.PLAYER_1
        ts1 = env_sp.step(env_sp.pass_action)  # P1 step 1
        assert not ts1.done
        ts2 = env_sp.step(env_sp.pass_action)  # P2 step 2 -> timeout, P1 wins, acting player is P2
        assert ts2.done
        assert np.isclose(float(ts2.reward), -0.5)
        assert np.isclose(float(ts2.info['eval_episode_return']), -1.0)

        # 2. Eval mode timeout: terminates on P2 (bot) step -> negated to P1 perspective
        cfg_ev = EasyDict(GalconEnv.default_config())
        cfg_ev.update(dict(
            battle_mode='eval_mode',
            bot_action_type='pass',
            num_planets=4,
            max_episode_steps=2,
            timeout_reward_scale=0.5,
            map_seed=2,
        ))
        env_ev = GalconEnv(cfg_ev)
        env_ev.reset()
        env_ev.planets[2].owner = env_ev.PLAYER_1
        ts_ev = env_ev.step(env_ev.pass_action)  # P1 step 1 + bot step 2 -> timeout, P1 wins
        assert ts_ev.done
        assert np.isclose(float(ts_ev.reward), 0.5)
        assert np.isclose(float(ts_ev.info['eval_episode_return']), 1.0)

        # 3. Elimination win with timeout_reward_scale=0.5 still pays full +1.0
        env_elim = GalconEnv(cfg_sp)
        env_elim.reset()
        # Remove all P2 planets/fleets so P1 wins by elimination on step 1
        for p in env_elim.planets:
            if p.owner == env_elim.PLAYER_2:
                p.owner = env_elim.PLAYER_1
        env_elim.fleets.clear()
        ts_elim = env_elim.step(env_elim.pass_action)
        assert ts_elim.done
        assert np.isclose(float(ts_elim.reward), 1.0)
        assert np.isclose(float(ts_elim.info['eval_episode_return']), 1.0)

    def test_random_maps_per_reset_for_both_collect_and_eval(self) -> None:
        cfg = EasyDict(GalconEnv.default_config())
        cfg.update(dict(num_planets=6, map_seed=None))

        def collect_layouts(env: GalconEnv, n: int = 5):
            layouts = []
            for _ in range(n):
                env.reset()
                layouts.append(tuple((round(p.x, 4), round(p.y, 4), round(p.ships, 4), round(p.production, 4)) for p in env.planets))
            return layouts

        # 1. Collector mode (dynamic_seed=True): distinct maps every reset, reproducible for same seed
        env_collect_1 = GalconEnv(cfg)
        env_collect_1.seed(0, dynamic_seed=True)
        collect_layouts_1 = collect_layouts(env_collect_1, 5)
        assert len(set(collect_layouts_1)) == 5

        env_collect_2 = GalconEnv(cfg)
        env_collect_2.seed(0, dynamic_seed=True)
        assert collect_layouts(env_collect_2, 5) == collect_layouts_1

        # 2. Evaluator mode (dynamic_seed=False): also distinct maps every reset when map_seed=None,
        # reproducible for same seed, and independent from collector workers with the same base seed.
        cfg_eval = EasyDict(GalconEnv.default_config())
        cfg_eval.update(dict(battle_mode='eval_mode', num_planets=6, map_seed=None))

        env_eval_1 = GalconEnv(cfg_eval)
        env_eval_1.seed(0, dynamic_seed=False)
        eval_layouts_1 = collect_layouts(env_eval_1, 5)
        assert len(set(eval_layouts_1)) == 5

        env_eval_2 = GalconEnv(cfg_eval)
        env_eval_2.seed(0, dynamic_seed=False)
        assert collect_layouts(env_eval_2, 5) == eval_layouts_1

        assert set(eval_layouts_1).isdisjoint(set(collect_layouts_1))

    def test_explicit_map_seed_stays_fixed_for_both_collect_and_eval(self) -> None:
        for battle_mode, dynamic_seed in [('self_play_mode', True), ('eval_mode', False)]:
            cfg_fixed = EasyDict(GalconEnv.default_config())
            cfg_fixed.update(dict(battle_mode=battle_mode, num_planets=6, map_seed=2))
            env = GalconEnv(cfg_fixed)
            env.seed(0, dynamic_seed=dynamic_seed)

            env.reset()
            layout_1 = [(p.x, p.y, p.ships, p.production) for p in env.planets]
            assert env._current_map_seed == 2

            env.reset()
            layout_2 = [(p.x, p.y, p.ships, p.production) for p in env.planets]
            assert env._current_map_seed == 2
            assert layout_1 == layout_2
