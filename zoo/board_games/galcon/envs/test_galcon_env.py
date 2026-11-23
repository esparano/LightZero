import numpy as np
import pytest
from easydict import EasyDict
from gymnasium import spaces
from lzero.mcts.buffer.game_segment import GameSegment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from zoo.board_games.galcon.envs.galcon_env import GalconEnv, Fleet, Planet
from lzero.mcts.buffer.game_buffer import reflect_game_segment

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

    def _create_dummy_symmetry_segment(self):
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
                observation_shape=(76, 4, 4),
                image_channel=76,
                action_space_size=4 * 4 * 4 * 4 + 1,
            ),
        ))
        segment = GameSegment(action_space=None, game_segment_length=20, config=game_segment_config)
        obs = np.zeros((76, 4, 4), dtype=np.float32)
        # Planet in cell (x=1, y=2)
        obs[0, 2, 1] = 0.2  # local_x
        obs[1, 2, 1] = 0.3  # local_y
        obs[2, 2, 1] = 0.5  # friendly ships
        obs[5, 2, 1] = 0.5  # friendly production

        # Fleet in cell (x=0, y=1), pointing to target at world (x, y) normalized to (0.1, 0.9)
        obs[8, 1, 0] = 0.1  # target x
        obs[9, 1, 0] = 0.9  # target y
        obs[11, 1, 0] = 10  # fleet ships
        segment.obs_segment = [obs]

        # Action: source (0, 1) -> cell 1*4+0 = 4; target (1, 2) -> cell 2*4+1 = 9
        # action = 4 * 16 + 9 + 1 = 74
        action = 74
        segment.action_segment = [action]
        action_mask = np.zeros(4 * 4 * 4 * 4 + 1)
        action_mask[0] = 1
        action_mask[action] = 1
        segment.action_mask_segment = [action_mask]
        segment.root_sampled_actions = [np.array([0, action])]
        return segment

    def test_reflect_game_segment_x_flip(self) -> None:
        segment = self._create_dummy_symmetry_segment()
        policy_config = EasyDict(dict(symmetric_augment_type='Galcon'))
        reflected = reflect_game_segment(policy_config, segment, reflect_x=True, reflect_y=False)

        assert reflected.obs_segment.shape == (1, 76, 4, 4)
        ref_obs = reflected.obs_segment[0]
        # Planet (x=1, y=2) -> X-flip: x' = 4-1-1 = 2, y' = 2
        assert ref_obs[2, 2, 2] == 0.5
        assert ref_obs[5, 2, 2] == 0.5
        assert np.isclose(ref_obs[0, 2, 2], 1.0 - 0.2)  # local_x inverted
        assert np.isclose(ref_obs[1, 2, 2], 0.3)        # local_y unchanged

        # Fleet (x=0, y=1) -> X-flip: x' = 4-1-0 = 3, y' = 1
        assert ref_obs[11, 1, 3] == 10
        assert np.isclose(ref_obs[8, 1, 3], 1.0 - 0.1)  # target_x inverted
        assert np.isclose(ref_obs[9, 1, 3], 0.9)        # target_y unchanged

        # Action: source (3, 1) -> cell 1*4+3 = 7; target (2, 2) -> cell 2*4+2 = 10
        # action = 7 * 16 + 10 + 1 = 123
        assert reflected.action_segment[0] == 123
        assert reflected.action_mask_segment[0][123] == 1
        assert reflected.action_mask_segment[0][0] == 1
        assert reflected.root_sampled_actions[0][0] == 0
        assert reflected.root_sampled_actions[0][1] == 123

    def test_reflect_game_segment_y_flip(self) -> None:
        segment = self._create_dummy_symmetry_segment()
        policy_config = EasyDict(dict(symmetric_augment_type='Galcon'))
        reflected = reflect_game_segment(policy_config, segment, reflect_x=False, reflect_y=True)

        assert reflected.obs_segment.shape == (1, 76, 4, 4)
        ref_obs = reflected.obs_segment[0]
        # Planet (x=1, y=2) -> Y-flip: x' = 1, y' = 4-1-2 = 1
        assert ref_obs[2, 1, 1] == 0.5
        assert ref_obs[5, 1, 1] == 0.5
        assert np.isclose(ref_obs[0, 1, 1], 0.2)        # local_x unchanged
        assert np.isclose(ref_obs[1, 1, 1], 1.0 - 0.3)  # local_y inverted

        # Fleet (x=0, y=1) -> Y-flip: x' = 0, y' = 4-1-1 = 2
        assert ref_obs[11, 2, 0] == 10
        assert np.isclose(ref_obs[8, 2, 0], 0.1)        # target_x unchanged
        assert np.isclose(ref_obs[9, 2, 0], 1.0 - 0.9)  # target_y inverted

        # Action: source (0, 2) -> cell 2*4+0 = 8; target (1, 1) -> cell 1*4+1 = 5
        # action = 8 * 16 + 5 + 1 = 134
        assert reflected.action_segment[0] == 134
        assert reflected.action_mask_segment[0][134] == 1
        assert reflected.action_mask_segment[0][0] == 1
        assert reflected.root_sampled_actions[0][0] == 0
        assert reflected.root_sampled_actions[0][1] == 134

    def test_reflect_game_segment_xy_flip(self) -> None:
        segment = self._create_dummy_symmetry_segment()
        policy_config = EasyDict(dict(symmetric_augment_type='Galcon'))
        reflected = reflect_game_segment(policy_config, segment, reflect_x=True, reflect_y=True)

        assert reflected.obs_segment.shape == (1, 76, 4, 4)
        ref_obs = reflected.obs_segment[0]
        # Planet (x=1, y=2) -> XY-flip: x' = 2, y' = 1
        assert ref_obs[2, 1, 2] == 0.5
        assert ref_obs[5, 1, 2] == 0.5
        assert np.isclose(ref_obs[0, 1, 2], 1.0 - 0.2)  # local_x inverted
        assert np.isclose(ref_obs[1, 1, 2], 1.0 - 0.3)  # local_y inverted

        # Fleet (x=0, y=1) -> XY-flip: x' = 3, y' = 2
        assert ref_obs[11, 2, 3] == 10
        assert np.isclose(ref_obs[8, 2, 3], 1.0 - 0.1)  # target_x inverted
        assert np.isclose(ref_obs[9, 2, 3], 1.0 - 0.9)  # target_y inverted

        # Action: source (3, 2) -> cell 2*4+3 = 11; target (2, 1) -> cell 1*4+2 = 6
        # action = 11 * 16 + 6 + 1 = 183
        assert reflected.action_segment[0] == 183
        assert reflected.action_mask_segment[0][183] == 1
        assert reflected.action_mask_segment[0][0] == 1
        assert reflected.root_sampled_actions[0][0] == 0
        assert reflected.root_sampled_actions[0][1] == 183

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
