import numpy as np
import pytest
from easydict import EasyDict
from gymnasium import spaces
from lzero.mcts.buffer.game_segment import GameSegment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from zoo.board_games.galcon.envs.galcon_env import GalconEnv
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
        assert obs['observation'].shape == (76, 10, 16)
        assert obs['action_mask'].shape == (16 * 10 * 16 * 10 + 1,)

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

    def test_reflect_game_segment(self) -> None:  
        env = GalconEnv(self.cfg)

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
        
        # Populate dummy segment data
        # H=4, W=4
        # obs: C=76, H=4, W=4. Let's make a dummy obs where:
        # Cell (1, 2) has a planet with friendly ships/production (channels 2, 5)
        # local offset (0.2, 0.3) in channels 0, 1
        obs = np.zeros((76, 4, 4), dtype=np.float32)
        obs[0, 2, 1] = 0.2  # local_x
        obs[1, 2, 1] = 0.3  # local_y
        obs[2, 2, 1] = 0.5  # friendly ships
        obs[5, 2, 1] = 0.5  # friendly production
        
        # Friendly fleet in cell (0, 1), pointing to target at world (x, y) normalized to (0.1, 0.9)
        # base_ch = 8 + slot * 5. Let's use slot 0 (base_ch = 8)
        obs[8, 1, 0] = 0.1  # target x
        obs[9, 1, 0] = 0.9  # target y
        obs[11, 1, 0] = 10  # fleet ships (so fleet is present)
        
        segment.obs_segment = [obs]
        
        # Action: source (0, 1) to target (1, 2))
        # Cell indices:
        # source: y*W + x = 1*4 + 0 = 4
        # target: y*W + x = 2*4 + 1 = 9
        # action = source * cell_count + target + 1 = 4 * 16 + 9 + 1 = 74
        action = 74
        segment.action_segment = [action]
        
        action_mask = np.zeros(4 * 4 * 4 * 4 + 1)
        action_mask[action] = 1
        segment.action_mask_segment = [action_mask]
        
        # Sampled actions: list of arrays
        segment.root_sampled_actions = [np.array([0, action])]
        
        # Run reflection
        # Not sure what the right config would be here
        policy_config = EasyDict(dict(
            # Use the Galcon obs and action space encoding 
            symmetric_augment_type='Galcon',
        ))
        reflected = reflect_game_segment(policy_config, segment, reflect_x=True, reflect_y=True)
        
        # Check shape
        assert reflected.obs_segment.shape == (1, 76, 4, 4)
        
        # Reflected positions:
        # fleet  (0, 1) -> ((W-1)-0, (H-1)-1) = (3, 2)
        # planet (1, 2) -> ((W-1)-1, (H-1)-2) = (2, 1)
        # Let's verify planet channel values at (3, 2) (which is grid_y=2, grid_x=3)
        ref_obs = reflected.obs_segment[0]
        assert ref_obs[2, 1, 2] == 0.5
        assert ref_obs[5, 1, 2] == 0.5
        # local_x at (2, 1) should be 1.0 - 0.2 = 0.8
        assert np.isclose(ref_obs[0, 1, 2], 0.8)
        # local_y at (2, 1) should be 1.0 - 0.3 = 0.7
        assert np.isclose(ref_obs[1, 1, 2], 0.7)
        
        # Reflected fleet at (3, 2) (which is grid_y=2, grid_x=3)
        assert ref_obs[11, 2, 3] == 10
        # target_x should be 1.0 - 0.1 = 0.9
        assert np.isclose(ref_obs[8, 2, 3], 0.9)
        # target_y should be 1.0 - 0.9 = 0.1
        assert np.isclose(ref_obs[9, 2, 3], 0.1)
        
        # Reflected action: source (3, 2) to target (2, 1)
        # source cell: 2 * 4 + 3 = 11
        # target cell: 1 * 4 + 2 = 6
        # action = 11 * 16 + 6 + 1 = 108
        assert reflected.action_segment[0] == 183
        assert reflected.action_mask_segment[0][183] == 1
        assert reflected.root_sampled_actions[0][0] == 0   # pass remains pass
        assert reflected.root_sampled_actions[0][1] == 183
