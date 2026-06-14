import math
from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
num_planets = 8
min_send_ships = 1
send_ratio = 0.5
tick_seconds = 0.25
fleet_speed = 40.0
max_episode_steps = 200
map_seed = None

# Max grid size could be as large as (12 + 12 + 6 + 6 + 0.5) / sqrt(2) = 25.8
grid_square_size = 25.0
grid_max_x = 200.0
grid_max_y = 125.0
# If 1, homes always spawn on an ellipse touching the sides of the box. If in between, the ellipse shrinks proportionally.
home_distance_fraction = 0.8
neutral_min_cost = 0
neutral_max_cost = 50
neutral_min_production = 15
neutral_max_production = 100
fleet_top_k = 1
grid_width = int(math.ceil((2 * grid_max_x) / grid_square_size))
grid_height = int(math.ceil((2 * grid_max_y) / grid_square_size))


collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 50
K = 16
update_per_collect = 50
reanalyze_ratio = 0.
batch_size = 256
max_env_step = int(1e9)

model_path = None

# TODO: ctree does not have the bug fixes for sampled discrete action spaces yet.
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

galcon_sampled_efficientzero_config = dict(
    exp_name='data_sez/galcon_sampled_efficientzero_bot_seed0',
    env=dict(
        # TODO: Reorganize parameters (grouping map gen parameters separately, etc.)
        battle_mode='play_with_bot_mode',
        # Which bot plays during eval. Values: ['random', 'pass', 'fixed']
        # "random" bot is actually quite strong on a map with few, inexpensive neutrals. Changing the default to "pass" for now.
        bot_action_type='random',
        num_planets=num_planets,
        min_send_ships=min_send_ships,
        send_ratio=send_ratio,
        tick_seconds=tick_seconds,
        fleet_speed=fleet_speed,
        max_episode_steps=max_episode_steps,
        map_seed=map_seed,
        grid_square_size=grid_square_size,
        # the map extends from -x to +x and -y to +y
        grid_max_x=grid_max_x,
        grid_max_y=grid_max_y,
        # If 1, homes always spawn on an ellipse touching the sides of the box. If in between, the ellipse shrinks proportionally.
        home_distance_fraction=home_distance_fraction,
        neutral_min_cost=neutral_min_cost,
        neutral_max_cost=neutral_max_cost,
        neutral_min_production=neutral_min_production,
        neutral_max_production=neutral_max_production,
        fleet_top_k=fleet_top_k,
        channel_last=False,
        scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        prob_random_action_in_bot=0.,
        prob_pass_action_in_bot=0.,
        render_mode=None,
        replay_path=None,
    ),
    policy=dict(
        model=dict(
            model_type='conv',
            # There are currently 76 different channels represending planet and fleet info
            observation_shape=(76, grid_height, grid_width),
            image_channel=76,
            # Only the most recent frame is passed to the NN
            frame_stack_num=1,
            action_space_size=grid_width * grid_height * grid_width * grid_height + 1,
            continuous_action_space=False,
            num_of_sampled_actions=K,
            downsample=False,
            self_supervised_learning_loss=True,
            # TODO: bump this up to ~4+ to make sure info from one side of the board can reach the other. 3x3 conv on an 8x8 board needs more res blocks to reach the other side.
            num_res_blocks=4,
            num_channels=64,
            lstm_hidden_size=128,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            # Could even go to 256 here for reward and value heads. Policy should be at least 256 if not more.
            reward_head_hidden_channels=[256],
            value_head_hidden_channels=[256],
            policy_head_hidden_channels=[512],
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
            discrete_action_encoding_type='one_hot',
            norm_type='BN',            
            # Reduce huge action space into a lower dimensional embedding space
            embed_actions= True,
            embedded_action_dim = 32,
        ),
        model_path=model_path,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        battle_mode='play_with_bot_mode',
        mcts_ctree=mcts_ctree,
        game_segment_length=max_episode_steps,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        td_steps=max_episode_steps,
        discount_factor=1,
        # Consider increasing this to 8 to allow 2 full seconds of gameplay...
        num_unroll_steps=5,
        lstm_horizon_len=5,
        policy_loss_type='cross_entropy',
        use_priority=False,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # Whether to reflect games on x and y to augment the game buffer
        symmetric_augment_x=True,
        symmetric_augment_y=True,
        # Use the Galcon obs and action space encoding 
        symmetric_augment_type='Galcon',
    ),
)

galcon_sampled_efficientzero_config = EasyDict(galcon_sampled_efficientzero_config)
main_config = galcon_sampled_efficientzero_config

galcon_sampled_efficientzero_create_config = dict(
    env=dict(
        type='galcon',
        import_names=['zoo.board_games.galcon.envs.galcon_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
)
galcon_sampled_efficientzero_create_config = EasyDict(galcon_sampled_efficientzero_create_config)
create_config = galcon_sampled_efficientzero_create_config

if __name__ == '__main__':
    from lzero.entry import train_muzero

    train_muzero(
        [main_config, create_config],
        seed=0,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )