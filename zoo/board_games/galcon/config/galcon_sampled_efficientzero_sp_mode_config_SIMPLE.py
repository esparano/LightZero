import math
from easydict import EasyDict

# Monitoring: 
# tensorboard --logdir=./data_sez/galcon_sampled_efficientzero_self-play_seed0_260608_030657/log/serial/ --host 0.0.0.0 --port 6006

# Debug: 
# python3 /home/evan/.antigravity-ide-server/extensions/ms-python.debugpy-2026.6.0/bundled/libs/debugpy/adapter/../../debugpy/launcher 44683 -- /home/evan/LightZero/zoo/board_games/galcon/config/galcon_sampled_efficientzero_sp_mode_config_SIMPLE.py 

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# num_planets = 8
num_planets = 6
min_send_ships = 1
send_ratio = 0.5
tick_seconds = 0.25
fleet_speed = 40.0
# 100 * 0.25 = 25 seconds per game
max_episode_steps = 100
map_seed = None

# grid square size could be as large as (12 + 12 + 6 + 6 + 0.5) / sqrt(2) = 25.8 while still preventing more than 1 planet per cell
grid_square_size = 25.0
# grid_max_x = 200.0
# grid_max_y = 125.0
grid_max_x = 100.0
grid_max_y = 100.0
# If 1, homes always spawn on an ellipse touching the sides of the box. If in between, the ellipse shrinks proportionally.
home_distance_fraction = 0.8
neutral_min_cost = 0
# neutral_max_cost = 50
neutral_max_cost = 10
neutral_min_production = 15
neutral_max_production = 100
fleet_top_k = 1
grid_width = int(math.ceil((2 * grid_max_x) / grid_square_size))
grid_height = int(math.ceil((2 * grid_max_y) / grid_square_size))

# TODO: Optimize all of this.
collector_env_num = 8
n_episode = 8
# The number of parallel environments to evaluate for
evaluator_env_num = 5
# Raising number of simulations a bit to better utilize the GPU and the more efficient non-Galcon code...
num_simulations = 25
K = 8
# Increase by 4x because symmetry augmentation is enabled
update_per_collect = 50 * 4
reanalyze_ratio = 0.
# This is recommended for RTX 5080 for network of this size... May need to reduce this if increasing action space size.
batch_size = 256
# Training will halt automatically after this many environment steps (ticks)
max_env_step = int(1e9)
model_path = None
# model_path = './data_sez/galcon_sampled_efficientzero_self-play_seed0_260608_224501/ckpt/iteration_31250.pth.tar'
mcts_ctree = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

galcon_sampled_efficientzero_config = dict(
    exp_name='data_sez/galcon_sampled_efficientzero_self-play_8-grid_adamw_1e4_symmetry',
    # exp_name='data_sez/galcon_sampled_efficientzero_self-play_seed0',
    env=dict(
        # TODO: Reorganize parameters (grouping map gen parameters separately, etc.)
        battle_mode='self_play_mode',
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
        prob_pass_action_in_bot=0.8,
        # replay_name_suffix='test',
        # render_mode='state_realtime_mode',
        # render_mode='image_realtime_mode',
        # render_mode='image_savefile_mode',
        # replay_path='./video/galcon',
        replay_path=None,
        # options are: {'mp4', 'gif'}. Only relevant for 'image_savefile_mode'
        # replay_format='gif',
        # replay_screen_scaling=9,
    ),
    policy=dict(
        model=dict(
            model_type='conv',
            # There are currently 44 different channels represending planet and fleet info
            observation_shape=(44, grid_height, grid_width),
            image_channel=44,
            # Only the most recent frame is passed to the NN
            frame_stack_num=1,
            action_space_size=grid_width * grid_height * grid_width * grid_height + 1,
            continuous_action_space=False,
            num_of_sampled_actions=K,
            downsample=False,
            self_supervised_learning_loss=True,
            num_res_blocks=1,
            num_channels=16,
            lstm_hidden_size=32,
            reward_head_channels=8,
            value_head_channels=8,
            policy_head_channels=16,
            reward_head_hidden_channels=[64],
            value_head_hidden_channels=[64],
            policy_head_hidden_channels=[128],
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # Reduce huge action space into a lower dimensional embedding space
            embed_actions= True,
            embedded_action_dim = 16,
        ),
        model_path=model_path,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        battle_mode='self_play_mode',
        mcts_ctree=mcts_ctree,
        game_segment_length=max_episode_steps,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0001,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        td_steps=max_episode_steps,
        discount_factor=1,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        policy_loss_type='cross_entropy',
        use_priority=False,
        n_episode=n_episode,
        eval_freq=int(500),
        # Increase replay buffer by 4 because symmetry augmentation is enabled
        # 100k is typical for alphazero / muzero. With update_per_collect = 200 and batch size 256, we train on 51.2k out of 400k samples. 
        # If validation loss is spiking, decrease update_per_collect. If running out of VRAM, decrease batch size.
        replay_buffer_size=int(1e5 * 4),
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