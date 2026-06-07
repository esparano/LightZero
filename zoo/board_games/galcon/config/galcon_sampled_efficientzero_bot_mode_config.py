from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
num_planets = 8
min_send_ships = 1
send_ratio = 0.5
tick_seconds = 0.25
max_episode_steps = 200

collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 50
K = 16
update_per_collect = 50
reanalyze_ratio = 0.
batch_size = 256
max_env_step = int(5e5)
model_path = None
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

galcon_sampled_efficientzero_config = dict(
    exp_name='data_sez/galcon_sampled_efficientzero_bot_seed0',
    env=dict(
        battle_mode='play_with_bot_mode',
        bot_action_type='fixed',
        num_planets=num_planets,
        min_send_ships=min_send_ships,
        send_ratio=send_ratio,
        tick_seconds=tick_seconds,
        max_episode_steps=max_episode_steps,
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
        render_mode=None,
        replay_path=None,
    ),
    policy=dict(
        model=dict(
            model_type='conv',
            # Placeholder until the grid observation spec is finalized.
            observation_shape=(1, num_planets, num_planets),
            image_channel=1,
            frame_stack_num=1,
            action_space_size=num_planets * num_planets + 1,
            continuous_action_space=False,
            num_of_sampled_actions=K,
            downsample=False,
            self_supervised_learning_loss=True,
            num_res_blocks=1,
            num_channels=64,
            lstm_hidden_size=128,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            reward_head_hidden_channels=[128],
            value_head_hidden_channels=[128],
            policy_head_hidden_channels=[128],
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
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
        num_unroll_steps=5,
        lstm_horizon_len=5,
        policy_loss_type='cross_entropy',
        use_priority=False,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
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