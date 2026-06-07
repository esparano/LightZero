from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 50
update_per_collect = 50
reanalyze_ratio = 0.
batch_size = 256
max_env_step = int(5e5)

# For Connect4 there are at most 7 legal actions.
# In sampled discrete board games, using K=7 avoids excluding legal columns at the root.
K = 7

model_path = None
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

connect4_sampled_efficientzero_config = dict(
    exp_name=f'data_sez/connect4_sampled_efficientzero_self-play_seed0',
    env=dict(
        battle_mode='self_play_mode',
        bot_action_type='rule',
        channel_last=False,
        scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),

        # Connect4Env-required fields used by rendering/bots/eval.
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        prob_random_action_in_bot=0.,
        screen_scaling=9,
        render_mode=None,
        replay_path=None,
    ),
    policy=dict(
        model=dict(
            model_type='conv',
            observation_shape=(3, 6, 7),
            image_channel=3,
            frame_stack_num=1,
            action_space_size=7,
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

        # Important for two-player board-game handling.
        env_type='board_games',
        action_type='varied_action_space',
        battle_mode='self_play_mode',

        # Start with Python tree because its adversarial to_play logic is explicit.
        mcts_ctree=mcts_ctree,

        game_segment_length=int(6 * 7),
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,

        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,

        # Board games should bootstrap from final outcome.
        # Number of steps to look ahead when figuring out the value of the current position.
        # The game always ends in 42 moves, so look ahead 42 moves for the ending value of the game (win/loss).
        td_steps=int(6 * 7),
        discount_factor=1,

        # EfficientZero / SampledEfficientZero rollout settings.
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

connect4_sampled_efficientzero_config = EasyDict(connect4_sampled_efficientzero_config)
main_config = connect4_sampled_efficientzero_config

connect4_sampled_efficientzero_create_config = dict(
    env=dict(
        type='connect4',
        import_names=['zoo.board_games.connect4.envs.connect4_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
)
connect4_sampled_efficientzero_create_config = EasyDict(connect4_sampled_efficientzero_create_config)
create_config = connect4_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero

    train_muzero(
        [main_config, create_config],
        seed=0,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )
