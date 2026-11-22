from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
n_evaluator_episode = 10
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(5e4)
reanalyze_ratio = 0.

# For TicTacToe there are at most 9 legal actions.
# Using K=9 samples all legal actions at the root; reduce (e.g. K=5) to test subsampling.
K = 9

model_path = None
mcts_ctree = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

tictactoe_sampled_efficientzero_config = dict(
    exp_name=f'data_sez/tictactoe_sez_sp-mode_ns{num_simulations}_upc{update_per_collect}_K{K}_ctree{mcts_ctree}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        bot_action_type='alpha_beta_pruning',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        # Keep above max possible return (+1) so training runs the full max_env_step curve.
        stop_value=2,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 3, 3),
            action_space_size=9,
            image_channel=3,
            continuous_action_space=False,
            num_of_sampled_actions=K,
            # Small model matching tictactoe_efficientzero_sp_mode_config.py.
            num_res_blocks=1,
            num_channels=16,
            lstm_hidden_size=32,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            reward_head_hidden_channels=[8],
            value_head_hidden_channels=[8],
            policy_head_hidden_channels=[8],
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
            downsample=False,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        model_path=model_path,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        battle_mode='self_play_mode',
        mcts_ctree=mcts_ctree,
        use_augmentation=False,
        game_segment_length=9,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE: In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        num_unroll_steps=3,
        lstm_horizon_len=3,
        # NOTE: In board_games, we set discount_factor=1.
        discount_factor=1,
        use_priority=False,
        n_episode=n_episode,
        eval_freq=500,
        replay_buffer_size=int(1e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
tictactoe_sampled_efficientzero_config = EasyDict(tictactoe_sampled_efficientzero_config)
main_config = tictactoe_sampled_efficientzero_config

tictactoe_sampled_efficientzero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
)
tictactoe_sampled_efficientzero_create_config = EasyDict(tictactoe_sampled_efficientzero_create_config)
create_config = tictactoe_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero(
        [main_config, create_config],
        seed=0,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step,
    )
