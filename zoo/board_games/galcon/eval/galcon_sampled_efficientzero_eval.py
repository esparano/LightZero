# from zoo.board_games.galcon.config.galcon_sez_bot_mode_config import main_config, create_config
from zoo.board_games.galcon.config.galcon_sez_sp_mode_config_SIMPLE import main_config, create_config
from lzero.entry import eval_muzero
import numpy as np
import re

if __name__ == '__main__':
    """
    Entry point for the evaluation of the Sampled EfficientZero model on the Galcon environment. 

    Variables:
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should point to the ckpt file of the 
        pretrained model. An absolute path is recommended. In LightZero, the path is usually something like 
        ``exp_name/ckpt/ckpt_best.pth.tar``.
        - returns_mean_seeds (:obj:`List[float]`): List to store the mean returns for each seed.
        - returns_seeds (:obj:`List[float]`): List to store the returns for each seed.
        - seeds (:obj:`List[int]`): List of seeds for the environment.
        - num_episodes_each_seed (:obj:`int`): Number of episodes to run for each seed.
        - total_test_episodes (:obj:`int`): Total number of test episodes, computed as the product of the number of 
        seeds and the number of episodes per seed. 
    """
    # model_path = './ckpt/ckpt_best.pth.tar'
    # model_path = './data_sez/galcon_sez_self-play_8-grid_adamw_1e4_260612_192141/ckpt/ckpt_best.pth.tar'
    # model_path = './data_sez/galcon_sez_self-play_8-grid_adamw_1e4_symmetry/ckpt/ckpt_best.pth.tar'
    # model_path = './data_sez/galcon_sez_self-play_8-grid_adamw_1e4_symmetry/ckpt/iteration_0.pth.tar'
    # model_path = 'data_sez/galcon_sez_self-play_8-grid_adamw_1e4_fixed_augment/ckpt/iteration_100000.pth.tar'
    # model_path = 'data_sez/galcon_sez_sp_8p_fixed_augment/ckpt/iteration_600000.pth.tar'
    # model_path = 'data_sez/galcon_sez_sp_8p_fixed_noaugment_map2/ckpt/iteration_130000.pth.tar'
    # model_path = 'data_sez/galcon_sez_sp_8p_fixed_noaugment_map2_pucttweak/ckpt/iteration_20000.pth.tar'
    # model_path = 'data_sez/galcon_sez_sp_8p_fixed_noaugment_map2_pucttweak_losstweak/ckpt/iteration_120000.pth.tar'
    model_path = 'data_sez/galcon_sez_sp_8p_fixed_noaugment_map2_rewardrange/ckpt/iteration_30000.pth.tar'
    # model_path = None
    # seeds = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    seeds = [2]
    main_config.env.map_seed = 2
    num_episodes_each_seed = 1
    # If True, you can play with the agent.
    # main_config.env.agent_vs_human = True
    main_config.env.agent_vs_human = False
    # main_config.env.render_mode = 'image_realtime_mode'
    main_config.env.render_mode = 'image_savefile_mode'
    main_config.env.replay_path = './video/galcon/eval'

    # Set base video name based on model path and extension
    base_replay_suffix = ''
    if model_path != None:
        exp_name_extracted = re.search(r'[^/]+/(.*?)/', model_path)
        if exp_name_extracted:
            model_version_extracted = re.search(r'([^/]+)\.pth\.tar$', model_path)
            if model_version_extracted:
                base_replay_suffix = f'_{exp_name_extracted.group(1)}_{model_version_extracted.group(1)}_'

    # Supported bots: ['fixed', 'random', 'pass', 'mcts']
    eval_bot_types = ['random', 'mcts']
    # probability of eval bot's action being overridden
    main_config.env.prob_pass_action_in_bot = 0
    main_config.env.prob_random_action_in_bot = 0.

    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1

    total_test_episodes = num_episodes_each_seed * len(seeds)
    for bot_type in eval_bot_types:
        main_config.env.bot_action_type = bot_type
        main_config.env.replay_name_suffix = f'{base_replay_suffix}{bot_type}_'
        returns_mean_seeds = []
        returns_seeds = []
        for seed in seeds:
            returns_mean, returns = eval_muzero(
                [main_config, create_config],
                seed=seed,
                num_episodes_each_seed=num_episodes_each_seed,
                print_seed_details=True,
                model_path=model_path
            )
            returns_mean_seeds.append(returns_mean)
            returns_seeds.append(returns)

        returns_mean_seeds = np.array(returns_mean_seeds)
        returns_seeds = np.array(returns_seeds)

        print("=" * 20)
        print(f"[Opponent: {bot_type}] Evaluated {len(seeds)} seed(s) x {num_episodes_each_seed} episode(s).")
        print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns are {returns_seeds}.")
        print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
        print(
            f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}'
        )
        print("=" * 20)
