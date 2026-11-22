from zoo.board_games.tictactoe.config.tictactoe_sampled_efficientzero_sp_mode_config import main_config, create_config
from lzero.entry import eval_muzero
import argparse
import numpy as np


## Example command: python3 -u zoo/board_games/tictactoe/entry/tictactoe_sampled_efficientzero_eval.py
## Optional: override the checkpoint, e.g.
##   python3 -u zoo/board_games/tictactoe/entry/tictactoe_sampled_efficientzero_eval.py \
##       --model_path data_sez/tictactoe_sampled_efficientzero_self-play_smoke_v2_seed0/ckpt/iteration_6260.pth.tar

if __name__ == "__main__":
    """
    Entry point for the evaluation of the Sampled EfficientZero model on the TicTacToe environment.

    Variables:
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should point to the ckpt file of the
        pretrained model. An absolute path is recommended. In LightZero, the path is usually something like
        ``exp_name/ckpt/ckpt_best.pth.tar``. Can be overridden via the ``--model_path`` CLI argument.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./data_sez/tictactoe_sampled_efficientzero_self-play_smoke_v2_seed0/ckpt/iteration_6260.pth.tar",
        help="Path to the SEZ checkpoint to evaluate.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    seeds = [0]
    # NOTE: 2-episode evals are far too noisy for tictactoe (a single [1,1] pair once triggered a
    # premature "converged" early-stop); use at least 10 episodes for a meaningful estimate.
    num_episodes_each_seed = 10
    # If True, you can play with the agent.
    main_config.env.agent_vs_human = False
    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    total_test_episodes = num_episodes_each_seed * len(seeds)

    # Enable saving of replay as a gif, specify the path to save the replay gif
    main_config.env.replay_path = './video'

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

    # Print evaluation results
    print("=" * 20)
    print(f"We evaluated a total of {len(seeds)} seeds. For each seed, we evaluated {num_episodes_each_seed} episode(s).")
    print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns are {returns_seeds}.")
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print(
        f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}'
    )
    print("=" * 20)