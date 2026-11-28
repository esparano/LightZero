import copy
import threading
from typing import Any, List, Tuple

import numpy as np
import torch
from ding.utils import BUFFER_REGISTRY

from lzero.mcts.tree_search.mcts_ctree_sampled import SampledEfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree_sampled import SampledEfficientZeroMCTSPtree as MCTSPtree
from lzero.mcts.utils import prepare_observation, generate_random_actions_discrete
from lzero.policy import DiscreteSupport, to_detach_cpu_numpy, concat_output, concat_output_value, inverse_scalar_transform
from .game_buffer_efficientzero import EfficientZeroGameBuffer


@BUFFER_REGISTRY.register('game_buffer_sampled_efficientzero')
class SampledEfficientZeroGameBuffer(EfficientZeroGameBuffer):
    """
    Overview:
        The specific game buffer for Sampled EfficientZero policy.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        assert self._cfg.env_type in ['not_board_games', 'board_games']
        assert self._cfg.action_type in ['fixed_action_space', 'varied_action_space']
        self.replay_buffer_size = self._cfg.replay_buffer_size
        self.batch_size = self._cfg.batch_size
        self._alpha = self._cfg.priority_prob_alpha
        self._beta = self._cfg.priority_prob_beta

        self.game_segment_buffer = []
        self.game_pos_priorities = []
        self.game_segment_game_pos_look_up = []

        self.keep_ratio = 1
        self.num_of_collected_episodes = 0
        self.base_idx = 0
        self.clear_time = 0

        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range)
        self._segment_update_lock = threading.Lock()
        self._discount_powers = None
        self._discount_powers_list = None
        self._discount_powers_gamma = None

    def _get_discount_powers(self, length: int) -> Tuple[np.ndarray, List[float]]:
        """Return cached ``(ndarray, list)`` of ``[gamma**0, ..., gamma**(L-1)]`` with ``L >= length``."""
        gamma = float(self._cfg.discount_factor)
        pows = getattr(self, '_discount_powers', None)
        pows_list = getattr(self, '_discount_powers_list', None)
        cached_gamma = getattr(self, '_discount_powers_gamma', None)
        if pows is None or pows_list is None or cached_gamma != gamma or len(pows) < length:
            alloc_len = max(int(length), 256)
            pows = gamma ** np.arange(alloc_len, dtype=np.float64)
            pows_list = pows.tolist()
            self._discount_powers = pows
            self._discount_powers_list = pows_list
            self._discount_powers_gamma = gamma
        return pows, pows_list

    def _get_segment_update_lock(self) -> threading.Lock:
        """Return the per-buffer lock guarding paired (child_visit, root_sampled_actions) updates."""
        lock = getattr(self, '_segment_update_lock', None)
        if lock is None:
            lock = threading.Lock()
            self._segment_update_lock = lock
        return lock

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_segment_update_lock', None)
        state.pop('_discount_powers', None)
        state.pop('_discount_powers_list', None)
        state.pop('_discount_powers_gamma', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._segment_update_lock = threading.Lock()
        self._discount_powers = None
        self._discount_powers_list = None
        self._discount_powers_gamma = None

    def sample(self, batch_size: int, policy: Any) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training
        Arguments:
            - batch_size (:obj:`int`): batch size
            - policy (:obj:`torch.tensor`): model of policy
        Returns:
            - train_data (:obj:`List`): List of train data
        """

        target_model = policy._target_model
        target_device = torch.device(self._cfg.device)
        first_param = next(target_model.parameters(), None)
        if first_param is None or first_param.device != target_device:
            target_model.to(target_device)
        if target_model.training:
            target_model.eval()

        reward_value_context, policy_re_context, policy_non_re_context, current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio
        )

        # target reward, target value
        batch_value_prefixs, batch_target_values = self._compute_target_reward_value(
            reward_value_context, policy._target_model
        )

        batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self._cfg.model.num_of_sampled_actions
        )

        batch_target_policies_re = []
        if policy_re_context is not None:
            # target policy
            assert (self._cfg.reanalyze_ratio > 0 and self._cfg.reanalyze_outdated is True), \
                "in sampled efficientzero, if self._cfg.reanalyze_ratio>0, you must set self._cfg.reanalyze_outdated=True"
            batch_target_policies_re, root_sampled_actions = self._compute_target_policy_reanalyzed(
                policy_re_context, policy._target_model
            )
            # ==============================================================
            # fix reanalyze in sez:
            # use the latest root_sampled_actions after the reanalyze process,
            # because the batch_target_policies_re is corresponding to the latest root_sampled_actions
            # ==============================================================
            # current_batch = [obs_list, action_list, root_sampled_actions_list, mask_list, batch_index_list, weights_list, make_time_list]
            reanalyze_num = batch_target_policies_re.shape[0]
            current_batch[2][:reanalyze_num] = root_sampled_actions.reshape(
                (reanalyze_num,) + current_batch[2].shape[1:]
            )

        if policy_re_context is not None and policy_non_re_context is not None:
            batch_target_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re], axis=0)
        elif policy_re_context is not None:
            batch_target_policies = batch_target_policies_re
        else:
            batch_target_policies = batch_target_policies_non_re

        target_batch = [batch_value_prefixs, batch_target_values, batch_target_policies]
        # a batch contains the current_batch and the target_batch
        train_data = [current_batch, target_batch]
        return train_data

    def _make_batch(self, batch_size: int, reanalyze_ratio: float) -> Tuple[Any]:
        """
        Overview:
            first sample orig_data through ``_sample_orig_data()``,
            then prepare the context of a batch:
                reward_value_context:        the context of reanalyzed value targets
                policy_re_context:           the context of reanalyzed policy targets
                policy_non_re_context:       the context of non-reanalyzed policy targets
                current_batch:                the inputs of batch
        Arguments:
            - batch_size (:obj:`int`): the batch size of orig_data from replay buffer.
            - reanalyze_ratio (:obj:`float`): ratio of reanalyzed policy (value is 100% reanalyzed)
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # obtain the batch context from replay buffer
        orig_data = self._sample_orig_data(batch_size)
        game_lst, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data
        batch_size = len(batch_index_list)
        num_unroll_steps = int(self._cfg.num_unroll_steps)
        unroll_stride = num_unroll_steps + 1
        continuous_action_space = bool(self._cfg.model.continuous_action_space)
        action_space_size = int(self._cfg.model.action_space_size)
        num_of_sampled_actions = int(self._cfg.model.num_of_sampled_actions)
        reshape_sampled = bool(self._cfg.mcts_ctree)

        obs_list, action_list = [], []
        mask_arr = np.zeros((batch_size, unroll_stride), dtype=np.float64)
        root_sampled_actions_list = []
        reanalyze_num = int(batch_size * reanalyze_ratio)
        non_re_child_visit_snapshots = []
        seg_lock = self._get_segment_update_lock()

        # Fast preallocated observation assembly for uniform conv np.ndarray frames (avoids double 16MB copy).
        use_fast_conv_obs = False
        obs_arr = None
        unroll_frames = 0
        expected_frame_shape = None
        if (
            batch_size > 0
            and self._cfg.model.model_type == 'conv'
            and not getattr(game_lst[0], 'transform2string', False)
        ):
            seg0_obs = getattr(game_lst[0], 'obs_segment', None)
            if seg0_obs is not None and len(seg0_obs) > 0:
                frame0 = seg0_obs[0]
                if isinstance(frame0, np.ndarray) and frame0.ndim == 3:
                    unroll_frames = int(game_lst[0].frame_stack_num) + num_unroll_steps
                    expected_frame_shape = frame0.shape
                    obs_arr = np.empty((batch_size, unroll_frames, *expected_frame_shape), dtype=frame0.dtype)
                    use_fast_conv_obs = True

        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_lst[i]
            pos_in_game_segment = pos_in_game_segment_list[i]
            # ==============================================================
            # sampled related core code
            # ==============================================================
            actions_slice = game.action_segment[pos_in_game_segment:pos_in_game_segment + num_unroll_steps]
            actions_tmp = actions_slice.tolist() if hasattr(actions_slice, 'tolist') else list(actions_slice)

            if reanalyze_ratio > 0:
                # Snapshot root_sampled_actions (and the needed unroll window of child_visit_segment
                # for non-reanalyzed rows) together under seg_lock so concurrent reanalyze updates
                # cannot cause a torn read.
                with seg_lock:
                    # NOTE: self._cfg.num_unroll_steps + 1
                    root_sampled_actions_slice = game.root_sampled_actions[
                        pos_in_game_segment:pos_in_game_segment + unroll_stride
                    ]
                    root_sampled_actions_tmp = [
                        np.array(a, copy=True) if isinstance(a, np.ndarray) else copy.deepcopy(a)
                        for a in root_sampled_actions_slice
                    ]
                    if i >= reanalyze_num:
                        cv = game.child_visit_segment
                        end_idx = min(pos_in_game_segment + unroll_stride, len(cv))
                        non_re_child_visit_snapshots.append({
                            t: (cv[t].copy() if isinstance(cv[t], np.ndarray) else copy.deepcopy(cv[t]))
                            for t in range(pos_in_game_segment, end_idx)
                        })
            else:
                # NOTE: self._cfg.num_unroll_steps + 1
                root_sampled_actions_tmp = list(
                    game.root_sampled_actions[pos_in_game_segment:pos_in_game_segment + unroll_stride]
                )

            # add mask for invalid actions (out of trajectory), 1 for valid, 0 for invalid
            valid_len = len(root_sampled_actions_tmp)
            if valid_len > 0:
                mask_arr[i, :min(valid_len, unroll_stride)] = 1.0

            # pad random action only when actually needed (avoids ~10k zero-length calls)
            act_pad = num_unroll_steps - len(actions_tmp)
            rsa_pad = unroll_stride - valid_len
            if continuous_action_space:
                if act_pad > 0:
                    actions_tmp += [np.random.randn(action_space_size) for _ in range(act_pad)]
                if rsa_pad > 0:
                    root_sampled_actions_tmp += [
                        np.random.rand(num_of_sampled_actions, action_space_size) for _ in range(rsa_pad)
                    ]
            else:
                if act_pad > 0:
                    actions_tmp += generate_random_actions_discrete(act_pad, action_space_size, 1)
                if rsa_pad > 0:
                    root_sampled_actions_tmp += generate_random_actions_discrete(
                        rsa_pad, action_space_size, num_of_sampled_actions, reshape=reshape_sampled
                    )

            # obtain the input observations
            if use_fast_conv_obs:
                if getattr(game, 'transform2string', False):
                    use_fast_conv_obs = False
                    obs_list = [
                        game_lst[j].get_unroll_obs(
                            pos_in_game_segment_list[j], num_unroll_steps=num_unroll_steps, padding=True
                        )
                        for j in range(i)
                    ]
                else:
                    frames = game.obs_segment[pos_in_game_segment:pos_in_game_segment + unroll_frames]
                    n_avail = len(frames)
                    if n_avail > 0:
                        row_ok = True
                        if n_avail == unroll_frames and isinstance(frames, np.ndarray) and frames.shape[1:] == expected_frame_shape:
                            obs_arr[i] = frames
                        else:
                            for fi in range(n_avail):
                                frame = frames[fi]
                                if isinstance(frame, np.ndarray) and frame.shape == expected_frame_shape:
                                    obs_arr[i, fi] = frame
                                else:
                                    row_ok = False
                                    break
                            if row_ok and n_avail < unroll_frames:
                                obs_arr[i, n_avail:] = frames[-1]
                        if not row_ok:
                            use_fast_conv_obs = False
                            obs_list = [
                                game_lst[j].get_unroll_obs(
                                    pos_in_game_segment_list[j], num_unroll_steps=num_unroll_steps, padding=True
                                )
                                for j in range(i)
                            ]
                    else:
                        use_fast_conv_obs = False
                        obs_list = [
                            game_lst[j].get_unroll_obs(
                                pos_in_game_segment_list[j], num_unroll_steps=num_unroll_steps, padding=True
                            )
                            for j in range(i)
                        ]

            if not use_fast_conv_obs:
                obs_list.append(
                    game.get_unroll_obs(
                        pos_in_game_segment, num_unroll_steps=num_unroll_steps, padding=True
                    )
                )

            action_list.append(actions_tmp)
            root_sampled_actions_list.append(root_sampled_actions_tmp)

        # formalize the input observations
        if use_fast_conv_obs:
            obs_list = obs_arr.reshape(
                batch_size, unroll_frames * expected_frame_shape[0], expected_frame_shape[1], expected_frame_shape[2]
            )
        else:
            obs_list = prepare_observation(obs_list, self._cfg.model.model_type)
        # ==============================================================
        # sampled related core code
        # ==============================================================
        # formalize the inputs of a batch
        current_batch = [
            obs_list,
            np.asarray(action_list),
            np.asarray(root_sampled_actions_list),
            mask_arr,
            np.asarray(batch_index_list),
            np.asarray(weights_list),
            np.asarray(make_time_list),
        ]

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(
            batch_index_list, game_lst, pos_in_game_segment_list, total_transitions
        )
        """
        only reanalyze recent reanalyze_ratio (e.g. 50%) data
        if self._cfg.reanalyze_outdated is True, batch_index_list is sorted according to its generated env_steps
        0: reanalyze_num -> reanalyzed policy, reanalyze_num:end -> non reanalyzed policy
        """
        # reanalyzed policy
        if reanalyze_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_lst[:reanalyze_num], pos_in_game_segment_list[:reanalyze_num]
            )
        else:
            policy_re_context = None

        # non reanalyzed policy
        if reanalyze_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_lst[reanalyze_num:], pos_in_game_segment_list[reanalyze_num:]
            )
            # Replace live child_visit references with the lock-consistent snapshots taken alongside root_sampled_actions.
            if reanalyze_ratio > 0:
                policy_non_re_context[1] = non_re_child_visit_snapshots
        else:
            policy_non_re_context = None

        context = reward_value_context, policy_re_context, policy_non_re_context, current_batch
        return context

    def _prepare_policy_reanalyzed_context(
            self, batch_index_list: List[Any], game_segment_list: List[Any], pos_in_game_segment_list: List[int]
    ) -> List[Any]:
        """Extend base reanalyze context with segment ``root_sampled_actions`` so in-place updates stay aligned."""
        policy_re_context = super()._prepare_policy_reanalyzed_context(
            batch_index_list, game_segment_list, pos_in_game_segment_list
        )
        root_sampled_actions_segments = [game_segment.root_sampled_actions for game_segment in game_segment_list]
        policy_re_context.append(root_sampled_actions_segments)
        return policy_re_context

    def _compute_target_reward_value(self, reward_value_context: List[Any], model: Any) -> List[np.ndarray]:
        """
        Overview:
            prepare reward and value targets from the context of rewards and values.
        Arguments:
            - reward_value_context (:obj:'list'): the reward value context
            - model (:obj:'torch.tensor'):model of the target model
        Returns:
            - batch_value_prefixs (:obj:'np.ndarray): batch of value prefix
            - batch_target_values (:obj:'np.ndarray): batch of value estimation
        """
        value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, game_segment_lens, td_steps_list, action_mask_segment, \
        to_play_segment = reward_value_context  # noqa

        # transition_batch_size = game_segment_batch_size * (num_unroll_steps+1)
        transition_batch_size = len(value_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)
        value_mask_arr = np.asarray(value_mask, dtype=np.float64)
        active_indices = np.flatnonzero(value_mask_arr)

        batch_target_values, batch_value_prefixs = [], []
        with torch.no_grad():
            if active_indices.size == 0:
                # All bootstrap targets are masked out (e.g. td_steps >= episode length),
                # so neither legal-action extraction nor target-model inference affects value targets.
                value_list = [0.0] * transition_batch_size
            elif self._cfg.use_root_value:
                active_count = int(active_indices.size)
                unroll_stride = self._cfg.num_unroll_steps + 1
                active_to_play = []
                active_legal_actions = []
                for idx in active_indices:
                    b, k = divmod(int(idx), unroll_stride)
                    bootstrap_idx = int(pos_in_game_segment_list[b]) + k + int(td_steps_list[idx])
                    tp_seg = to_play_segment[b]
                    tp = int(tp_seg[bootstrap_idx]) if bootstrap_idx < len(tp_seg) else -1
                    active_to_play.append(tp)
                    if self._cfg.model.continuous_action_space is True:
                        # NOTE: in continuous action space env: we set all legal_actions as -1
                        active_legal_actions.append([-1 for _ in range(self._cfg.model.action_space_size)])
                    else:
                        am_seg = action_mask_segment[b]
                        am = (
                            am_seg[bootstrap_idx]
                            if bootstrap_idx < len(am_seg)
                            else np.ones(self._cfg.model.action_space_size, dtype=np.int8)
                        )
                        active_legal_actions.append(np.nonzero(am)[0].tolist())

                active_obs_list = (
                    value_obs_list
                    if active_count == transition_batch_size
                    else [value_obs_list[idx] for idx in active_indices]
                )
                value_obs_arr = prepare_observation(active_obs_list, self._cfg.model.model_type)
                # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
                slices = int(np.ceil(active_count / self._cfg.mini_infer_size))
                network_output = []
                for i in range(slices):
                    beg_index = self._cfg.mini_infer_size * i
                    end_index = self._cfg.mini_infer_size * (i + 1)
                    m_obs = torch.from_numpy(value_obs_arr[beg_index:end_index]).to(self._cfg.device).float()

                    # calculate the target value
                    m_output = model.initial_inference(m_obs)

                    # TODO(pu)
                    if not model.training:
                        # if not in training, obtain the scalars of the value/reward
                        [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                            [
                                m_output.latent_state,
                                inverse_scalar_transform(
                                    m_output.value,
                                    self.value_support,
                                    categorical_distribution=self._cfg.model.categorical_distribution,
                                ),
                                m_output.policy_logits
                            ]
                        )
                        m_output.reward_hidden_state = (
                            m_output.reward_hidden_state[0].detach().cpu().numpy(),
                            m_output.reward_hidden_state[1].detach().cpu().numpy()
                        )

                    network_output.append(m_output)

                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, latent_state_roots, reward_hidden_state_roots = concat_output(
                    network_output, data_type='efficientzero'
                )
                value_prefix_pool = np.asarray(value_prefix_pool).reshape(-1).tolist()
                policy_logits_pool = policy_logits_pool.tolist()
                # generate the noises for the root nodes
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.num_of_sampled_actions
                                        ).astype(np.float32).tolist() for _ in range(active_count)
                ] if self._cfg.reanalyze_noise else None

                if self._cfg.mcts_ctree:
                    # cpp mcts_tree
                    # prepare the root nodes for MCTS
                    roots = MCTSCtree.roots(
                        active_count, active_legal_actions, self._cfg.model.action_space_size,
                        self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                    )
                    if self._cfg.reanalyze_noise:
                        roots.prepare(
                            self._cfg.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, active_to_play
                        )
                    else:
                        roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, active_to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTSCtree(self._cfg).search(
                        roots, model, latent_state_roots, reward_hidden_state_roots, active_to_play
                    )
                else:
                    # python mcts_tree
                    roots = MCTSPtree.roots(
                        active_count, active_legal_actions, self._cfg.model.action_space_size,
                        self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                    )
                    if self._cfg.reanalyze_noise:
                        roots.prepare(
                            self._cfg.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, active_to_play
                        )
                    else:
                        roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, active_to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTSPtree(self._cfg).search(
                        roots, model, latent_state_roots, reward_hidden_state_roots, active_to_play
                    )

                active_values = np.asarray(roots.get_values(), dtype=np.float64)
                if active_count == transition_batch_size:
                    value_list = active_values
                else:
                    value_list = np.zeros(transition_batch_size, dtype=active_values.dtype)
                    value_list[active_indices] = active_values
            else:
                # use_root_value=False: legal_actions/to_play are unused and only unmasked bootstrap
                # observations contribute to value targets.
                active_obs_list = (
                    value_obs_list
                    if active_indices.size == transition_batch_size
                    else [value_obs_list[idx] for idx in active_indices]
                )
                active_obs_arr = prepare_observation(active_obs_list, self._cfg.model.model_type)
                slices = int(np.ceil(active_indices.size / self._cfg.mini_infer_size))
                network_output = []
                for i in range(slices):
                    beg_index = self._cfg.mini_infer_size * i
                    end_index = self._cfg.mini_infer_size * (i + 1)
                    m_obs = torch.from_numpy(active_obs_arr[beg_index:end_index]).to(self._cfg.device).float()
                    m_output = model.initial_inference(m_obs)
                    if not model.training:
                        [m_output.value] = to_detach_cpu_numpy(
                            [
                                inverse_scalar_transform(
                                    m_output.value,
                                    self.value_support,
                                    categorical_distribution=self._cfg.model.categorical_distribution,
                                )
                            ]
                        )
                    network_output.append(m_output)

                active_values = concat_output_value(network_output).reshape(-1)
                if active_indices.size == transition_batch_size:
                    value_list = active_values
                else:
                    value_list = np.zeros(transition_batch_size, dtype=active_values.dtype)
                    value_list[active_indices] = active_values

            is_two_player_zero_sum = bool(
                self._cfg.env_type == 'board_games'
                and len(to_play_segment) > 0
                and len(to_play_segment[0]) > 0
                and to_play_segment[0][0] in (1, 2)
            )
            unroll_stride = int(self._cfg.num_unroll_steps) + 1
            lstm_horizon_len = int(self._cfg.lstm_horizon_len)

            has_boot_values = active_indices.size > 0
            if has_boot_values:
                td_arr = np.asarray(td_steps_list, dtype=np.int64)
                gamma_pows, gamma_list = self._get_discount_powers(int(td_arr.max()) + unroll_stride + 1)
                discounts = gamma_pows[td_arr]
                if is_two_player_zero_sum:
                    discounts = np.where((td_arr & 1) == 0, discounts, -discounts)
                val_list = (np.asarray(value_list, dtype=np.float64).reshape(-1) * discounts * value_mask_arr).tolist()
            else:
                max_span_est = int(self._cfg.td_steps) + unroll_stride + 1
                gamma_pows, gamma_list = self._get_discount_powers(max_span_est)
                val_list = None

            tv_rows = []
            vp_rows = []

            horizon_id = 0
            value_index = 0
            for game_segment_len_non_re, reward_list, state_index, to_play_list in zip(
                game_segment_lens, rewards_list, pos_in_game_segment_list, to_play_segment
            ):
                r_len = len(reward_list)

                # Compute exact upper bound touched by any unroll step in this segment.
                max_boot = 0
                for k in range(unroll_stride):
                    bi = state_index + k + td_steps_list[value_index + k]
                    if bi > max_boot:
                        max_boot = bi
                if max_boot > r_len:
                    max_boot = r_len

                span = max_boot - state_index
                if span > 0:
                    if span + 1 > len(gamma_list):
                        gamma_pows, gamma_list = self._get_discount_powers(span + 1)

                    if type(reward_list) is list:
                        seg = reward_list[state_index:max_boot]
                        nnz = len(seg) - seg.count(0.0)
                        if nnz == 0:
                            mode = 0
                        elif nnz == 1 and seg[-1] != 0.0:
                            mode = 1
                            p0 = max_boot - 1
                            r0 = float(seg[-1])
                            tp0 = to_play_list[p0] if is_two_player_zero_sum else 0
                        elif nnz <= 8:
                            mode = 2
                            nz_pos = []
                            nz_vals = []
                            nz_tp = [] if is_two_player_zero_sum else None
                            for off, rv in enumerate(seg):
                                if rv != 0.0:
                                    p = state_index + off
                                    nz_pos.append(p)
                                    nz_vals.append(float(rv))
                                    if is_two_player_zero_sum:
                                        nz_tp.append(to_play_list[p])
                        else:
                            mode = 3
                            seg_arr = np.asarray(seg, dtype=np.float64)
                            tp_arr = np.asarray(to_play_list[state_index:max_boot], dtype=np.int64) if is_two_player_zero_sum else None
                    else:
                        if isinstance(reward_list, np.ndarray):
                            seg_view = reward_list[state_index:max_boot]
                        else:
                            seg_view = np.asarray(reward_list[state_index:max_boot], dtype=np.float64)
                        nz_rel = seg_view.nonzero()[0]
                        nnz = int(nz_rel.size)
                        if nnz == 0:
                            mode = 0
                        elif nnz == 1:
                            mode = 1
                            off0 = int(nz_rel[0])
                            p0 = state_index + off0
                            r0 = float(seg_view[off0])
                            tp0 = to_play_list[p0] if is_two_player_zero_sum else 0
                        elif nnz <= 8:
                            mode = 2
                            nz_pos = [state_index + int(off) for off in nz_rel]
                            nz_vals = [float(seg_view[int(off)]) for off in nz_rel]
                            nz_tp = [to_play_list[p] for p in nz_pos] if is_two_player_zero_sum else None
                        else:
                            mode = 3
                            seg_arr = seg_view.astype(np.float64, copy=False)
                            tp_arr = np.asarray(to_play_list[state_index:max_boot], dtype=np.int64) if is_two_player_zero_sum else None
                else:
                    mode = 0

                tv_row = [0.0] * unroll_stride
                vp_row = [0.0] * unroll_stride
                value_prefix = 0.0
                for k in range(unroll_stride):
                    current_index = state_index + k
                    bootstrap_index = current_index + td_steps_list[value_index]
                    boot_clamped = bootstrap_index if bootstrap_index < r_len else r_len

                    acc = val_list[value_index] if has_boot_values else 0.0
                    if mode == 1:
                        if current_index <= p0 < boot_clamped:
                            w = gamma_list[p0 - current_index]
                            if is_two_player_zero_sum:
                                acc += (r0 * w) if tp0 == to_play_list[current_index] else (-r0 * w)
                            else:
                                acc += r0 * w
                    elif mode == 2:
                        if current_index < boot_clamped:
                            if is_two_player_zero_sum:
                                cur_tp = to_play_list[current_index]
                                for p, r_val, p_tp in zip(nz_pos, nz_vals, nz_tp):
                                    if current_index <= p < boot_clamped:
                                        w = gamma_list[p - current_index]
                                        acc += (r_val * w) if p_tp == cur_tp else (-r_val * w)
                            else:
                                for p, r_val in zip(nz_pos, nz_vals):
                                    if current_index <= p < boot_clamped:
                                        acc += r_val * gamma_list[p - current_index]
                    elif mode == 3:
                        if current_index < boot_clamped:
                            rel_start = current_index - state_index
                            rel_end = boot_clamped - state_index
                            r_slice = seg_arr[rel_start:rel_end]
                            n_slice = rel_end - rel_start
                            if is_two_player_zero_sum:
                                cur_tp = tp_arr[rel_start]
                                signed_r = np.where(tp_arr[rel_start:rel_end] == cur_tp, r_slice, -r_slice)
                                acc += float(np.dot(signed_r, gamma_pows[:n_slice]))
                            else:
                                acc += float(np.dot(r_slice, gamma_pows[:n_slice]))

                    # reset every lstm_horizon_len
                    if horizon_id % lstm_horizon_len == 0:
                        value_prefix = 0.0
                    horizon_id += 1

                    if current_index < game_segment_len_non_re:
                        tv_row[k] = acc
                        if mode != 0 or current_index >= max_boot:
                            value_prefix += float(reward_list[current_index])
                        vp_row[k] = value_prefix
                    else:
                        vp_row[k] = value_prefix

                    value_index += 1

                tv_rows.append(tv_row)
                vp_rows.append(vp_row)

            batch_value_prefixs = np.asarray(vp_rows, dtype=np.float64)
            batch_target_values = np.asarray(tv_rows, dtype=np.float64)

        return batch_value_prefixs, batch_target_values

    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the reanalyzed context of policies
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to reanalyzed
        Returns:
            - batch_target_policies_re
        """
        if policy_re_context is None:
            return [], []
        target_type = self._cfg.get('policy_target_type', 'visit_count')
        if target_type not in ('visit_count', 'completed_q'):
            raise ValueError(
                f"Unsupported policy_target_type={target_type!r}; expected 'visit_count' or 'completed_q'."
            )
        use_completed_q = target_type == 'completed_q'
        if use_completed_q and self._cfg.model.continuous_action_space:
            raise NotImplementedError(
                "policy_target_type='completed_q' is currently only supported for discrete action spaces."
            )
        batch_target_policies_re = []

        if len(policy_re_context) == 10:
            (
                policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list,
                child_visits, root_values, game_segment_lens, action_mask_segment,
                to_play_segment, root_sampled_actions_segments
            ) = policy_re_context
        else:
            (
                policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list,
                child_visits, root_values, game_segment_lens, action_mask_segment,
                to_play_segment
            ) = policy_re_context
            root_sampled_actions_segments = None
        # transition_batch_size = game_segment_batch_size * (self._cfg.num_unroll_steps + 1)
        transition_batch_size = len(policy_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)
        k_sampled = self._cfg.model.num_of_sampled_actions
        policy_mask_arr = np.asarray(policy_mask, dtype=np.int64)
        active_indices = np.flatnonzero(policy_mask_arr)
        active_count = int(active_indices.size)

        if self._cfg.model.continuous_action_space:
            root_sampled_actions = np.zeros(
                (transition_batch_size, k_sampled, self._cfg.model.action_space_size), dtype=np.float32
            )
        elif self._cfg.mcts_ctree:
            root_sampled_actions = np.zeros((transition_batch_size, k_sampled, 1), dtype=np.int64)
        else:
            root_sampled_actions = np.zeros((transition_batch_size, k_sampled), dtype=np.int64)

        roots_distributions = [None] * transition_batch_size
        roots_values = [0.0] * transition_batch_size
        roots_child_q_values = [None] * transition_batch_size if use_completed_q else None
        raw_values_pool = [0.0] * transition_batch_size
        policy_logits_pool = [None] * transition_batch_size

        with torch.no_grad():
            if active_count > 0:
                to_play, action_mask = self._preprocess_to_play_and_action_mask(
                    game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
                )
                active_to_play = [to_play[idx] for idx in active_indices]
                if self._cfg.model.continuous_action_space is True:
                    # NOTE: in continuous action space env, we set all legal_actions as -1
                    active_legal_actions = [
                        [-1 for _ in range(self._cfg.model.action_space_size)] for _ in range(active_count)
                    ]
                else:
                    active_legal_actions = [np.nonzero(action_mask[idx])[0].tolist() for idx in active_indices]

                active_obs_list = (
                    policy_obs_list
                    if active_count == transition_batch_size
                    else [policy_obs_list[idx] for idx in active_indices]
                )
                active_obs_arr = prepare_observation(active_obs_list, self._cfg.model.model_type)
                # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
                slices = int(np.ceil(active_count / self._cfg.mini_infer_size))
                network_output = []
                for i in range(slices):
                    beg_index = self._cfg.mini_infer_size * i
                    end_index = self._cfg.mini_infer_size * (i + 1)
                    m_obs = torch.from_numpy(active_obs_arr[beg_index:end_index]).to(self._cfg.device).float()

                    m_output = model.initial_inference(m_obs)

                    if not model.training:
                        # if not in training, obtain the scalars of the value/reward
                        [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                            [
                                m_output.latent_state,
                                inverse_scalar_transform(
                                    m_output.value,
                                    self.value_support,
                                    categorical_distribution=self._cfg.model.categorical_distribution,
                                ),
                                m_output.policy_logits
                            ]
                        )
                        m_output.reward_hidden_state = (
                            m_output.reward_hidden_state[0].detach().cpu().numpy(),
                            m_output.reward_hidden_state[1].detach().cpu().numpy()
                        )

                    network_output.append(m_output)

                active_raw_values, value_prefix_pool, active_policy_logits, latent_state_roots, reward_hidden_state_roots = concat_output(
                    network_output, data_type='efficientzero'
                )

                active_raw_values = np.asarray(active_raw_values).reshape(-1).tolist()
                value_prefix_pool = np.asarray(value_prefix_pool).reshape(-1).tolist()
                active_policy_logits = active_policy_logits.tolist()
                # noises are not necessary for reanalyze
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * k_sampled
                                        ).astype(np.float32).tolist() for _ in range(active_count)
                ] if self._cfg.reanalyze_noise else None
                if self._cfg.mcts_ctree:
                    # ==============================================================
                    # sampled related core code
                    # ==============================================================
                    # cpp mcts_tree
                    roots = MCTSCtree.roots(
                        active_count, active_legal_actions, self._cfg.model.action_space_size,
                        k_sampled, self._cfg.model.continuous_action_space
                    )
                    if self._cfg.reanalyze_noise:
                        roots.prepare(
                            self._cfg.root_noise_weight, noises, value_prefix_pool, active_policy_logits, active_to_play
                        )
                    else:
                        roots.prepare_no_noise(value_prefix_pool, active_policy_logits, active_to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTSCtree(self._cfg).search(
                        roots, model, latent_state_roots, reward_hidden_state_roots, active_to_play
                    )
                else:
                    # python mcts_tree
                    roots = MCTSPtree.roots(
                        active_count, active_legal_actions, self._cfg.model.action_space_size,
                        k_sampled, self._cfg.model.continuous_action_space
                    )
                    if self._cfg.reanalyze_noise:
                        roots.prepare(
                            self._cfg.root_noise_weight, noises, value_prefix_pool, active_policy_logits, active_to_play
                        )
                    else:
                        roots.prepare_no_noise(value_prefix_pool, active_policy_logits, active_to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTSPtree(self._cfg).search(
                        roots, model, latent_state_roots, reward_hidden_state_roots, active_to_play
                    )

                active_distributions = roots.get_distributions()
                active_values = roots.get_values()
                active_child_q_values = (
                    roots.get_children_q_values(float(self._cfg.discount_factor)) if use_completed_q else None
                )
                active_roots_sampled = roots.get_sampled_actions()
                active_sampled_actions = np.asarray([
                    [action.value if hasattr(action, 'value') else action for action in row]
                    for row in active_roots_sampled
                ])
                if active_sampled_actions.shape[1:] != root_sampled_actions.shape[1:]:
                    active_sampled_actions = active_sampled_actions.reshape(
                        (active_count,) + root_sampled_actions.shape[1:]
                    )

                for local_i, global_idx in enumerate(active_indices):
                    roots_distributions[global_idx] = active_distributions[local_i]
                    roots_values[global_idx] = active_values[local_i]
                    raw_values_pool[global_idx] = active_raw_values[local_i]
                    policy_logits_pool[global_idx] = active_policy_logits[local_i]
                    if use_completed_q:
                        roots_child_q_values[global_idx] = active_child_q_values[local_i]
                root_sampled_actions[active_indices] = active_sampled_actions

            compute_completed_q_policy = None
            if use_completed_q:
                from lzero.policy.sampled_efficientzero import compute_completed_q_policy

            seg_lock = self._get_segment_update_lock()
            rsa_segments_iter = (
                root_sampled_actions_segments
                if root_sampled_actions_segments is not None
                else [None] * len(pos_in_game_segment_list)
            )
            policy_index = 0
            for state_index, child_visit, root_value, root_sampled_actions_seg in zip(
                pos_in_game_segment_list, child_visits, root_values, rsa_segments_iter
            ):
                target_policies = []
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]
                    searched_value = roots_values[policy_index]
                    # ==============================================================
                    # sampled related core code
                    # ==============================================================
                    if policy_mask[policy_index] == 0:
                        # NOTE: the invalid padding target policy, O is to make sure the corresponding cross_entropy_loss=0
                        target_policies.append([0 for _ in range(self._cfg.model.num_of_sampled_actions)])
                    else:
                        if distributions is None:
                            # if at some obs, the legal_action is None, then add the fake target_policy
                            target_policies.append(
                                list(
                                    np.ones(self._cfg.model.num_of_sampled_actions) /
                                    self._cfg.model.num_of_sampled_actions
                                )
                            )
                        else:
                            # Update the data in game segment:
                            # after the reanalyze search, new target policies, root values, and sampled actions are obtained.
                            # Update all three atomically under seg_lock so subsequent non-reanalyzed samples never
                            # pair refreshed child_visit distributions with stale root_sampled_actions.
                            if use_completed_q:
                                sampled_row = root_sampled_actions[policy_index]
                                distributions = compute_completed_q_policy(
                                    visit_counts=distributions,
                                    child_q_values=roots_child_q_values[policy_index],
                                    sampled_actions=sampled_row,
                                    policy_logits=policy_logits_pool[policy_index],
                                    raw_value=float(raw_values_pool[policy_index]),
                                    c_visit=float(self._cfg.get('completed_q_c_visit', 50.0)),
                                    c_scale=float(self._cfg.get('completed_q_c_scale', 1.0)),
                                )
                            sim_num = float(sum(distributions))
                            if sim_num > 0:
                                policy = [float(visit_count) / sim_num for visit_count in distributions]
                            else:
                                policy = [1.0 / len(distributions) for _ in distributions]

                            with seg_lock:
                                if root_sampled_actions_seg is not None:
                                    if current_index < len(root_sampled_actions_seg):
                                        existing_sampled = root_sampled_actions_seg[current_index]
                                        new_sampled = np.asarray(root_sampled_actions[policy_index]).copy()
                                        if hasattr(existing_sampled, 'shape') and new_sampled.shape != existing_sampled.shape:
                                            new_sampled = new_sampled.reshape(existing_sampled.shape)
                                        if hasattr(existing_sampled, 'dtype') and np.issubdtype(
                                            existing_sampled.dtype, np.integer
                                        ):
                                            new_sampled = np.rint(new_sampled).astype(existing_sampled.dtype)
                                        child_visit[current_index] = policy
                                        root_value[current_index] = searched_value
                                        root_sampled_actions_seg[current_index] = new_sampled
                                else:
                                    child_visit[current_index] = policy
                                    root_value[current_index] = searched_value

                            target_policies.append(policy)

                    policy_index += 1

                batch_target_policies_re.append(target_policies)

        batch_target_policies_re = np.array(batch_target_policies_re)

        return batch_target_policies_re, root_sampled_actions

    def _compute_target_policy_non_reanalyzed(
            self, policy_non_re_context: List[Any], policy_shape: int
    ) -> np.ndarray:
        """
        Overview:
            Prepare policy targets from the non-reanalyzed context of policies for Sampled EfficientZero.
            This overrides the base class method to handle sampled action spaces correctly.

            In sampled buffers, ``child_visit[current_index]`` is already a K-length distribution
            (visit counts normalized during collection), so we use it directly.
            The base class's approach of indexing via raw action IDs (``policy_tmp[legal_action]``)
            would crash for large/sparse action spaces (e.g. Galcon's ~10k actions) because
            ``policy_shape = num_of_sampled_actions = K`` is only 16 or so.
        Arguments:
            - policy_non_re_context (:obj:`List`): List of policy context
            - policy_shape (:obj:`int`): should be num_of_sampled_actions (K)
        Returns:
            - batch_target_policies_non_re (:obj:`np.ndarray`)
        """
        if policy_non_re_context is None:
            return []

        pos_in_game_segment_list, child_visits, game_segment_lens, action_mask_segment, to_play_segment = policy_non_re_context
        batch_size = len(pos_in_game_segment_list)
        unroll_stride = int(self._cfg.num_unroll_steps) + 1
        zero_policy = [0.0] * policy_shape
        uniform_policy = [1.0 / policy_shape] * policy_shape
        flat_policies = []

        for game_segment_len, child_visit, state_index in zip(
            game_segment_lens, child_visits, pos_in_game_segment_list
        ):
            max_k = game_segment_len - state_index
            if max_k > unroll_stride:
                max_k = unroll_stride
            elif max_k < 0:
                max_k = 0
            for k in range(max_k):
                distributions = child_visit[state_index + k]
                if distributions is None or len(distributions) == 0:
                    flat_policies.append(uniform_policy)
                else:
                    flat_policies.append(distributions)
            if max_k < unroll_stride:
                flat_policies.extend([zero_policy] * (unroll_stride - max_k))

        return np.asarray(flat_policies, dtype=np.float64).reshape(batch_size, unroll_stride, policy_shape)

    def update_priority(self, train_data: List[np.ndarray], batch_priorities: Any) -> None:
        """
        Overview:
            Update the priority of training data.
        Arguments:
            - train_data (:obj:`Optional[List[Optional[np.ndarray]]]`): training data to be updated priority.
            - batch_priorities (:obj:`batch_priorities`): priorities to update to.
        NOTE:
            train_data = [current_batch, target_batch]
            current_batch = [obs_list, action_list, root_sampled_actions_list, mask_list, batch_index_list, weights_list, make_time_list]
        """

        batch_index_list = train_data[0][4]
        metas = {'make_time': train_data[0][6], 'batch_priorities': batch_priorities}
        # only update the priorities for data still in replay buffer
        for i in range(len(batch_index_list)):
            if metas['make_time'][i] > self.clear_time:
                idx, prio = batch_index_list[i], metas['batch_priorities'][i]
                self.game_pos_priorities[idx] = prio
