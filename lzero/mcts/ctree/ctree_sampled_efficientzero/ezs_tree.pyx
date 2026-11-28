# distutils:language=c++
# cython:language_level=3
from libcpp.vector cimport vector
import numpy as np

cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def __dealloc__(self):
        del self.cmin_max_stats_lst

cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)

    def get_search_len(self):
        return self.cresults.search_lens

cdef class Action:
    cdef int is_root_action
    cdef vector[float] value
    cdef CAction action

    def __cinit__(self):
        pass

    def __cinit__(self, vector[float] value, int is_root_action):
        self.is_root_action = is_root_action
        self.value = value

cdef class Roots:
    cdef int root_num
    cdef int action_space_size
    cdef int num_of_sampled_actions
    cdef CRoots *roots
    cdef bool continuous_action_space

    def __cinit__(self):
        pass

    def __cinit__(self, int root_num, list legal_actions_list, int action_space_size, int num_of_sampled_actions,
                  bool continuous_action_space):
        #def __cinit__(self, int root_num, list legal_actions_list, int action_space_size, int num_of_sampled_actions):
        self.root_num = root_num
        self.action_space_size = action_space_size
        self.num_of_sampled_actions = num_of_sampled_actions
        self.roots = new CRoots(root_num, legal_actions_list, action_space_size, num_of_sampled_actions,
                                continuous_action_space)

    def prepare(self, float root_noise_weight, list noises, list value_prefix_pool, list policy_logits_pool,
                vector[int] & to_play_batch):
        self.roots[0].prepare(root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play_batch)

    def prepare_no_noise(self, list value_prefix_pool, list policy_logits_pool, vector[int] & to_play_batch):
        self.roots[0].prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play_batch)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_children_q_values(self, float discount_factor):
        return self.roots[0].get_children_q_values(discount_factor)

    def get_sampled_actions(self):
        return self.roots[0].get_sampled_actions()

    def get_values(self):
        return self.roots[0].get_values()

    def clear(self):
        self.roots[0].clear()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num

cdef class Node:
    cdef CNode cnode
    cdef bool continuous_action_space

    def __cinit__(self):
        pass

    #def __cinit__(self, float prior, vector[int] &legal_actions, int action_space_size, int num_of_sampled_actions):
    def __cinit__(self, float prior, vector[int] & legal_actions, int action_space_size, int num_of_sampled_actions,
                  bool continuous_action_space):
        pass

    def expand(self, int to_play, int current_latent_state_index, int batch_index, float value_prefix,
               list policy_logits):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, current_latent_state_index, batch_index, value_prefix, cpolicy)

def batch_backpropagate(int current_latent_state_index, float discount_factor, object value_prefixs, object values, object policies,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results, object is_reset_list,
                         list to_play_batch):
    cdef const float[:] vp_mv
    cdef const float[:] val_mv
    cdef const float[:, :] pol_mv
    cdef const int[:] rst_mv
    cdef vector[int] cto_play
    cdef int n = results.cresults.num
    cdef int vp_stride
    cdef int val_stride
    cdef int pol_row_stride
    cdef int rst_stride
    cdef int pol_dim

    if (
        isinstance(policies, np.ndarray)
        and policies.ndim == 2
        and policies.dtype == np.float32
        and policies.strides[1] == sizeof(float)
        and isinstance(value_prefixs, np.ndarray)
        and value_prefixs.ndim == 1
        and value_prefixs.dtype == np.float32
        and isinstance(values, np.ndarray)
        and values.ndim == 1
        and values.dtype == np.float32
        and isinstance(is_reset_list, np.ndarray)
        and is_reset_list.ndim == 1
        and is_reset_list.dtype == np.int32
        and n > 0
        and policies.shape[0] >= n
        and policies.shape[1] > 0
        and value_prefixs.shape[0] >= n
        and values.shape[0] >= n
        and is_reset_list.shape[0] >= n
    ):
        vp_mv = value_prefixs
        val_mv = values
        pol_mv = policies
        rst_mv = is_reset_list
        cto_play = to_play_batch
        vp_stride = vp_mv.strides[0] // sizeof(float)
        val_stride = val_mv.strides[0] // sizeof(float)
        pol_row_stride = pol_mv.strides[0] // sizeof(float)
        rst_stride = rst_mv.strides[0] // sizeof(int)
        pol_dim = pol_mv.shape[1]
        cbatch_backpropagate_strided(
            current_latent_state_index,
            discount_factor,
            &vp_mv[0],
            vp_stride,
            &val_mv[0],
            val_stride,
            &pol_mv[0, 0],
            pol_dim,
            pol_row_stride,
            min_max_stats_lst.cmin_max_stats_lst,
            results.cresults,
            &rst_mv[0],
            rst_stride,
            cto_play,
        )
        return

    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies
    cdef vector[int] cis_reset = is_reset_list
    cdef vector[int] cto_play_fallback = to_play_batch

    cbatch_backpropagate(current_latent_state_index, discount_factor, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, cis_reset, cto_play_fallback)

def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results, list virtual_to_play_batch, bool continuous_action_space):
    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst.cmin_max_stats_lst, results.cresults,
                    virtual_to_play_batch, continuous_action_space)

    return results.cresults.latent_state_index_in_search_path, results.cresults.latent_state_index_in_batch, results.cresults.last_actions, results.cresults.virtual_to_play_batchs
