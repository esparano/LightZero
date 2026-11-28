// C++11

#include <iostream>
#include "cnode.h"
#include <algorithm>
#include <map>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <stack>
#include <math.h>

#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <time.h>
#include <cassert>

#ifdef _WIN32
#include "..\..\common_lib\utils.cpp"
#else
#include "../../common_lib/utils.cpp"
#endif



template <class T>
size_t hash_combine(std::size_t &seed, const T &val)
{
    /*
    Overview:
        Combines a hash value with a new value using a bitwise XOR and a rotation.
        This function is used to create a hash value for multiple values.
    Arguments:
        - seed The current hash value to be combined with.
        - val The new value to be hashed and combined with the seed.
    */
    std::hash<T> hasher;  // Create a hash object for the new value.
    seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  // Combine the new hash value with the seed.
    return seed;
}

// Sort by the value of second in descending order.
static inline bool cmp(const std::pair<int, double> &x, const std::pair<int, double> &y)
{
    return x.second > y.second;
}

// Min-heap comparator (keeps smallest second at heap front for top-K tracking).
static inline bool cmp_min_heap(const std::pair<int, double> &x, const std::pair<int, double> &y)
{
    return x.second > y.second;
}

static std::mt19937_64 &get_expand_rng()
{
    thread_local std::mt19937_64 generator(
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())
        ^ (static_cast<uint64_t>(std::random_device{}()) << 1)
    );
    return generator;
}

namespace tree
{
    //*********************************************************

    CAction::CAction()
    {
        /*
        Overview:
            Initialization of CAction. Parameterized constructor.
        */
        this->is_root_action = 0;
        this->cached_combined_hash = 0;
        this->has_cached_hash = false;
    }

    CAction::CAction(std::vector<float> value, int is_root_action)
    {
        /*
        Overview:
            Initialization of CAction with value and is_root_action. Default constructor.
        Arguments:
            - value: a multi-dimensional action.
            - is_root_action: whether value is a root node.
        */
        this->value = std::move(value);
        this->is_root_action = is_root_action;
        this->cached_combined_hash = 0;
        this->has_cached_hash = false;
    }

    CAction::~CAction() {} // Destructors.

    std::vector<size_t> CAction::get_hash(void) const
    {
        /*
        Overview:
            get a hash value for each dimension in the multi-dimensional action.
        */
        std::vector<size_t> hash;
        hash.reserve(this->value.size());
        for (size_t i = 0; i < this->value.size(); ++i)
        {
            std::size_t hash_i = std::hash<std::string>()(std::to_string(this->value[i]));
            hash.push_back(hash_i);
        }
        return hash;
    }
    size_t CAction::get_combined_hash(void) const
    {
        /*
        Overview:
            Get the final combined hash value from the hash values of each dimension of the multi-dimensional action.
        */
        if (this->has_cached_hash)
        {
            return this->cached_combined_hash;
        }
        std::vector<size_t> hash = this->get_hash();
        size_t combined_hash = hash[0];

        if (hash.size() >= 1)
        {
            for (size_t i = 1; i < hash.size(); ++i)
            {
                combined_hash = hash_combine(combined_hash, hash[i]);
            }
        }

        this->cached_combined_hash = combined_hash;
        this->has_cached_hash = true;
        return combined_hash;
    }

    //*********************************************************

    CSearchResults::CSearchResults()
    {
        /*
        Overview:
            Initialization of CSearchResults, the default result number is set to 0.
        */
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num)
    {
        /*
        Overview:
            Initialization of CSearchResults with result number.
        */
        this->num = num;
        for (int i = 0; i < num; ++i)
        {
            this->search_paths.push_back(std::vector<CNode *>());
        }
    }

    CSearchResults::~CSearchResults() {}

    //*********************************************************

    CNode::CNode()
    {
        /*
        Overview:
            Initialization of CNode.
        */
        this->prior = 0;
        this->action_space_size = 9;
        this->num_of_sampled_actions = 20;
        this->continuous_action_space = false;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        CAction best_action;
        this->best_action = best_action;

        this->to_play = 0;
        this->value_prefix = 0.0;
        this->parent_value_prefix = 0.0;
    }

    CNode::CNode(float prior, std::vector<CAction> &legal_actions, int action_space_size, int num_of_sampled_actions, bool continuous_action_space)
    {
        /*
        Overview:
            Initialization of CNode with prior, legal actions, action_space_size, num_of_sampled_actions, continuous_action_space.
        Arguments:
            - prior: the prior value of this node.
            - legal_actions: a vector of legal actions of this node.
            - action_space_size: the size of action space of the current env.
            - num_of_sampled_actions: the number of sampled actions, i.e. K in the Sampled MuZero papers.
            - continuous_action_space: whether the action space is continous in current env.
        */
        this->prior = prior;
        this->legal_actions = legal_actions;

        this->action_space_size = action_space_size;
        this->num_of_sampled_actions = num_of_sampled_actions;
        this->continuous_action_space = continuous_action_space;
        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->to_play = 0;
        this->value_prefix = 0.0;
        this->parent_value_prefix = 0.0;
        this->current_latent_state_index = -1;
        this->batch_index = -1;
    }

    CNode::~CNode() {}


    void CNode::expand(int to_play, int current_latent_state_index, int batch_index, float value_prefix, const std::vector<float> &policy_logits)
    {
        this->expand_ptr(
            to_play,
            current_latent_state_index,
            batch_index,
            value_prefix,
            policy_logits.data(),
            static_cast<int>(policy_logits.size())
        );
    }

    void CNode::expand_ptr(int to_play, int current_latent_state_index, int batch_index, float value_prefix, const float *policy_logits, int policy_len)
    {
        /*
        Overview:
            Expand the child nodes of the current node from a contiguous policy_logits buffer.
        Arguments:
            - to_play: which player to play the game in the current node.
            - current_latent_state_index: the x/first index of hidden state vector of the current node, i.e. the search depth.
            - batch_index: the y/second index of hidden state vector of the current node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
            - value_prefix: the value prefix of the current node.
            - policy_logits: contiguous pointer to the policy logits of the child nodes.
            - policy_len: number of floats in policy_logits.
        */
        this->to_play = to_play;
        this->current_latent_state_index = current_latent_state_index;
        this->batch_index = batch_index;
        this->value_prefix = value_prefix;

        std::vector<std::vector<float> > sampled_actions_after_tanh;
        std::vector<float> sampled_actions_log_probs_after_tanh;

        std::vector<int> sampled_actions;
        std::vector<float> sampled_actions_probs;

        if (this->continuous_action_space == true)
        {
            // continuous action space for sampled algo..
            this->action_space_size = policy_len / 2;
            std::vector<float> mu(this->action_space_size);
            std::vector<float> sigma(this->action_space_size);
            for (int i = 0; i < this->action_space_size; ++i)
            {
                mu[i] = policy_logits[i];
                sigma[i] = policy_logits[this->action_space_size + i];
            }

            std::mt19937_64 &generator = get_expand_rng();
            sampled_actions_after_tanh.reserve(this->num_of_sampled_actions);
            sampled_actions_log_probs_after_tanh.reserve(this->num_of_sampled_actions);

            for (int i = 0; i < this->num_of_sampled_actions; ++i)
            {
                float sampled_action_prob_before_tanh = 1.0f;
                std::vector<float> sampled_action_after_tanh;
                sampled_action_after_tanh.reserve(this->action_space_size);
                float y_sum = 0.0f;

                for (int j = 0; j < this->action_space_size; ++j)
                {
                    std::normal_distribution<float> distribution(mu[j], sigma[j]);
                    float sampled_action_one_dim_before_tanh = distribution(generator);
                    float diff = sampled_action_one_dim_before_tanh - mu[j];
                    sampled_action_prob_before_tanh *= exp(-(diff * diff) / (2.0f * sigma[j] * sigma[j]) - log(sigma[j]) - log(sqrt(2.0f * M_PI)));
                    float t = tanh(sampled_action_one_dim_before_tanh);
                    sampled_action_after_tanh.push_back(t);
                    y_sum += (1.0f - t * t + 1e-6f);
                }
                sampled_actions_after_tanh.push_back(std::move(sampled_action_after_tanh));
                sampled_actions_log_probs_after_tanh.push_back(log(sampled_action_prob_before_tanh) - log(y_sum));
            }
        }
        else
        {
            // discrete action space for sampled algo (weighted sampling without replacement).
            const bool has_legal = !this->legal_actions.empty();
            const int n_cand = has_legal ? static_cast<int>(this->legal_actions.size()) : this->action_space_size;
            assert(n_cand > 0);
            assert(policy_len == this->action_space_size || policy_len == n_cand);

            std::vector<int> candidate_actions;
            std::vector<float> probs(n_cand);
            float max_logit = 0.0f;

            if (!has_legal)
            {
                // Fast path for non-root expansions: candidate actions are 0..n_cand-1.
                max_logit = policy_logits[0];
                for (int i = 1; i < n_cand; ++i)
                {
                    if (policy_logits[i] > max_logit)
                    {
                        max_logit = policy_logits[i];
                    }
                }
                float logits_exp_sum = 0.0f;
                for (int i = 0; i < n_cand; ++i)
                {
                    float e = exp(policy_logits[i] - max_logit);
                    probs[i] = e;
                    logits_exp_sum += e;
                }
                const float denom = logits_exp_sum + 1e-6f;
                for (int i = 0; i < n_cand; ++i)
                {
                    probs[i] /= denom;
                }
            }
            else
            {
                candidate_actions.resize(n_cand);
                const bool full_logits = (policy_len == this->action_space_size);
                for (int i = 0; i < n_cand; ++i)
                {
                    int action = static_cast<int>(this->legal_actions[i].value[0]);
                    candidate_actions[i] = action;
                    float logit = full_logits ? policy_logits[action] : policy_logits[i];
                    probs[i] = logit;
                    if (i == 0 || logit > max_logit)
                    {
                        max_logit = logit;
                    }
                }
                float logits_exp_sum = 0.0f;
                for (int i = 0; i < n_cand; ++i)
                {
                    float e = exp(probs[i] - max_logit);
                    probs[i] = e;
                    logits_exp_sum += e;
                }
                const float denom = logits_exp_sum + 1e-6f;
                for (int i = 0; i < n_cand; ++i)
                {
                    probs[i] /= denom;
                }
            }

            const int num_sampled = std::min(this->num_of_sampled_actions, n_cand);
            sampled_actions.reserve(num_sampled);
            sampled_actions_probs.reserve(num_sampled);

            if (num_sampled == 1 && n_cand == 1)
            {
                sampled_actions.push_back(has_legal ? candidate_actions[0] : 0);
                sampled_actions_probs.push_back(probs[0]);
            }
            else
            {
                // Log-space Efraimidis-Spirakis weighted sampling without replacement:
                // key = log(u^(1/p)) = log(u) / p is strictly monotonic with u^(1/p) without
                // double-precision underflow on large action spaces (e.g. A=4097, p ~ 2.4e-4).
                // Track only the top-K candidates in a K-element min-heap (O(A + K log K)).
                std::mt19937_64 &generator = get_expand_rng();
                std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
                std::vector<std::pair<int, double> > topk;
                topk.reserve(num_sampled);

                for (int i = 0; i < n_cand; ++i)
                {
                    double u = uniform_distribution(generator);
                    if (u <= 1e-300)
                    {
                        u = 1e-300;
                    }
                    double p = static_cast<double>(probs[i]);
                    double key = (p > 1e-30) ? (std::log(u) / p) : (-1e300);

                    if (i < num_sampled)
                    {
                        topk.push_back(std::make_pair(i, key));
                        if (i == num_sampled - 1)
                        {
                            std::make_heap(topk.begin(), topk.end(), cmp_min_heap);
                        }
                    }
                    else if (key > topk[0].second)
                    {
                        std::pop_heap(topk.begin(), topk.end(), cmp_min_heap);
                        topk.back() = std::make_pair(i, key);
                        std::push_heap(topk.begin(), topk.end(), cmp_min_heap);
                    }
                }

                std::sort(topk.begin(), topk.end(), cmp);

                for (int k = 0; k < num_sampled; ++k)
                {
                    int idx = topk[k].first;
                    sampled_actions.push_back(has_legal ? candidate_actions[idx] : idx);
                    sampled_actions_probs.push_back(probs[idx]);
                }
            }
        }

        this->legal_actions.clear();
        this->sampled_actions.clear();

        if (this->continuous_action_space == true)
        {
            this->legal_actions.reserve(this->num_of_sampled_actions);
            this->sampled_actions.reserve(this->num_of_sampled_actions);
            std::vector<CAction> empty_legal;
            for (int i = 0; i < this->num_of_sampled_actions; ++i)
            {
                CAction action(sampled_actions_after_tanh[i], 0);
                this->children[action.get_combined_hash()] = CNode(sampled_actions_log_probs_after_tanh[i], empty_legal, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space);
                this->legal_actions.push_back(action);
                this->sampled_actions.push_back(std::move(action));
            }
        }
        else
        {
            this->legal_actions.reserve(sampled_actions.size());
            this->sampled_actions.reserve(this->num_of_sampled_actions);
            std::vector<CAction> empty_legal;
            for (size_t i = 0; i < sampled_actions.size(); ++i)
            {
                std::vector<float> sampled_action_tmp(1, static_cast<float>(sampled_actions[i]));
                CAction action(std::move(sampled_action_tmp), 0);
                this->children[action.get_combined_hash()] = CNode(sampled_actions_probs[i], empty_legal, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space);
                this->legal_actions.push_back(action);
                this->sampled_actions.push_back(std::move(action));
            }
            while (!this->sampled_actions.empty() && static_cast<int>(this->sampled_actions.size()) < this->num_of_sampled_actions)
            {
                this->sampled_actions.push_back(this->sampled_actions.back());
            }
        }
    }

    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises)
    {
        /*
        Overview:
            Add a noise to the prior of the child nodes.
        Arguments:
            - exploration_fraction: the fraction to add noise.
            - noises: the vector of noises added to each child node.
        */
        float noise, prior;
        int noise_num = std::min(int(this->legal_actions.size()), int(noises.size()));
        for (int i = 0; i < noise_num; ++i)
        {

            noise = noises[i];
            CNode *child = this->get_child(this->legal_actions[i]);
            prior = child->prior;
            if (this->continuous_action_space == true)
            {
                // if prior is log_prob
                child->prior = log(exp(prior) * (1 - exploration_fraction) + noise * exploration_fraction + 1e-6);
            }
            else
            {
                // if prior is prob
                child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
            }
        }
    }

    float CNode::compute_mean_q(int isRoot, float parent_q, float discount_factor)
    {
        /*
        Overview:
            Compute the mean q value of the current node.
        Arguments:
            - isRoot: whether the current node is a root node.
            - parent_q: the q value of the parent node.
            - discount_factor: the discount_factor of reward.
        */
        float total_unsigned_q = 0.0;
        int total_visits = 0;
        float parent_value_prefix = this->value_prefix;
        for (const auto &a : this->legal_actions)
        {
            CNode *child = this->get_child(a);
            if (child->visit_count > 0)
            {
                float true_reward = child->value_prefix - parent_value_prefix;
                if (this->is_reset == 1)
                {
                    true_reward = child->value_prefix;
                }
                float qsa = true_reward + discount_factor * child->value();
                total_unsigned_q += qsa;
                total_visits += 1;
            }
        }

        float mean_q = 0.0;
        if (isRoot && total_visits > 0)
        {
            mean_q = (total_unsigned_q) / (total_visits);
        }
        else
        {
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
        }
        return mean_q;
    }

    void CNode::print_out()
    {
        return;
    }

    int CNode::expanded()
    {
        /*
        Overview:
            Return whether the current node is expanded.
        */
        return this->children.size() > 0;
    }

    float CNode::value()
    {
        /*
        Overview:
            Return the real value of the current tree.
        */
        float true_value = 0.0;
        if (this->visit_count == 0)
        {
            return true_value;
        }
        else
        {
            true_value = this->value_sum / this->visit_count;
            return true_value;
        }
    }

    std::vector<std::vector<float> > CNode::get_trajectory()
    {
        /*
        Overview:
            Find the current best trajectory starts from the current node.
        Returns:
            - traj: a vector of node index, which is the current best trajectory from this node.
        */
        std::vector<CAction> traj;

        CNode *node = this;
        CAction best_action = node->best_action;
        while (best_action.is_root_action != 1)
        {
            traj.push_back(best_action);
            node = node->get_child(best_action);
            best_action = node->best_action;
        }

        std::vector<std::vector<float> > traj_return;
        for (int i = 0; i < traj.size(); ++i)
        {
            traj_return.push_back(traj[i].value);
        }
        return traj_return;
    }

    std::vector<int> CNode::get_children_distribution()
    {
        /*
        Overview:
            Get the distribution of child nodes in the format of visit_count.
        Returns:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        */
        std::vector<int> distribution;
        if (this->expanded())
        {
            for (auto a : this->legal_actions)
            {
                CNode *child = this->get_child(a);
                distribution.push_back(child->visit_count);
            }
            while (distribution.size() < this->sampled_actions.size())
            {
                distribution.push_back(0);
            }
        }
        return distribution;
    }

    std::vector<float> CNode::get_children_q_values(float discount_factor)
    {
        /*
        Overview:
            Get the minimax-aware Q values for each unique sampled child in legal_actions.
            Unvisited children emit 0.0f; pair with get_children_distribution() to check visit_count > 0.
        Returns:
            - q_values: a vector of Q(root, a) values aligned with the unique prefix of sampled_actions.
        */
        std::vector<float> q_values;
        if (this->expanded())
        {
            q_values.reserve(this->legal_actions.size());
            int players = (this->to_play == -1) ? 1 : 2;
            for (auto a : this->legal_actions)
            {
                CNode *child = this->get_child(a);
                if (child->visit_count == 0)
                {
                    q_values.push_back(0.0f);
                }
                else
                {
                    float true_reward = child->value_prefix - this->value_prefix;
                    if (this->is_reset == 1)
                    {
                        true_reward = child->value_prefix;
                    }
                    if (players == 1)
                    {
                        q_values.push_back(true_reward + discount_factor * child->value());
                    }
                    else
                    {
                        q_values.push_back(true_reward + discount_factor * (-child->value()));
                    }
                }
            }
        }
        return q_values;
    }

    CNode *CNode::get_child(const CAction &action)
    {
        /*
        Overview:
            Get the child node corresponding to the input action.
        Arguments:
            - action: the action to get child.
        */
        return &(this->children[action.get_combined_hash()]);
    }

    //*********************************************************

    CRoots::CRoots()
    {
        this->root_num = 0;
        this->num_of_sampled_actions = 20;
    }

    CRoots::CRoots(int root_num, std::vector<std::vector<float> > legal_actions_list, int action_space_size, int num_of_sampled_actions, bool continuous_action_space)
    {
        /*
        Overview:
            Initialization of CNode with root_num, legal_actions_list, action_space_size, num_of_sampled_actions, continuous_action_space.
        Arguments:
            - root_num: the number of the current root.
            - legal_action_list: the vector of the legal action of this root.
            - action_space_size: the size of action space of the current env.
            - num_of_sampled_actions: the number of sampled actions, i.e. K in the Sampled MuZero papers.
            - continuous_action_space: whether the action space is continous in current env.
        */
        this->root_num = root_num;
        this->legal_actions_list = legal_actions_list;
        this->continuous_action_space = continuous_action_space;

        // sampled related core code
        this->num_of_sampled_actions = num_of_sampled_actions;
        this->action_space_size = action_space_size;

        for (int i = 0; i < this->root_num; ++i)
        {
            if (this->continuous_action_space == true)
            {
                // continous action space
                std::vector<CAction> legal_actions;
                this->roots.push_back(CNode(0, legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));
            }
            else if (this->legal_actions_list.size() <= i || this->legal_actions_list[i].empty() || this->legal_actions_list[i][0] == -1)
            {
                // sampled
                // discrete action space without action mask
                std::vector<CAction> legal_actions;
                this->roots.push_back(CNode(0, legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));
            }

            else
            {
                // TODO(pu): discrete action space
                std::vector<CAction> c_legal_actions;
                for (int j = 0; j < this->legal_actions_list[i].size(); ++j)
                {
                    std::vector<float> legal_action;
                    legal_action.push_back(this->legal_actions_list[i][j]);
                    CAction c_legal_action = CAction(legal_action, 0);
                    c_legal_actions.push_back(c_legal_action);
                }
                this->roots.push_back(CNode(0, c_legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));
            }
        }
    }

    CRoots::~CRoots() {}

    void CRoots::prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the roots and add noises.
        Arguments:
            - root_noise_weight: the exploration fraction of roots
            - noises: the vector of noise add to the roots.
            - value_prefixs: the vector of value prefixs of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        */

        // sampled related core code
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, value_prefixs[i], policies[i]);
            this->roots[i].add_exploration_noise(root_noise_weight, noises[i]);
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the roots without noise.
        Arguments:
            - value_prefixs: the vector of value prefixs of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        */
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, value_prefixs[i], policies[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear()
    {
        this->roots.clear();
    }

    std::vector<std::vector<std::vector<float> > > CRoots::get_trajectories()
    {
        /*
        Overview:
            Find the current best trajectory starts from each root.
        Returns:
            - traj: a vector of node index, which is the current best trajectory from each root.
        */
        std::vector<std::vector<std::vector<float> > > trajs;
        trajs.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int> > CRoots::get_distributions()
    {
        /*
        Overview:
            Get the children distribution of each root.
        Returns:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        */
        std::vector<std::vector<int> > distributions;
        distributions.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<std::vector<float> > CRoots::get_children_q_values(float discount_factor)
    {
        std::vector<std::vector<float> > q_values_list;
        q_values_list.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            q_values_list.push_back(this->roots[i].get_children_q_values(discount_factor));
        }
        return q_values_list;
    }

    // sampled related core code
    std::vector<std::vector<std::vector<float> > > CRoots::get_sampled_actions()
    {
        /*
        Overview:
            Get the sampled_actions of each root.
        Returns:
            - python_sampled_actions: a vector of sampled_actions for each root, e.g. the size of original action space is 6, the K=3, 
            python_sampled_actions = [[1,3,0], [2,4,0], [5,4,1]].
        */
        std::vector<std::vector<CAction> > sampled_actions;
        std::vector<std::vector<std::vector<float> > > python_sampled_actions;

        //  sampled_actions.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            std::vector<CAction> sampled_action;
            sampled_action = this->roots[i].sampled_actions;
            std::vector<std::vector<float> > python_sampled_action;

            for (int j = 0; j < sampled_action.size(); ++j)
            {
                python_sampled_action.push_back(sampled_action[j].value);
            }
            python_sampled_actions.push_back(python_sampled_action);
        }

        return python_sampled_actions;
    }

    std::vector<float> CRoots::get_values()
    {
        /*
        Overview:
            Return the estimated value of each root.
        */
        std::vector<float> values;
        for (int i = 0; i < this->root_num; ++i)
        {
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    //*********************************************************
    //
    void update_tree_q(CNode *root, tools::CMinMaxStats &min_max_stats, float discount_factor, int players)
    {
        /*
        Overview:
            Update the q value of the root and its child nodes.
        Arguments:
            - root: the root that update q value from.
            - min_max_stats: a tool used to min-max normalize the q value.
            - discount_factor: the discount factor of reward.
            - players: the number of players.
        */
        std::stack<CNode *> node_stack;
        node_stack.push(root);
        float parent_value_prefix = 0.0;
        int is_reset = 0;
        while (node_stack.size() > 0)
        {
            CNode *node = node_stack.top();
            node_stack.pop();

            if (node != root)
            {
                // NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                // true_reward = node.value_prefix - (- parent_value_prefix)
                float true_reward = node->value_prefix - node->parent_value_prefix;

                if (is_reset == 1)
                {
                    true_reward = node->value_prefix;
                }
                float qsa;
                if (players == 1)
                    qsa = true_reward + discount_factor * node->value();
                else if (players == 2)
                    // TODO(pu): why only the last reward multiply the discount_factor?
                    qsa = true_reward + discount_factor * (-1) * node->value();

                min_max_stats.update(qsa);
            }

            for (auto a : node->legal_actions)
            {
                CNode *child = node->get_child(a);
                if (child->expanded())
                {
                    child->parent_value_prefix = node->value_prefix;
                    node_stack.push(child);
                }
            }

            is_reset = node->is_reset;
        }
    }

    void cbackpropagate(std::vector<CNode *> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount_factor)
    {
        /*
        Overview:
            Update the value sum and visit count of nodes along the search path.
        Arguments:
            - search_path: a vector of nodes on the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - to_play: which player to play the game in the current node.
            - value: the value to propagate along the search path.
            - discount_factor: the discount factor of reward.
        */
        assert(to_play == -1 || to_play == 1 || to_play == 2);
        if (to_play == -1)
        {
            // for play-with-bot-mode
            float bootstrap_value = value;
            int path_len = search_path.size();
            for (int i = path_len - 1; i >= 0; --i)
            {
                CNode *node = search_path[i];
                node->value_sum += bootstrap_value;
                node->visit_count += 1;

                float parent_value_prefix = 0.0;
                int is_reset = 0;
                if (i >= 1)
                {
                    CNode *parent = search_path[i - 1];
                    parent_value_prefix = parent->value_prefix;
                    is_reset = parent->is_reset;
                }

                float true_reward = node->value_prefix - parent_value_prefix;
                min_max_stats.update(true_reward + discount_factor * node->value());

                if (is_reset == 1)
                {
                    // parent is reset.
                    true_reward = node->value_prefix;
                }

                bootstrap_value = true_reward + discount_factor * bootstrap_value;
            }
        }
        else
        {
            // for self-play-mode
            float bootstrap_value = value;
            int path_len = search_path.size();
            for (int i = path_len - 1; i >= 0; --i)
            {
                CNode *node = search_path[i];
                if (node->to_play == to_play)
                    node->value_sum += bootstrap_value;
                else
                    node->value_sum += -bootstrap_value;
                node->visit_count += 1;

                float parent_value_prefix = 0.0;
                int is_reset = 0;
                if (i >= 1)
                {
                    CNode *parent = search_path[i - 1];
                    parent_value_prefix = parent->value_prefix;
                    is_reset = parent->is_reset;
                }

                // NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                float true_reward = node->value_prefix - parent_value_prefix;

                min_max_stats.update(true_reward + discount_factor * node->value());

                if (is_reset == 1)
                {
                    // parent is reset.
                    true_reward = node->value_prefix;
                }
                if (node->to_play == to_play)
                    bootstrap_value = -true_reward + discount_factor * bootstrap_value;
                else
                    bootstrap_value = true_reward + discount_factor * bootstrap_value;
            }
        }
    }

    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_list, std::vector<int> &to_play_batch)
    {        
        /*
        Overview:
            Expand the nodes along the search path and update the infos.
        Arguments:
            - current_latent_state_index: The index of latent state of the leaf node in the search path.
            - discount_factor: the discount factor of reward.
            - value_prefixs: the value prefixs of nodes along the search path.
            - values: the values to propagate along the search path.
            - policies: the policy logits of nodes along the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - results: the search results.
            - is_reset_list: the vector of is_reset nodes along the search path, where is_reset represents for whether the parent value prefix needs to be reset.
            - to_play_batch: the batch of which player is playing on this node.
        */
        for (int i = 0; i < results.num; ++i)
        {
            results.nodes[i]->expand(to_play_batch[i], current_latent_state_index, i, value_prefixs[i], policies[i]);
            // reset
            results.nodes[i]->is_reset = is_reset_list[i];

            cbackpropagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[i], discount_factor);
        }
    }

    void cbatch_backpropagate_strided(
        int current_latent_state_index,
        float discount_factor,
        const float *value_prefixs,
        int vp_stride,
        const float *values,
        int val_stride,
        const float *policies,
        int policy_dim,
        int policy_row_stride,
        tools::CMinMaxStatsList *min_max_stats_lst,
        CSearchResults &results,
        const int *is_reset_list,
        int reset_stride,
        const std::vector<int> &to_play_batch
    )
    {
        for (int i = 0; i < results.num; ++i)
        {
            results.nodes[i]->expand_ptr(
                to_play_batch[i],
                current_latent_state_index,
                i,
                value_prefixs[i * vp_stride],
                policies + static_cast<size_t>(i) * static_cast<size_t>(policy_row_stride),
                policy_dim
            );
            results.nodes[i]->is_reset = is_reset_list[i * reset_stride];
            cbackpropagate(
                results.search_paths[i],
                min_max_stats_lst->stats_lst[i],
                to_play_batch[i],
                values[i * val_stride],
                discount_factor
            );
        }
    }

    static inline float cucb_score_precomputed(
        CNode *child,
        tools::CMinMaxStats &min_max_stats,
        float parent_mean_q,
        int is_reset,
        float pb_c_base_factor,
        float empirical_prob_sum,
        float parent_value_prefix,
        float discount_factor,
        int players,
        bool continuous_action_space
    )
    {
        float pb_c = pb_c_base_factor / (child->visit_count + 1);
        float prior_score = 0.0f;
        if (continuous_action_space == true)
        {
            prior_score = pb_c * exp(child->prior) / (empirical_prob_sum + 1e-6f);
        }
        else
        {
            prior_score = pb_c * child->prior / (empirical_prob_sum + 1e-6f);
        }

        float value_score = 0.0f;
        if (child->visit_count == 0)
        {
            value_score = parent_mean_q;
        }
        else
        {
            float true_reward = child->value_prefix - parent_value_prefix;
            if (is_reset == 1)
            {
                true_reward = child->value_prefix;
            }
            if (players == 1)
                value_score = true_reward + discount_factor * child->value();
            else if (players == 2)
                value_score = true_reward + discount_factor * (-child->value());
        }

        value_score = min_max_stats.normalize(value_score);
        if (value_score < 0.0f)
            value_score = 0.0f;
        if (value_score > 1.0f)
            value_score = 1.0f;

        return prior_score + value_score;
    }

    CAction cselect_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players, bool continuous_action_space)
    {
        /*
        Overview:
            Select the child node of the roots according to ucb scores.
        */
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001f;
        const size_t n_children = root->legal_actions.size();

        // Single pass to gather child pointers and compute empirical_prob_sum once (O(K) instead of O(K^2)).
        std::vector<CNode *> child_ptrs;
        child_ptrs.reserve(n_children);
        float empirical_prob_sum = 0.0f;
        if (continuous_action_space == true)
        {
            for (size_t i = 0; i < n_children; ++i)
            {
                CNode *child = root->get_child(root->legal_actions[i]);
                child_ptrs.push_back(child);
                empirical_prob_sum += exp(child->prior);
            }
        }
        else
        {
            for (size_t i = 0; i < n_children; ++i)
            {
                CNode *child = root->get_child(root->legal_actions[i]);
                child_ptrs.push_back(child);
                empirical_prob_sum += child->prior;
            }
        }

        const float total_children_visit_counts = static_cast<float>(root->visit_count - 1);
        const float pb_c_base_factor =
            (log((total_children_visit_counts + pb_c_base + 1.0f) / pb_c_base) + pb_c_init)
            * sqrt(total_children_visit_counts);

        std::vector<size_t> max_indices;
        max_indices.reserve(n_children);
        for (size_t i = 0; i < n_children; ++i)
        {
            float temp_score = cucb_score_precomputed(
                child_ptrs[i], min_max_stats, mean_q, root->is_reset,
                pb_c_base_factor, empirical_prob_sum, root->value_prefix,
                discount_factor, players, continuous_action_space
            );

            if (max_score < temp_score)
            {
                max_score = temp_score;
                max_indices.clear();
                max_indices.push_back(i);
            }
            else if (temp_score >= max_score - epsilon)
            {
                max_indices.push_back(i);
            }
        }

        CAction action;
        if (!max_indices.empty())
        {
            int rand_index = rand() % max_indices.size();
            action = root->legal_actions[max_indices[rand_index]];
        }
        return action;
    }

    // sampled related core code
    float cucb_score(CNode *parent, CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount_factor, int players, bool continuous_action_space)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - parent_mean_q: the mean q value of the parent node.
            - is_reset: whether the value prefix needs to be reset.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - parent_value_prefix: the value prefix of parent node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - players: the number of players.
            - continuous_action_space: whether the action space is continous in current env.
        Returns:
            - ucb_value: the ucb score of the child.
        */
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        // prior_score = pb_c * child->prior;

        // sampled related core code
        // TODO(pu): empirical distribution
        std::string empirical_distribution_type = "density";
        if (empirical_distribution_type == "density")
        {
            if (continuous_action_space == true)
            {
                float empirical_prob_sum = 0;
                for (int i = 0; i < parent->children.size(); ++i)
                {
                    empirical_prob_sum += exp(parent->get_child(parent->legal_actions[i])->prior);
                }
                prior_score = pb_c * exp(child->prior) / (empirical_prob_sum + 1e-6);
            }
            else
            {
                float empirical_prob_sum = 0;
                for (int i = 0; i < parent->children.size(); ++i)
                {
                    empirical_prob_sum += parent->get_child(parent->legal_actions[i])->prior;
                }
                prior_score = pb_c * child->prior / (empirical_prob_sum + 1e-6);
            }
        }
        else if (empirical_distribution_type == "uniform")
        {
            prior_score = pb_c * 1 / parent->children.size();
        }
        // sampled related core code
        if (child->visit_count == 0)
        {
            value_score = parent_mean_q;
        }
        else
        {
            float true_reward = child->value_prefix - parent_value_prefix;
            if (is_reset == 1)
            {
                true_reward = child->value_prefix;
            }

            if (players == 1)
                value_score = true_reward + discount_factor * child->value();
            else if (players == 2)
                value_score = true_reward + discount_factor * (-child->value());
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0)
            value_score = 0;
        if (value_score > 1)
            value_score = 1;

        float ucb_value = prior_score + value_score;
        return ucb_value;
    }

    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch, bool continuous_action_space)
    {
        /*
        Overview:
            Search node path from the roots.
        Arguments:
            - roots: the roots that search from.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - min_max_stats: a tool used to min-max normalize the score.
            - results: the search results.
            - virtual_to_play_batch: the batch of which player is playing on this node.
            - continuous_action_space: whether the action space is continous in current env.
        */
        // set seed
        get_time_and_set_rand_seed();

        std::vector<float> null_value;
        for (int i = 0; i < 1; ++i)
        {
            null_value.push_back(i + 0.1);
        }
        // CAction last_action = CAction(null_value, 1);
        std::vector<float> last_action;
        results.search_lens = std::vector<int>();

        int players = 0;
        int largest_element = *max_element(virtual_to_play_batch.begin(), virtual_to_play_batch.end()); // 0 or 2
        if (largest_element == -1)
            players = 1;
        else
            players = 2;

        for (int i = 0; i < results.num; ++i)
        {
            float parent_q = 0.0;
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;
            results.search_paths[i].push_back(node);

            while (node->expanded())
            {
                float mean_q = node->compute_mean_q(is_root, parent_q, discount_factor);
                is_root = 0;
                parent_q = mean_q;

                CAction action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players, continuous_action_space);
                if (players > 1)
                {
                    assert(virtual_to_play_batch[i] == 1 || virtual_to_play_batch[i] == 2);
                    if (virtual_to_play_batch[i] == 1)
                        virtual_to_play_batch[i] = 2;
                    else
                        virtual_to_play_batch[i] = 1;
                }

                node->best_action = action; // CAction
                // next
                node = node->get_child(action);
                last_action = action.value;

                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            CNode *parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.latent_state_index_in_search_path.push_back(parent->current_latent_state_index);
            results.latent_state_index_in_batch.push_back(parent->batch_index);

            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
            results.virtual_to_play_batchs.push_back(virtual_to_play_batch[i]);
        }
    }

}
