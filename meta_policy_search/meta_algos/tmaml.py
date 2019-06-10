from meta_policy_search.meta_algos.dice_maml import DICEMAML

import tensorflow as tf
from collections import OrderedDict


class TMAML(DICEMAML):
    """
    Algorithm for VPG + MAML + DICE

    Args:
        max_path_length (int): maximum path length
        policy (Policy) : policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for the meta-objective
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(
            self,
            max_path_length,
            *args,
            name="vpg_tmaml",
            **kwargs
            ):
        super(TMAML, self).__init__(max_path_length, *args, **kwargs)

        self._optimization_keys = ['observations', 'actions', 'advantages', 'adjusted_rewards', 'mask', 'agent_infos', 'meta_baselines']
        self.name = name

        self.build_graph()

    def _adapt_objective_sym(self, action_stacked_sym, adj_reward_sym, mask_sym, dist_info_stacked_sym, metabaseline_sym):
        with tf.variable_scope("log_likelihood"):
            log_likelihood_adapt = self.policy.distribution.log_likelihood_sym(action_stacked_sym, dist_info_stacked_sym)
            log_likelihood_adapt = tf.reshape(log_likelihood_adapt, tf.shape(mask_sym))
        with tf.variable_scope("tmaml_loss"):
            obj_adapt = - tf.reduce_mean(magic_box(log_likelihood_adapt) * adj_reward_sym * mask_sym)
            term1 = tf.cumsum(log_likelihood_adapt, axis=1) - log_likelihood_adapt
            term1 = tf.concat([term1[1:, :], tf.expand_dims(tf.ones([term1.shape[1]], dtype=tf.float32), axis=0)], axis=0)
            term1 = 1 - detach_ratio(term1)
            term2 = detach_ratio(log_likelihood_adapt) # 1-detach_ration(log_likelihood_adapt) or detach_ration(log_likelihood_adapt)
            tmaml = tf.reduce_mean(term1 * term2 * metabaseline_sym)
            obj_adapt = obj_adapt - tmaml
        return obj_adapt

    def build_graph(self):
        """
        Creates the computation graph for DICE MAML
        """

        """ Build graph for sampling """
        with tf.variable_scope(self.name + '_sampling'):
            self.step_sizes = self._create_step_size_vars()

            """ --- Build inner update graph for adapting the policy and sampling trajectories --- """
            # this graph is only used for adapting the policy and not computing the meta-updates
            self.adapted_policies_params, self.adapt_input_ph_dict = self._build_inner_adaption()


        """ Build graph for meta-update """
        meta_update_scope = tf.variable_scope(self.name + '_meta_update')

        with meta_update_scope:
            obs_phs, action_phs, adj_reward_phs, mask_phs, dist_info_old_phs, all_phs_dict, metabaseline_phs = self._make_tmaml_input_placeholders('step0')
            self.meta_op_phs_dict = OrderedDict(all_phs_dict)

            distribution_info_vars, current_policy_params, all_surr_objs = [], [], []

        for i in range(self.meta_batch_size):
            obs_stacked = self._reshape_obs_phs(obs_phs[i])
            dist_info_sym = self.policy.distribution_info_sym(obs_stacked, params=None)
            distribution_info_vars.append(dist_info_sym)  # step 0
            current_policy_params.append(self.policy.policy_params) # set to real policy_params (tf.Variable)

        with meta_update_scope:
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                with tf.variable_scope("inner_update_%i"%step_id):
                    surr_objs, adapted_policy_params = [], []

                    # inner adaptation step for each task
                    for i in range(self.meta_batch_size):
                        action_stacked = self._reshape_action_phs(action_phs[i])
                        surr_loss = self._adapt_objective_sym(action_stacked, adj_reward_phs[i], mask_phs[i], distribution_info_vars[i], metabaseline_phs[i])

                        adapted_params_var = self._adapt_sym(surr_loss, current_policy_params[i])

                        adapted_policy_params.append(adapted_params_var)
                        surr_objs.append(surr_loss)

                    all_surr_objs.append(surr_objs)
                    # Create new placeholders for the next step
                obs_phs, action_phs, adj_reward_phs, mask_phs, dist_info_old_phs, all_phs_dict, metabaseline_phs = self._make_tmaml_input_placeholders('step%i' % step_id)
                self.meta_op_phs_dict.update(all_phs_dict)

                # dist_info_vars_for_next_step
                distribution_info_vars = []
                for i in range(self.meta_batch_size):
                    obs_stacked = self._reshape_obs_phs(obs_phs[i])
                    distribution_info_vars.append(self.policy.distribution_info_sym(obs_stacked, params=adapted_policy_params[i]))

                current_policy_params = adapted_policy_params

            """ Outer (meta-)objective """
            with tf.variable_scope("outer_update"):
                adv_phs, phs_dict = self._make_advantage_phs('step%i' % self.num_inner_grad_steps)
                self.meta_op_phs_dict.update(phs_dict)

                surr_objs = []

                # meta-objective
                for i in range(self.meta_batch_size):
                    action_stacked = self._reshape_action_phs(action_phs[i])
                    log_likelihood = self.policy.distribution.log_likelihood_sym(action_stacked, distribution_info_vars[i])
                    log_likelihood = tf.reshape(log_likelihood, tf.shape(mask_phs[i]))
                    surr_obj = - tf.reduce_mean(log_likelihood * adv_phs[i] * mask_phs[i])
                    surr_objs.append(surr_obj)

                """ Mean over meta tasks """
                meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0))

                self.optimizer.build_graph(
                    loss=meta_objective,
                    target=self.policy,
                    input_ph_dict=self.meta_op_phs_dict,
                )

    def _make_advantage_phs(self, prefix=''):
        adv_phs = []
        all_phs_dict = OrderedDict()

        for task_id in range(self.meta_batch_size):
            # advantage ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length], name='advantage' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'advantages')] = ph
            adv_phs.append(ph)

        return adv_phs, all_phs_dict


    def _build_inner_adaption(self):
        obs_phs, action_phs, adj_reward_phs, mask_phs, dist_info_old_phs, adapt_input_ph_dict, metabaseline_phs = self._make_tmaml_input_placeholders('adapt')

        adapted_policies_params = []

        for i in range(self.meta_batch_size):
            with tf.variable_scope("adapt_task_%i" % i):
                with tf.variable_scope("adapt_objective"):
                    obs_stacked = self._reshape_obs_phs(obs_phs[i])
                    action_stacked = self._reshape_action_phs(action_phs[i])
                    distribution_info_stacked = self.policy.distribution_info_sym(obs_stacked,
                                                                              params=self.policy.policies_params_phs[i])

                    # inner surrogate objective
                    adapt_loss = self._adapt_objective_sym(action_stacked, adj_reward_phs[i], mask_phs[i],
                                                           distribution_info_stacked, metabaseline_phs[i])

                # get tf operation for adapted (post-update) policy
                with tf.variable_scope("adapt_step"):
                    adapted_policy_param = self._adapt_sym(adapt_loss, self.policy.policies_params_phs[i])
                adapted_policies_params.append(adapted_policy_param)

        return adapted_policies_params, adapt_input_ph_dict


    def _make_tmaml_input_placeholders(self, prefix=''):
        """
        In contrast to make_input_placeholders each placeholder has one dimension more with the size of self.max_path_length
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task,
            and for convenience, a list containing all placeholders created
        """
        obs_phs, action_phs, adj_reward, mask_phs, dist_info_phs, metabaseline_phs = [], [], [], [], [], []
        dist_info_specs = self.policy.distribution.dist_info_specs

        all_phs_dict = OrderedDict()

        for task_id in range(self.meta_batch_size):
            # observation ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length, self.policy.obs_dim],
                                name='obs' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'observations')] = ph
            obs_phs.append(ph)

            # action ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length, self.policy.action_dim],
                                name='action' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'actions')] = ph
            action_phs.append(ph)

            # adjusted reward ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length], name='adjusted_rewards' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'adjusted_rewards')] = ph
            adj_reward.append(ph)

            # metabaseline ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length], name='meta_baselines' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'meta_baselines')] = ph
            metabaseline_phs.append(ph)

            # mask ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length],
                                name='mask' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'mask')] = ph
            mask_phs.append(ph)

            # distribution / agent info
            dist_info_ph_dict = {}
            for info_key, shape in dist_info_specs:
                ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length] + list(shape),
                                    name='%s_%s_%i' % (info_key, prefix, task_id))
                all_phs_dict['%s_task%i_agent_infos/%s' % (prefix, task_id, info_key)] = ph
                dist_info_ph_dict[info_key] = ph
            dist_info_phs.append(dist_info_ph_dict)

        return obs_phs, action_phs, adj_reward, mask_phs, dist_info_phs, all_phs_dict, metabaseline_phs

def magic_box(logprobs):
    """
    Dice magic box operator

    Args:
        logprobs: 2d tensor of log probabilities (batch_size, max_path_length)

    Returns: tf.Tensor of shape : Dice magic box operator

    """
    tf.assert_rank(logprobs, 2)
    with tf.variable_scope("magic_box"):
        tau = tf.cumsum(logprobs, axis=1)
        magic_box = tf.exp(tau - tf.stop_gradient(tau))
    return magic_box

def detach_ratio(tensor):
    with tf.variable_scope("detach_ratio"):
        ratio = tf.exp(tensor - tf.stop_gradient(tensor))
    return ratio