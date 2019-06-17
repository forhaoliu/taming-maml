from meta_policy_search.utils import utils, logger
from meta_policy_search.samplers.base import SampleProcessor
import numpy as np
import tensorflow as tf
import time

class TMAMLMetaSampleProcessor(SampleProcessor):
    """
    Sample processor for DICE implementations
        - fits a reward baseline (use zero baseline to skip this step)
        - computes adjusted rewards (reward - baseline)
        - normalize adjusted rewards if desired
        - zero-pads paths to max_path_length
        - stacks the padded path data

    Args:
        baseline (Baseline) : a time dependent reward baseline object
        max_path_length (int): maximum path length
        discount (float) : reward discount factor
        normalize_adv (bool) : indicates whether to normalize the estimated advantages (zero mean and unit std)
        positive_adv (bool) : indicates whether to shift the (normalized) advantages so that they are all positive
        return_baseline (Baseline): (optional) a state(-time) dependent baseline -
                                    if provided it is also fitted and used to calculate GAE advantage estimates

    """

    def __init__(
            self,
            baseline,
            metabaseline,
            max_path_length,
            discount=0.99,
            gae_lambda=1,
            normalize_adv=True,
            positive_adv=False,
            return_baseline=None
    ):

        assert 0 <= discount <= 1.0, 'discount factor must be in [0,1]'
        assert max_path_length > 0
        assert hasattr(baseline, 'fit') and hasattr(baseline, 'predict')

        self.max_path_length = max_path_length
        self.baseline = baseline
        self.metabaseline = metabaseline
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv
        self.return_baseline = return_baseline

    def process_samples(self, paths_meta_batch, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """
        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        assert self.baseline, 'baseline must be specified'

        samples_data_meta_batch = []
        all_paths = []

        start_time = time.time()
        for _, paths in paths_meta_batch.items():

            # fits baseline, compute advantages and stack path data
            samples_data, paths = self._compute_samples_data(paths)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)
        print("Computting baselines .... %s seconds"%(time.time() - start_time))

        # 7) compute normalized trajectory-batch rewards (for E-MAML)
        overall_avg_reward = np.mean(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))
        overall_avg_reward_std = np.std(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))

        for samples_data in samples_data_meta_batch:
            samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (overall_avg_reward_std + 1e-8)

        # 8) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        return samples_data_meta_batch

    """ helper functions """

    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards and return
        paths = self._compute_discounted_rewards(paths)

        # 2) fit a meta baseline
        for _, path in enumerate(paths):
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)
        self.metabaseline.fit(paths, target_key="returns")
        for _, path in enumerate(paths):
            path["meta_baselines_nu"] = self.metabaseline.predict(path)

        # 3) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key='discounted_rewards')
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 4) compute adjusted rewards (r - b)
        paths = self._compute_adjusted_rewards(paths, all_path_baselines)

        # 5) stack path data
        mask, observations, actions, rewards, adjusted_rewards, env_infos, agent_infos, meta_baselines = self._pad_and_stack_paths(paths)

        # 6) if desired normalize / shift adjusted_rewards
        if self.normalize_adv:
            adjusted_rewards = utils.normalize_advantages(adjusted_rewards)
            meta_baselines = utils.normalize_metabaselines(meta_baselines)
        if self.positive_adv:
            adjusted_rewards = utils.shift_advantages_to_positive(adjusted_rewards)

        # 7) create samples_data object
        samples_data = dict(
            mask=mask,
            observations=observations,
            actions=actions,
            rewards=rewards,
            env_infos=env_infos,
            agent_infos=agent_infos,
            adjusted_rewards=adjusted_rewards,
            meta_baselines=meta_baselines,
        )

        # if return baseline is provided also compute GAE advantage estimates
        if self.return_baseline is not None:
            paths, advantages = self._fit_reward_baseline_compute_advantages(paths)
            samples_data['advantages'] = advantages

        return samples_data, paths

    def _log_path_stats(self, paths, log=False, log_prefix=''):
        # compute log stats
        average_discounted_return = [sum(path["discounted_rewards"]) for path in paths]
        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        if log == 'reward':
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))

        elif log == 'all' or log is True:
            logger.logkv(log_prefix + 'AverageDiscountedReturn', np.mean(average_discounted_return))
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.logkv(log_prefix + 'NumTrajs', len(paths))
            logger.logkv(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.logkv(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.logkv(log_prefix + 'MinReturn', np.min(undiscounted_returns))

    def _compute_discounted_rewards(self, paths):
        discount_array = np.cumprod(np.concatenate([np.ones(1), np.ones(self.max_path_length - 1) * self.discount]))

        for path in paths:
            path_length = path['rewards'].shape[0]
            path["discounted_rewards"] = path['rewards'] * discount_array[:path_length]
        return paths

    def _compute_adjusted_rewards(self, paths, all_path_baselines):
        assert len(paths) == len(all_path_baselines)

        for idx, path in enumerate(paths):
            path_baselines = all_path_baselines[idx]
            deltas = path["discounted_rewards"] - path_baselines
            path["adjusted_rewards"] = deltas
        return paths

    def _pad_and_stack_paths(self, paths):
        mask, observations, actions, rewards, adjusted_rewards, env_infos, agent_infos, meta_baselines = [], [], [], [], [], [], [], []
        for path in paths:
            # zero-pad paths if they don't have full length +  create mask
            path_length = path["observations"].shape[0]
            assert self.max_path_length >= path_length

            mask.append(self._pad(np.ones(path_length), path_length))
            observations.append(self._pad(path["observations"], path_length))
            actions.append(self._pad(path["actions"], path_length))
            rewards.append(self._pad(path["rewards"], path_length))
            adjusted_rewards.append(self._pad(path["adjusted_rewards"], path_length))
            env_infos.append(dict([(key, self._pad(array, path_length)) for key, array in path["env_infos"].items()]))
            agent_infos.append((dict([(key, self._pad(array, path_length)) for key, array in path["agent_infos"].items()])))
            meta_baselines.append(self._pad(path['meta_baselines_nu'], path_length))

        # stack
        mask = np.stack(mask, axis=0) # shape: (batch_size, max_path_length)
        observations = np.stack(observations, axis=0) # shape: (batch_size, max_path_length, ndim_act)
        actions = np.stack(actions, axis=0) # shape: (batch_size, max_path_length, ndim_obs)
        rewards = np.stack(rewards, axis=0) # shape: (batch_size, max_path_length)
        adjusted_rewards = np.stack(adjusted_rewards, axis=0) # shape: (batch_size, max_path_length)
        env_infos = utils.stack_tensor_dict_list(env_infos) # dict of ndarrays of shape: (batch_size, max_path_length, ?)
        agent_infos = utils.stack_tensor_dict_list(agent_infos) # dict of ndarrays of shape: (batch_size, max_path_length, ?)
        meta_baselines = np.stack(meta_baselines, axis=0) # shape: (batch_size, max_path_length)

        return mask, observations, actions, rewards, adjusted_rewards, env_infos, agent_infos, meta_baselines

    def _pad(self, array, path_length):
        assert path_length == array.shape[0]
        if array.ndim == 2:
            return np.pad(array, ((0, self.max_path_length - path_length), (0, 0)),  mode='constant')
        elif array.ndim == 1:
            return np.pad(array, (0, self.max_path_length - path_length), mode='constant')
        else:
            raise NotImplementedError

    def _fit_reward_baseline_compute_advantages(self, paths):
        """
        only to be called if return_baseline is provided. Computes GAE advantage estimates
        """
        assert self.return_baseline is not None

        # a) compute returns
        for idx, path in enumerate(paths):
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)

        # b) fit return baseline estimator using the path returns and predict the return baselines
        self.return_baseline.fit(paths, target_key='returns')
        all_path_baselines = [self.return_baseline.predict(path) for path in paths]

        # c) generalized advantage estimation
        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)

        # d) pad paths and stack them
        advantages = []
        for path in paths:
            path_length = path["observations"].shape[0]
            advantages.append(self._pad(path["advantages"], path_length))

        advantages = np.stack(advantages, axis=0)

        # e) desired normalize / shift advantages
        if self.normalize_adv:
            advantages = utils.normalize_advantages(advantages)
        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)

        return paths, advantages