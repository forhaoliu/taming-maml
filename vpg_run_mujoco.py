# from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
# from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc import HumanoidRandDirecEnv
# from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
# from meta_policy_search.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv
# from meta_policy_search.envs.mujoco_envs.ant_rand_direc_2d import AntRandDirec2DEnv
# from meta_policy_search.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
# from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
# from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
# from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc import HumanoidRandDirecEnv
# from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
# from meta_policy_search.envs.mujoco_envs.walker2d_rand_direc import Walker2DRandDirecEnv
# from meta_policy_search.envs.mujoco_envs.walker2d_rand_vel import Walker2DRandVelEnv
# from meta_policy_search.envs.mujoco_envs.swimmer_rand_vel import SwimmerRandVelEnv
# from meta_policy_search.envs.normalized_env import normalize
# from meta_policy_search.meta_algos.vpg_maml import VPGMAML
# from meta_policy_search.meta_trainer import Trainer
# from meta_policy_search.samplers.meta_sampler import MetaSampler
# from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
# from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
# from meta_policy_search.utils import logger
# from meta_policy_search.utils.utils import set_seed, ClassEncoder

# import numpy as np
# import os
# import json
# import argparse
# import time

# meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

# def main(config):
#     set_seed(config['seed'])


#     baseline =  globals()[config['baseline']]() #instantiate baseline

#     env = globals()[config['env']]() # instantiate env
#     env = normalize(env) # apply normalize wrapper to env

#     policy = MetaGaussianMLPPolicy(
#             name="meta-policy",
#             obs_dim=np.prod(env.observation_space.shape),
#             action_dim=np.prod(env.action_space.shape),
#             meta_batch_size=config['meta_batch_size'],
#             hidden_sizes=config['hidden_sizes'],
#         )

#     sampler = MetaSampler(
#         env=env,
#         policy=policy,
#         rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
#         meta_batch_size=config['meta_batch_size'],
#         max_path_length=config['max_path_length'],
#         parallel=config['parallel'],
#     )

#     sample_processor = MetaSampleProcessor(
#         baseline=baseline,
#         discount=config['discount'],
#         gae_lambda=config['gae_lambda'],
#         normalize_adv=config['normalize_adv'],
#     )

#     algo = VPGMAML(
#         policy=policy,
#         step_size=config['step_size'],
#         inner_type=config['inner_type'],
#         inner_lr=config['inner_lr'],
#         meta_batch_size=config['meta_batch_size'],
#         num_inner_grad_steps=config['num_inner_grad_steps'],
#         exploration=False,
#     )

#     trainer = Trainer(
#         algo=algo,
#         policy=policy,
#         env=env,
#         sampler=sampler,
#         sample_processor=sample_processor,
#         n_itr=config['n_itr'],
#         num_inner_grad_steps=config['num_inner_grad_steps'],
#     )

#     trainer.train()

# if __name__=="__main__":
#     parser = argparse.ArgumentParser(description='Taming MAML')
#     parser.add_argument('--dump_path', type=str, default=os.path.join(os.path.realpath(os.path.dirname(__file__)), 'logs'))
#     parser.add_argument('--seed', type=int, help='random seed', default=0)
#     parser.add_argument('--env', type=str, help='env name', default='HalfCheetahRandDirecEnv')
#     parser.add_argument('--inner_type', type=str, help='type of inner loss function', default='log_likelihood')
#     parser.add_argument('--n_iter', type=int, help='number of iterations', default=5001)

#     args = parser.parse_args()

#     os.makedirs(args.dump_path, exist_ok=True)

#     config = {
#         'seed': 1,

#         'baseline': 'LinearFeatureBaseline',

#         'env': 'HalfCheetahRandDirecEnv',

#         # sampler config
#         'rollouts_per_meta_task': 20,
#         'max_path_length': 100,
#         'parallel': True,

#         # sample processor config
#         'discount': 0.99,
#         'gae_lambda': 1,
#         'normalize_adv': True,

#         # policy config
#         'hidden_sizes': (64, 64),
#         'learn_std': True, # whether to learn the standard deviation of the gaussian policy

#         'inner_lr': 0.1, # adaptation step size
#         'learning_rate': 1e-3, # meta-policy gradient step size
#         'step_size': 0.01, # size of the TRPO trust-region
#         'n_itr': 5001, # number of overall training iterations
#         'meta_batch_size': 40, # number of sampled meta-tasks per iterations
#         'num_inner_grad_steps': 1, # number of inner / adaptation gradient steps
#         'inner_type' : 'log_likelihood', # type of inner loss function used

#     }

#     config.update(vars(args))

#     # configure logger
#     logger.configure(dir=args.dump_path, format_strs=['stdout', 'log', 'csv', 'tensorboard'],
#                      snapshot_mode='last_gap')

#     # dump run configuration before starting training
#     json.dump(config, open(args.dump_path + '/params.json', 'w'), cls=ClassEncoder)

#     # start the actual algorithm
#     main(config)


from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc import HumanoidRandDirecEnv
from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_direc_2d import AntRandDirec2DEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc import HumanoidRandDirecEnv
from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
from meta_policy_search.envs.mujoco_envs.walker2d_rand_direc import Walker2DRandDirecEnv
from meta_policy_search.envs.mujoco_envs.walker2d_rand_vel import Walker2DRandVelEnv
from meta_policy_search.envs.mujoco_envs.swimmer_rand_vel import SwimmerRandVelEnv
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.meta_algos.vpg_maml import VPGMAML
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder

import numpy as np
import os
import json
import argparse
import time
import os

def main(config):
    set_seed(config['seed'])


    baseline =  globals()[config['baseline']]() #instantiate baseline

    env = globals()[config['env']]() # instantiate env
    env = normalize(env) # apply normalize wrapper to env

    policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )

    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        # max_path_length=config['max_path_length'],
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )

    algo = VPGMAML(
        policy=policy,
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
    )

    trainer.train()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Taming MAML')
    parser.add_argument('--dump_path', type=str, default=os.path.join(os.path.realpath(os.path.dirname(__file__)), 'logs'))
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--env', type=str, help='env name', default='HalfCheetahRandDirecEnv')
    parser.add_argument('--inner_type', type=str, help='type of inner loss function', default='log_likelihood')
    parser.add_argument('--n_iter', type=int, help='number of iterations', default=5001)

    args = parser.parse_args()

    os.makedirs(args.dump_path, exist_ok=True)

    config = {
        'seed': 1,

        'baseline': 'LinearFeatureBaseline',

        'env': 'HalfCheetahRandDirecEnv',

        # sampler config
        'rollouts_per_meta_task': 40,
        'max_path_length': 100,
        'parallel': True,

        # sample processor config
        'discount': 0.99,
        'gae_lambda': 1,
        'normalize_adv': True,

        # policy config
        'hidden_sizes': (64, 64),
        'learn_std': True, # whether to learn the standard deviation of the gaussian policy

        'inner_lr': 0.1, # adaptation step size
        'learning_rate': 1e-3, # meta-policy gradient step size
        'step_size': 0.01, # size of the TRPO trust-region
        'n_itr': 5001, # number of overall training iterations
        'meta_batch_size': 40, # number of sampled meta-tasks per iterations
        'num_inner_grad_steps': 1, # number of inner / adaptation gradient steps
        'inner_type' : 'log_likelihood', # type of inner loss function used

    }

    config.update(vars(args))

    # configure logger
    logger.configure(dir=args.dump_path, format_strs=['stdout', 'log', 'csv', 'tensorboard'],
                     snapshot_mode='last_gap')

    # dump run configuration before starting training
    json.dump(config, open(args.dump_path + '/params.json', 'w'), cls=ClassEncoder)

    # start the actual algorithm
    main(config)