#!/usr/bin/python
# -*- coding: utf-8 -*-

import yaml
import subprocess
import sys
import time
import datetime
import pdb
import os
from tqdm import tqdm

import argparse


# envs = {
#     'HalfCheetahRandDirecEnv': HalfCheetahRandDirecEnv,
#     'HalfCheetahRandVelEnv': HalfCheetahRandVelEnv,
#     'AntRandDirecEnv': AntRandDirecEnv,
#     'AntRandDirec2DEnv': AntRandDirec2DEnv,
#     'AntRandGoalEnv': AntRandGoalEnv,
#     'SwimmerRandVelEnv': SwimmerRandVelEnv,
#     'HumanoidRandDirecEnv': HumanoidRandDirecEnv,
#     'HumanoidRandDirec2DEnv': HumanoidRandDirec2DEnv,
#     'Walker2DRandDirecEnv': Walker2DRandDirecEnv,
#     'Walker2DRandVelEnv': Walker2DRandVelEnv,
#     'MetaPointEnvCorner': MetaPointEnvCorner,
#     'MetaPointEnvWalls': MetaPointEnvWalls,
#     'MetaPointEnvMomentum': MetaPointEnvMomentum,
#     'SawyerPickAndPlaceEnv': SawyerPickAndPlaceEnv,
#     'SawyerPushEnv': SawyerPushEnv,
#     'SawyerPushSimpleEnv': SawyerPushSimpleEnv
# }

envs = [
    'HalfCheetahRandDirecEnv',
    'HalfCheetahRandVelEnv',
    # 'AntRandDirecEnv',
    # 'AntRandDirec2DEnv',
    # 'AntRandGoalEnv',
    # 'SwimmerRandVelEnv',
    'HumanoidRandDirecEnv',
    'HumanoidRandDirec2DEnv',
    'Walker2DRandDirecEnv',
    # 'Walker2DRandVelEnv'
]

seeds = [1001, 6006, 9009]

n_iter = 3001

parser = argparse.ArgumentParser(description='gcp launch experiment')
parser.add_argument('--savedir', type=str, default='tmp3')
parser.add_argument('--istmaml', action='store_true')
parser.add_argument('--isdice', action='store_true')
parser.add_argument('--ismaml', action='store_true')
args = parser.parse_args()

isdice = args.isdice
istmaml = args.istmaml
ismaml = args.ismaml

assert sum([ismaml, istmaml, isdice]
           ) == 1, 'make sure only one type of algo is true'

savedir = args.savedir

for idx, env in enumerate(envs):

    # wait for a while since kuberenetes might try to provision standard96 VM for each requested pod thinking that there is no space available

    for seed in seeds:
        if ismaml:
            file_name = 'vpg_run_mujoco'
        elif isdice:
            file_name = 'dice_vpg_run_mujoco'
        elif istmaml:
            file_name = 'tmaml_run_mujoco'
        else:
            assert False
        pod_name = 'sfr-pod-hao-liu-env-name-{}-seed-{}'.format(
            env, seed)
        exp_dir = './logs/env={}-seed={}'.format(env, seed)
        if ismaml:
            pod_name += '-ismaml'
            exp_dir += '-ismaml'
        elif isdice:
            pod_name += '-isdice'
            exp_dir += '-isdice'
        elif istmaml:
            pod_name += '-istmaml'
            exp_dir += '-istmaml'
        else:
            assert False
        timestamp = datetime.datetime.now().strftime('-%Y-%m-%d-%H-%M-%S-%f')
        # exp_dir += timestamp
        # pod_name += timestamp
        pod_name = pod_name.lower().replace("_", "-")

        # command = ['/usr/bin/zsh', '-c',
        #            'source ~/.zshrc && cd /export/home/taming-maml-tf/ && cp /export/home/mjkey /root/.mujoco/mjkey.txt && conda activate mlkit36 && pip install -r requirements.txt && pip install -e . && python {}.py --env {} --seed {} --dump_path {} --n_iter {}'.format(file_name, env, seed, exp_dir, n_iter)]
        command = ['/usr/bin/zsh', '-c',
                   'source ~/.zshrc && cd /export/home/taming-maml-tf/ && cp /export/home/mjkey /root/.mujoco/mjkey.txt && conda activate mlkit37 && pip install -r requirements.txt && pip install -e . && python {}.py --env {} --seed {} --dump_path {} --n_iter {}'.format(file_name, env, seed, exp_dir, n_iter)]

        print(command)

        config = {
            'apiVersion': 'v1',
            'spec': {
                'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution':  {'nodeSelectorTerms': [{'matchExpressions': [{'key': 'sfr-node-type', 'values': ['standard96'], 'operator': 'In'
                                                                                                                                               }]}]}}},
                'volumes': [{'name': 'sfr-home-pv-hao-liu',
                             'persistentVolumeClaim': {'claimName': 'sfr-home-pvc-hao-liu'
                                                       }}],
                'containers': [{
                    'name': 'haoliu-scratchpad',
                    'image': 'gcr.io/salesforce-research-internal/lhao-cpu36:latest',
                    'volumeMounts': [{'name': 'sfr-home-pv-hao-liu',
                                      'mountPath': '/export/home'}],
                    'command': command,
                    'resources': {'limits': {'cpu': '30.0', 'memory': '81920Mi'
                                             }},
                }],
                'restartPolicy': 'Never',
            },
            'kind': 'Pod',
            'metadata': {'name': pod_name,
                         'namespace': 'sfr-ns-hao-liu'},
        }

        outfile_name = os.path.join(
            savedir, '{}.yaml'.format(pod_name))
        os.makedirs(savedir, exist_ok=True)
        with open(outfile_name, 'w+') as outfile:
            yaml.dump(config, outfile,
                      default_flow_style=False)

        subprocess.Popen('kubectl create -f {}'.format(outfile_name),
                         stdout=subprocess.PIPE, shell=True)
