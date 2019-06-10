# Taming MAML: Efficient unbiased meta-reinforcement learning

Reference Tensorflow implementation of [Taming MAML: Efficient unbiased meta-reinforcement learning](http://proceedings.mlr.press/v97/liu19g.html).
We will release Pytorch version later.


## Getting started
You can use [Dockerfile](Dockerfile) to build an image with conda environment called _tmaml_ included, activating this conda env:
```
conda activate tmaml
```
you can also use [`tmaml.yml`](tmaml.yml) to create a conda env called _tmaml_.
```
conda env create -f tmaml.yml
```
then activate this conda env
```
conda activate tmaml
```

## Usage
You can use the [`tmaml_run_mujoco.py`](tmaml_run_mujoco.py) , [`vpg_run_mujoco.py`](vpg_run_mujoco.py) and [`dice_vpg_run_mujoco.py`](dice_vpg_run_mujoco.py) scripts in order to run reinforcement learning experiments with different algorithm.
MAML:
```
python vpg_run_mujoco.py --env HalfCheetahRandDirecEnv
```
MAML + DICE:
```
python dice_vpg_run_mujoco.py --env HalfCheetahRandDirecEnv
```
TMAML:
```
python tmaml_run_mujoco.py --env HalfCheetahRandDirecEnv
```


### References
To cite TMAML please use
```
@InProceedings{pmlr-v97-liu19g,
  title = 	 {Taming {MAML}: Efficient unbiased meta-reinforcement learning},
  author = 	 {Liu, Hao and Socher, Richard and Xiong, Caiming},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {4061--4071},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
}
```

#### TODOs
- [x] Adding TMAML
- [x] Adding MAML
- [x] Adding DICE
- [ ] Benchmarking
- [ ] Pytorch version

### Acknowledgements
This repository is based on [ProMP repo](https://github.com/jonasrothfuss/ProMP).