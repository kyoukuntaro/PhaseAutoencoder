# Phase autoencoder for limit-cycle oscillators
This is the program used in the following paper.\
Title: Phase autoencoder for limit-cycle oscillators\
Author: Koichiro Yawata, Kai Fukami, Kunihiko Taira, Hiroya Nakao\
arxiv preprint: https://arxiv.org/abs/2403.06992

## Files
* `main.py`
    - Training program for Phase Autoencoder.
* `PhaseReductionNet.py`
    - Program for describe model.
* `utils/limitcycle.py`
    - Program to generate limit-cycles
* `utils/dataset.py`
    - Programs to format training datasets.
* `data`
    - Data on pre-computed limit-cycle trajectories and phase sensitivity functions. The phase sensitivity function was calculated using adjoint method.
* `out`
    - Learning results. The paper's results and figures were generated using.
* `note`
    - Jupyter notebook used for making the figures.

## Traing script using paper's experint
```
# Stuart-Landau oscillator
python main.py --ex_name SL --epoch_size 50 --lc_name SL --step_interval 10 --data_interval 5 --lr 0.001 --w_step 0.5 --w_z1 2.0 --hidden_dim 100 --step_num 20 --noise_rate 0.5 
# FitzHugh-Nagumo model
python main.py --ex_name FHN --epoch_size 50 --lc_name FHN --step_interval 36 --data_interval 50 --lr 0.001 --w_step 0.5 --w_z1 2.0 --hidden_dim 100 --step_num 20 --train_traj_num 1000 --noise_rate 0.5
# Hodgkin-Huxley mode
python main.py --ex_name HH --epoch_size 50 --lc_name HH --step_interval 100 --data_interval 5 --lr 0.001 --w_step 0.5 --w_z1 2.0 --hidden_dim 100 --step_num 20 --train_traj_num 1000 --noise_rate 0.5
# Collectively Oscillating Network
python main.py --ex_name CO --epoch_size 50 --lc_name FHNR --step_interval 24 --data_interval 37 --lr 0.001 --w_step 0.5 --w_z1 2.0 --hidden_dim 100 --step_num 20 --train_traj_num 1000 --noise_rate 0.5
```