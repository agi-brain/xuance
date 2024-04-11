## Installation
- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.14.0), numpy (1.18.2)

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"cn"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `800`)

- `--num-units`: number of units in the MLP (default: `128`)

### Training for prior network 
- `--prior-buffer-size`: prior network training buffer size

- `--prior-num-iter`: prior network training iterations

- `--prior-training-rate`: prior network training rate

- `--prior-training-percentile`: control threshold for KL value to get labels

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `None`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `""`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)

- `--restore_all`: whether to restore existing I2C network


### Training procedure
 I2C  be learned end-to-end or in a two-phase manner. This code is implemented for end-to-end manner which could take more training time compared with the latter manner

For Cooperative Navigation, 
`python3 train.py --scenario 'cn' --prior-training-percentile 60 --lr 1e-2`

For Predator Prey, 
`python3 train.py --scenario 'pp' --prior-training-percentile 40 --lr 1e-3`

### Citations

If you are using the codes, please cite our paper.

[Ziluo Ding, Tiejun Huang, and Zongqing Lu. *Learning Individually Inferred Communication for Multi-Agent Cooperation*. NeurIPS'20.](https://arxiv.org/abs/2006.06455)

	@inproceedings{ding2020learning,
        	title={Learning Individually Inferred Communication for Multi-Agent Cooperation},
        	author={Ding, Ziluo and Huang, Tiejun and Lu, Zongqing},
        	booktitle={NeurIPS},
        	year={2020}
	}

### Acknowledgements

This code is developed based on the source code of MADDPG by Ryan Lowe




