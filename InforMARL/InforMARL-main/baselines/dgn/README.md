# pytorch_DGN

This is a pytorch implementation of [DGN](https://arxiv.org/abs/1810.09202).


### Surviving

Here, we propose a multi-agent environment Surviving to evaluate DGN in large-scale situation. There are 100 agents interacting with the environment. Each agent corresponds to one grid and has a vary limited local observation that contains a square view with 3 x 3 grids centered at the agent. The agent could communicate with the neighboring agents in the square region with 7 x 7 grids. At each timestep, each
 agent can move to one of four neighboring grids or eat the food at its location. Agents begin with 10 health and lose 1 health per step. Eating food could increase the health. If the agent hits 0 health, it gets a reward -0.2, otherwise the reward is 0.4. 

<img src="./Surviving/surviving.png" alt="Surviving" width="500">


### Citation

If you are using the codes, please cite our paper.

[Jiechuan Jiang, Chen Dun, Tiejun Huang, and Zongqing Lu. *Graph convolutional reinforcement learning*. ICLR'20.](https://arxiv.org/abs/1810.09202)

	@inproceedings{jiang2020graph,
	    	title={Graph Convolutional Reinforcement Learning},
	    	author={Jiang, Jiechuan and Dun, Chen and Huang, Tiejun and Lu, Zongqing},
	    	booktitle={ICLR},
	    	year={2020}
	}
