<div align="center">

# InforMARL

**Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation** 

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-coming_soon-red.svg)](https://github.com/nsidn98/InforMARL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: MIT](https://img.shields.io/badge/arXiv-2211.02127-green)](http://arxiv.org/abs/2211.02127)
[![License: MIT](https://img.shields.io/badge/Project-Website-blue)](https://nsidn98.github.io/InforMARL/)

</div>


A graph neural network framework for multi-agent reinforcement learning with limited local observability for each agent. This is an official implementation of the model described in:

"[Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation](http://arxiv.org/abs/2211.02127)",

[Siddharth Nayak](http://nsidn98.github.io/), [Kenneth Choi](https://www.linkedin.com/in/kennethschoi/), [Wenqi Ding](https://github.com/dingwq22), [Sydney Dolan](https://sydneyidolan.com/), [Karthik Gopalakrishnan](https://karthikg.mit.edu/about-me), [Hamsa Balakrishnan](http://www.mit.edu/~hamsa/)


April 2023 - The paper was accepted to ICML'2023! See you in Honolulu in July 2023

Dec 2022 - Presented a short version of this paper at the [Strategic Multi-Agent Interactions: Game Theory for Robot Learning and Decision Making Workshop](https://sites.google.com/view/corl-2022-games-workshop/) at [CoRL](https://corl2022.org/) in Auckland. You can find the recording [here](https://youtu.be/8Ig2LYGvRuk?t=9617).

Please let us know if anything here is not working as expected, and feel free to create [new issues](https://github.com/nsidn98/InforMARL/issues) with any questions.



## Abstract:
We consider the problem of multi-agent navigation and collision avoidance when observations are limited to the local neighborhood of each agent. We propose *InforMARL*, a novel architecture for multi-agent reinforcement learning (MARL) which uses local information intelligently to compute paths for all the agents in a decentralized manner. Specifically, InforMARL aggregates information about the local neighborhood of agents for both the actor and the critic using a graph neural network and can be used in conjunction with any standard MARL algorithm. We show that (1) in training, InforMARL has better sample efficiency and performance than baseline approaches, despite using less information, and (2) in testing, it scales well to environments with arbitrary numbers of agents and obstacles.

![image](https://raw.githubusercontent.com/nsidn98/nsidn98.github.io/master/files/Publications_assets/InforMARL/figures/graphMARLArch.v8.png)

**Overview of InforMARL**: (i) Environment: The agents are depicted by green circles, the goals are depicted by red rectangles, and the unknown obstacles are depicted by gray circles. $x^{i}_{agg}$ represents the aggregated information from the neighborhood, which is the output of the GNN. A graph is created by connecting entities within the sensing-radius of the agents. (ii)  Information Aggregation: Each agent's observation is concatenated with $x^{i}_{\mathrm{agg}}$. The inter-agent edges are bidirectional, while the edges between agents and non-agent entities are unidirectional. (iii) Graph Information Aggregation: The aggregated vector from all the agents is averaged to get $X_{\mathrm{agg}}$.
        (iv) Actor-Critic: The concatenated vector $[o^{i}, x^{i}_{\mathrm{agg}}]$ is fed into the actor network to get the action, and $X_{\mathrm{agg}}$ is fed into the critic network to get the state-action values.

<!-- <p align="left">
  <img src="https://raw.githubusercontent.com/nsidn98/nsidn98.github.io/master/files/Publications_assets/InforMARL/figures/graphMARLArch.v8.png" width="800"/>
</p> -->



## Usage:
To train InforMARL:
```bash
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "informarl" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 0 \
--experiment_name "informarl" \
--scenario_name "navigation_graph" \
--num_agents 3 \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 128
```

## Graph Neural Network Compatible Navigation Environment:
We also provide with code for the navigation environment which is compatible to be used with graph neural networks.

**Note**: A more thorough documentation will be up soon.

`python multiagent/custom_scenarios/navigation_graph.py`

```python
from multiagent.environment import MultiAgentGraphEnv
from multiagent.policy import InteractivePolicy

# makeshift argparser
class Args:
    def __init__(self):
        self.num_agents:int=3
        self.world_size=2
        self.num_scripted_agents=0
        self.num_obstacles:int=3
        self.collaborative:bool=False 
        self.max_speed:Optional[float]=2
        self.collision_rew:float=5
        self.goal_rew:float=5
        self.min_dist_thresh:float=0.1
        self.use_dones:bool=False
        self.episode_length:int=25
        self.max_edge_dist:float=1
        self.graph_feat_type:str='global'
args = Args()

scenario = Scenario()
# create world
world = scenario.make_world(args)
# create multiagent environment
env = MultiAgentGraphEnv(world=world, reset_callback=scenario.reset_world, 
                    reward_callback=scenario.reward, 
                    observation_callback=scenario.observation, 
                    graph_observation_callback=scenario.graph_observation,
                    info_callback=scenario.info_callback, 
                    done_callback=scenario.done,
                    id_callback=scenario.get_id,
                    update_graph=scenario.update_graph,
                    shared_viewer=False)
# render call to create viewer window
env.render()
# create interactive policies for each agent
policies = [InteractivePolicy(env,i) for i in range(env.n)]
# execution loop
obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
stp=0
while True:
    # query for action from each agent's policy
    act_n = []

    for i, policy in enumerate(policies):
        act_n.append(policy.action(obs_n[i]))
    # step environment
    obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)

    # render all agent views
    env.render()
```

Here `env.reset()` returns `obs_n, agent_id_n, node_obs_n, adj_n` where:
- `obs_n`: Includes local observations (position, velocity, relative goal position) of each agent.
- `agent_id_n`: Includes the 'ID' for each agent. This can be used to query any agent specific features in the replay buffer
- `node_obs_n`: Includes node observations for graphs formed wrt each agent $i$. Here each node can be any entity in the environment namely: agent, goal or obstacle. The node features include relative position, relative velocity of the entity and the relative position of the goal on the entity.
- `adj_n`: Includes the adjacency matrix of the graphs formed.

This can also be used with an environment wrapper:
```python
from multiagent.MPE_env import GraphMPEEnv
# all_args can be pulled config.py or refer to `onpolicy/scripts/train_mpe.py`
env = MPEEnv(all_args)
obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
```

## Dependencies:
* [Multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs): We have pulled the relevant folder from the repo to modify it.
    * `pip install gym==0.10.5` (newer versions also seem to work)
    * `pip install numpy-stl`
    * torch==1.11.0              
    * torch-geometric==2.0.4
    * torch-scatter==2.0.8
    * torch-sparse==0.6.12


## Baseline Sources
We compare our methods with other MARL baselines:
* Pulled the MAPPO code from [here](https://github.com/marlbenchmark/on-policy) which was used in this [paper](https://arxiv.org/abs/2103.01955). Also worth taking a look at this [branch](https://github.com/marlbenchmark/on-policy/tree/222626ebef82adbb809adbc011923cf837dd6e89) for their benchmarked code.
* [MADDPG, MATD3, QMIX, VDN](https://github.com/marlbenchmark/off-policy)
* [Graph Policy Gradients](https://github.com/arbaazkhan2/gpg_labeled) (GPG); [Paper](https://arxiv.org/abs/1907.03822)
* [Graph Convolutional Reinforcement Learning](https://github.com/jiechuanjiang/pytorch_DGN) (DGN); [Paper](https://arxiv.org/abs/1810.09202)
* [Entity Message Passing Network](https://github.com/sumitsk/marl_transfer) (EMP); [Paper](https://arxiv.org/abs/1906.01202)


## Troubleshooting:
* `OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.`: Install nomkl by running [`conda install nomkl`](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial)

* `AttributeError: dlsym(RTLD_DEFAULT, CFStringCreateWithCString): symbol not found`: This issue arises with MacOS Big Sur. A hacky fix for this is to revert change the `pyglet` version to maintenance version using `pip install --user --upgrade git+http://github.com/pyglet/pyglet@pyglet-1.5-maintenance`

* `AttributeError: 'NoneType' object has no attribute 'origin'`: This error arises whilst using `torch-geometric` with CUDA. Uninstall `torch_geometric`, `torch-cluster`, `torch-scatter`, `torch-sparse`, and `torch-spline-conv`. Then re-install using:
    ```
    TORCH="1.8.0"
    CUDA="cu102"
    pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html --user
    pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html --user
    pip install torch-geometric --user
    ```

## Questions/Requests

Please file an issue if you have any questions or requests about the code or the paper. If you prefer your question to be private, you can alternatively email me at sidnayak@mit.edu

## Citation

If you found this codebase useful in your research, please consider citing

```bibtex
@article{nayak22informarl,
  doi = {10.48550/ARXIV.2211.02127},
  url = {https://arxiv.org/abs/2211.02127},
  author = {Nayak, Siddharth and Choi, Kenneth and Ding, Wenqi and Dolan, Sydney and Gopalakrishnan, Karthik and Balakrishnan, Hamsa},
  keywords = {Multiagent Systems (cs.MA), Artificial Intelligence (cs.AI), Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Contributing
We would love to include more scenarios from the multi-agent particle environment to be compatible with graph neural networks and would be happy to accept PRs.

## License

MIT License