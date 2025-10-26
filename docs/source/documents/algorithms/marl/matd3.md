# Multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3)

**Paper Link:** [**https://arxiv.org/abs/1910.01465**](https://arxiv.org/abs/1910.01465)

This table lists some general features about c algorithm:   

| Features of MATD3                                       | Values | Description                                                                                                   |
|---------------------------------------------------------|-----|---------------------------------------------------------------------------------------------------------------|
| Fully Decentralized                                     |     | There is no communication between agents.                                                                     |
| Fully Centralized                                       |     | Agents send all information to the central controller, and the controller will make decisions for all agents. | 
| Centralized Training With Decentralized Execution(CTDE) |     | The central controller is used in training and abandoned in execution.                                        | 
| On-policy                                               | ❌   | The evaluate policy is the same as the target policy.                                                         | 
| Off-policy                                              | ✅   | The evaluate policy is different from the target policy.                                                      |   
| Model-free                                              | ✅   | No need to prepare an environment dynamics model.                                                             |
| Model-based                                             | ❌   | Need an environment model to train the policy.                                                                | 
| Discrete Action                                         | ✅   | Deal with discrete action space.                                                                              | 
| Continuous Action                                       | ✅   | Deal with continuous action space.                                                                            |

## Citation
```{code-block} bash
@misc{ackermann2019reducingoverestimationbiasmultiagent,
      title={Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics}, 
      author={Johannes Ackermann and Volker Gabler and Takayuki Osa and Masashi Sugiyama},
      year={2019},
      eprint={1910.01465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1910.01465}, 
}
```