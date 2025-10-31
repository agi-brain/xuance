# DreamerV2

**Paper Link:** [**ICLR 2021 Conference**](https://openreview.net/pdf?id=0oabwyZbOu).



This table lists some general features about **DreamerV2**:

| Features of DreamerV2 | Values | Description                                              |
|-----------------------|--------|----------------------------------------------------------|
| On-policy             | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy            | ✅      | The evaluate policy is different from the target policy. | 
| Model-free            | ❌      | No need to prepare an environment dynamics model.        | 
| Model-based           | ✅      | Need an environment model to train the policy.           | 
| Discrete Action       | ✅      | Focus on discrete action space mainly.                   |   
| Continuous Action     | ✅      | Focus on continuous action space mainly.                 |    

## World Model Framework

The world model consists of an image encoder, a Recurrent State-Space Model to learn the dynamics, 
and predictors for the image, reward, and discount factor:

![world model](./../../../_static/figures/algo_framework/dreamerv2_world_model_framework.png)


## Citation

```{code-block} bash
@inproceedings{
hafner2021mastering,
title={Mastering Atari with Discrete World Models},
author={Danijar Hafner and Timothy P Lillicrap and Mohammad Norouzi and Jimmy Ba},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=0oabwyZbOu}
}
```
