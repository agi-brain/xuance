MFAC
======================

算法描述
----------------------

MFAC算法的全称是Mean-Field Actor-Critic，是一种基于Actor-Critic的多智能体强化学习算法，
其基本思想和MFQ算法相同，将智能体的局部观测、局部动作和邻居智能体的平均动作作为网路的输入。
MFAC采用Actor-Critic结构实现策略的更新，网路结构和A2C相似。

算法出处
----------------------

**该算法的编写参考如下文献**:

`Mean field multi-agent reinforcement learning 
<http://proceedings.mlr.press/v80/yang18d/yang18d.pdf>`_

**论文引用信息**:

::

    @inproceedings{yang2018mean,
        title={Mean field multi-agent reinforcement learning},
        author={Yang, Yaodong and Luo, Rui and Li, Minne and Zhou, Ming and Zhang, Weinan and Wang, Jun},
        booktitle={International conference on machine learning},
        pages={5571--5580},
        year={2018},
        organization={PMLR}
    }
