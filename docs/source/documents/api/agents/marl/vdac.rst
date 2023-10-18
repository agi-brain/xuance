VDAC
======================

算法描述
----------------------

VDAC算法全称为Value Decomposition Actor-Critic，是一种基于Actor-Critic的多智能体强化学习算法。
VDAC算法发扬了VDN算法和A2C算法的优势，将值函数分解思想引入Actor-Critic结构中，实现多智能体分布式策略的优化。

在本算法库中，VDAC的结构和A2C相同，不同之处在于VDAC将各智能体独立的Q值经过VDN mixer得到整体Q值，进而实现端到端训练。

算法出处
----------------------

**该算法的编写参考如下论文**:
`Value-decomposition multi-agent actor-critics 
<https://ojs.aaai.org/index.php/AAAI/article/view/17353>`_

**论文引用信息**:

::

    @inproceedings{su2021value,
        title={Value-decomposition multi-agent actor-critics},
        author={Su, Jianyu and Adams, Stephen and Beling, Peter},
        booktitle={Proceedings of the AAAI conference on artificial intelligence},
        volume={35},
        number={13},
        pages={11352--11360},
        year={2021}
    }
