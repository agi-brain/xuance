from argparse import Namespace
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.common import Union
from xuance.torch.agents.multi_agent_rl.iac_agents import IAC_Agents


class IC3Net_Agents(IAC_Agents):
    """The implementation of IC3Net_Agents agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv]):
        super(IC3Net_Agents, self).__init__(config, envs)
