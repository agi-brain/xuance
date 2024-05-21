from xuance.environment.utils.wrapper import XuanCeEnvWrapper, XuanCeMultiAgentEnvWrapper


class MakeEnvironment(XuanCeEnvWrapper):
    """Creates a environment that can run in XuanCe agents"""
    def __init__(self, raw_env):
        super(MakeEnvironment, self).__init__(raw_env)


class MakeMultiAgentEnvironment(XuanCeMultiAgentEnvWrapper):
    """Creates a multi-agent environment that can run in XuanCe agents"""
    def __init__(self, raw_multi_agent_env):
        super(MakeMultiAgentEnvironment, self).__init__(raw_multi_agent_env)
