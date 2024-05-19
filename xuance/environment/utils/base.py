"""
This is an example of creating a new environment in XuanCe.
"""
from xuance.environment.utils.wrapper import XuanCeEnvWrapprer


class NewEnvironment(XuanCeEnvWrapprer):
    """Creates a new environment that can run in XuanCe agents"""
    def __init__(self, raw_env):
        super(NewEnvironment, self).__init__(raw_env)
