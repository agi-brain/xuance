from xuance.environment.new_env_mas import DummyVecEnv_New_MAS, SubprocVecEnv_New_MAS


class DummyVecEnv_RoboticWarehouse(DummyVecEnv_New_MAS):
    def __init__(self, env_fns):
        super(DummyVecEnv_RoboticWarehouse, self).__init__(env_fns)


class SubprocVecEnv_RoboticWarehouse(SubprocVecEnv_New_MAS):
    def __init__(self, env_fns, context='spawn', in_series=1):
        super(SubprocVecEnv_RoboticWarehouse, self).__init__(env_fns, context, in_series)
