from argparse import Namespace
from .robotic_warehouse_env import RoboticWarehouseEnv
from .robotic_warehouse_vec_env import DummyVecEnv_RoboticWarehouse, SubprocVecEnv_RoboticWarehouse

ENV_IDs = {
    "rware-tiny-2ag-v1": "rware:rware-tiny-2ag-v1"
}


def make_envs(config: Namespace):
    def _thunk():
        return RoboticWarehouseEnv(config, render_mode=config.render_mode)

    if config.vectorize == "Dummy_RoboticWarehouse":
        return DummyVecEnv_RoboticWarehouse([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "Subproc_RoboticWarehouse":
        return SubprocVecEnv_RoboticWarehouse([_thunk for _ in range(config.parallels)])
    else:
        raise NotImplementedError
