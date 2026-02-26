import os
from xuance.common import get_arguments
from .run_basic import RunnerBase
from .run_drl import RunnerDRL
from .run_marl import RunnerMARL
from .run_sc2 import RunnerSC2
from .run_football import RunnerFootball
from .run_competition import RunnerCompetition
from .run_offlinerl import RunnerOfflineRL


REGISTRY_Runner = {
    "DRL": RunnerDRL,  # For single-agent DRL
    "MARL": RunnerMARL,  # For MARL
    "RunnerStarCraft2": RunnerSC2,  # For StarCraft MARL
    "RunnerFootball": RunnerFootball,   # For GoogleFootballResearch MARL
    "RunnerCompetition": RunnerCompetition,  # For MARL with competing tasks
    "OfflineRL": RunnerOfflineRL,  # For Offline RL
}

__all__ = [
    "RunnerBase",
    "RunnerDRL",
    "RunnerMARL",
    "RunnerSC2",
    "RunnerFootball",
    "RunnerCompetition",
    "RunnerOfflineRL",
    "REGISTRY_Runner",
]


def get_runner(
        algo,
        env,
        env_id,
        config_path=None,
        parser_args=None
) -> RunnerBase:
    """
    This method returns a runner that specified by the users according to the inputs.
    Args:
        algo: the algorithm name that will be implemented,
        env: The name of the environment,
        env_id: The name of the scenario in the environment.
        config_path: default is None, if None, the default configs (``xuance/configs/.../*.yaml``) will be loaded.
        parser_args: arguments that specified by parser tools.

    Returns:
        An implementation of a runner that enables to run the DRL algorithms.
    """
    args = get_arguments(algo, env, env_id, config_path, parser_args)

    if type(args) == list:
        device = args[0].device
        distributed_training = True if args[0].distributed_training else False
    else:
        device = args.device
        distributed_training = True if args.distributed_training else False

    dl_toolbox = args[0].dl_toolbox if type(args) == list else args.dl_toolbox  # The choice of deep learning toolbox.
    rank = 0  # Avoid printing the same information when using distributed training.

    if dl_toolbox == "torch":
        rank = int(os.environ['RANK']) if distributed_training else 0
        if rank == 0:
            print("Deep learning toolbox: PyTorch.")

    elif dl_toolbox == "mindspore":
        import mindspore as ms
        print("Deep learning toolbox: MindSpore.")
        ms.set_device(device_target=device)  # Set the calculating device.

    elif dl_toolbox == "tensorflow":
        print("Deep learning toolbox: TensorFlow.")
        if device in ["cpu", "CPU"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    else:
        if dl_toolbox == '':
            raise AttributeError("You have to assign a deep learning toolbox")
        else:
            raise AttributeError("Cannot find a deep learning toolbox named " + dl_toolbox)

    if distributed_training:
        if rank == 0:
            print(f"Calculating device: Multi-GPU distributed training.")
    else:
        print(f"Calculating device: {device}")

    if type(args) == list:
        agents_name_string = []
        for i_alg in range(len(algo)):
            if i_alg < len(algo) - 1:
                agents_name_string.append(args[i_alg].agent + " vs")
            else:
                agents_name_string.append(args[i_alg].agent)
            args[i_alg].agent_name = algo[i_alg]
            relative_log_dir = getattr(args[i_alg], "log_dir", f"logs/{algo}")
            relative_model_dir = getattr(args[i_alg], "model_dir", f"logs/{algo}")
            args[i_alg].log_dir = os.path.join(relative_log_dir, args[i_alg].env_id, f"side_{i_alg}")
            args[i_alg].model_dir = os.path.join(relative_model_dir, args[i_alg].env_id, f"side_{i_alg}")
            args[i_alg].result_dir = os.path.join(f"results/{algo}", args[i_alg].env_id, f"side_{i_alg}")

        if rank == 0:
            print("Algorithm:", *agents_name_string)
            print("Environment:", args[0].env_name)
            print("Scenario:", args[0].env_id)
        runner_name = args[0].runner
        for arg in args:
            if arg.runner == runner_name:
                runner_name = arg.runner
            else:
                raise AttributeError("The runner should remain consistent across different agents.")
        if runner_name != "random":
            runner = REGISTRY_Runner[runner_name](args)
            return runner
        raise AttributeError("Both sides of policies are random!")
    else:
        args.agent_name = algo
        relative_log_dir = getattr(args, "log_dir", f"logs/{algo}")
        relative_model_dir = getattr(args, "model_dir", f"logs/{algo}")
        args.log_dir = os.path.join(relative_log_dir, args.env_id)
        args.model_dir = os.path.join(relative_model_dir, args.env_id)
        args.result_dir = os.path.join(f"results/{algo}", args.env_id)
        if rank == 0:
            print("Algorithm:", args.agent)
            print("Environment:", args.env_name)
            print("Scenario:", args.env_id)
        runner = REGISTRY_Runner[args.runner](args)
        return runner
