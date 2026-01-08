import os
import platform
import xuance
import tensorflow as tf
from abc import ABC, abstractmethod
from xuance.common import Optional
from xuance.environment import make_envs
from xuance.tensorflow.agents import Agent


class RunnerBase(ABC):
    """Abstract base class for all runners in XuanCe.

    RunnerBase defines the common interface and shared infrastructure for
    all concrete runners (e.g., DRLRunner, MARLRunner).

    A runner is responsible for experiment orchestration, including:
        - Environment lifecycle management.
        - Training / testing / benchmarking workflows.
        - Resource ownership and cleanup semantics.
        - Rank-aware logging in distributed settings.

    Notes:
        - Algorithm-specific logic must remain in the Agent.
        - Subclasses must implement the abstract methods declared here.
    """
    def __init__(self, config, envs=None, agent=None, manage_resources=None):
        """Initialize the runner base.

        This constructor handles environment creation and resource ownership
        inference. By default, environments are created internally using
        the provided configuration.

        Args:
            config: Experiment configuration object.
            envs (optional): Pre-created environments. If None, environments
                will be created internally via `make_envs(config)`.
            agent (optional): Pre-created agent instance. Ownership is inferred
                based on whether the agent is injected externally.
            manage_resources (optional): Whether the runner takes ownership of
                resource lifecycle management (envs.close(), agent.finish()).
                - If None, ownership is inferred automatically:
                  `manage_resources = (_own_envs or _own_agent)`.
                - If True, the runner will finalize/close any existing `agent`
                  and `envs` held by this runner instance at the end of `run()`,
                  even if they were injected externally.
                - If False, the caller is responsible for closing/finalizing
                  injected resources.
        """
        # Build or attach environments
        if envs is None:
            # Runner owns environments created internally
            self.envs = make_envs(config)
            self._own_envs = True
        else:
            # Environments are injected externally
            self.envs = envs
            self._own_envs = False

        # Agent ownership flag (actual agent creation happens in subclasses)
        self._own_agent = True if agent is None else False

        # Infer resource management responsibility
        self.manage_resources = (
                self._own_envs or self._own_agent
        ) if manage_resources is None else manage_resources

        # Reset environments once at initialization
        # Note: This assumes online RL-style runners. Subclasses should be aware
        # of this design choice if different behavior is required.
        self.envs.reset()

        # Number of parallel environments
        self.n_envs = self.envs.num_envs

        # Default rank (may be overridden by distributed runners)
        self.rank = 0

        self.agent: Optional[Agent] = agent

    @abstractmethod
    def _run_train(self, **kwargs):
        """Execute the training workflow.

        Subclasses must implement this method to define how training is performed.
        """
        pass

    @abstractmethod
    def _run_test(self, **kwargs):
        """Execute the testing/evaluation workflow.

        Subclasses must implement this method to define how evaluation is performed.
        """
        pass

    @abstractmethod
    def _run_benchmark(self, **kwargs):
        """Execute the benchmarking workflow.

        Subclasses must implement this method to define how benchmarking
        (training + periodic evaluation) is performed.
        """
        pass

    def run(self, mode: str = "train", **kwargs):
        """Run the experiment.

        This method serves as the main entry point of the runner.
        Concrete runners must implement this method or rely on a
        template-method implementation provided by the base class.
        """
        handlers = {
            "train": self._run_train,
            "test": self._run_test,
            "benchmark": self._run_benchmark
        }
        if mode not in handlers:
            raise ValueError(
                f"Invalid run mode: '{mode}'. "
                "Supported modes are: 'train', 'test', and 'benchmark'."
            )

        try:
            return handlers[mode](**kwargs)
        finally:
            self._finalize()

    def collect_device_info(self) -> dict:
        """Collect runtime device / system info for reproducibility (TensorFlow 2.x).

        Returns a JSON-serializable dict.
        """
        info = {
            "Platform": platform.platform(),
            "Python": platform.python_version(),
            "XuanCe": xuance.__version__,
            "PID": os.getpid(),
            "Rank": int(getattr(self, "rank", 0)),
        }

        try:
            info["TensorFlow"] = getattr(tf, "__version__", "unknown")

            # Physical devices visible to TF
            gpus = tf.config.list_physical_devices("GPU")
            cpus = tf.config.list_physical_devices("CPU")

            info["CUDA_Available"] = bool(gpus)  # “TF能看到GPU”通常就是最关心的
            info["num_gpus"] = len(gpus)
            info["num_cpus"] = len(cpus)

            # GPU details (best-effort; names are not always available)
            gpu_details = []
            for i, d in enumerate(gpus):
                # d.name often like '/physical_device:GPU:0'
                gpu_details.append({"index": i, "name": getattr(d, "name", str(d)), "device_type": "GPU"})
            info["gpus"] = gpu_details

            # Logical devices (useful when virtual GPUs / memory limits are set)
            logical_gpus = tf.config.list_logical_devices("GPU")
            info["num_logical_gpus"] = len(logical_gpus)

            # Build info sometimes contains cuda/cudnn versions (not always present)
            build_info = {}
            try:
                build_info = tf.sysconfig.get_build_info() or {}
            except Exception:
                build_info = {}

            # These keys vary across TF versions; keep it best-effort & JSON-safe
            if build_info:
                info["tf_build_info"] = {k: str(v) for k, v in build_info.items()}

            # Optional: record current visible devices env var (helps debug)
            info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", None)

            # Optional: if your Agent exposes its own device/strategy info, store it
            agent = getattr(self, "agent", None)
            if agent is not None:
                # common patterns: agent.device / agent.strategy
                if hasattr(agent, "device"):
                    info["agent_device"] = str(getattr(agent, "device"))
                if hasattr(agent, "strategy"):
                    try:
                        info["tf_strategy"] = type(getattr(agent, "strategy")).__name__
                    except Exception:
                        pass

        except Exception as e:
            # Keep it minimal but valid if TF isn't available or anything fails.
            info["CUDA_Available"] = False
            info["device_info_error"] = repr(e)

        return info

    def _finalize(self):
        """Finalize resources held by the runner.

        This method releases resources referenced by this runner instance, such
        as environments and the agent. It is typically invoked in a `finally`
        block to ensure cleanup even when exceptions occur.

        Cleanup rules:
            - If `manage_resources` is False, this method performs no action.
            - If `manage_resources` is True, the runner takes ownership of the
              lifecycle of any `agent` and `envs` attached to this instance and
              will call `agent.finish()` / `envs.close()` when available,
              regardless of whether they were created internally or injected
              externally.

        Notes:
            - Subclasses may override this method to add extra cleanup steps, but
              should generally call `super()._finalize()` to preserve base cleanup.
            - After closing envs, `self.envs` is set to None to help prevent
              accidental re-use.
        """
        if getattr(self, "manage_resources", True):
            # Finalize agent if it exists.
            if hasattr(self, "agent") and getattr(self, "agent") is not None:
                self.agent.finish()
            if hasattr(self, "agents") and getattr(self, "agents") is not None:
                self.agents.finish()

            # Close environments if they exist.
            if hasattr(self, "envs") and self.envs is not None:
                self.envs.close()
                # Help prevent accidental re-use after closing.
                self.envs = None
        else:
            return
