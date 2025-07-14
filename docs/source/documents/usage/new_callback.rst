Customized Callback
---------------------------------

The agents in XuanCe support the injection of user-defined callbacks of enhanced customization
and control during training and testing.

You can create a subclass of ``BaseCallback``, override any of the following methods,
and pass an instance of your callback to the agent:

Available callback hooks:

- ``on_update_start(...)``: Called before the policy update begins.
- ``on_update_end(...)``: Called after the policy update is completed.
- ``on_train_step(...)``: Called after each training step.
- ``on_train_epochs_end(...)``: Called after each training epoch (i.e., after collecting one transition).
- ``on_train_episode_info(...)``: Called at the termination or truncation of one episode for an environment.
- ``on_train_step_end(...)``: Called after a training step is completed (includes update, logging, etc.).
- ``on_test_step(...)``: Called during each step in the testing loop.
- ``on_test_end(...)``: Called at the end of the testing loop
- ``on_update_agent_wise(...)``: Called after updating an agent's policy.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example demonstrating how to create a custom callback to inject hooks during training.
In this example, additional environment-related information is visualized in TensorBoard.

Before using the callback, ensure that the info dictionary returned by the environment's step() function contains the keys 'info_1' and 'info_2'.
These values can then be logged using SummaryWriter and displayed in TensorBoard.

The example code is provided below:

.. code-block:: python

    import os
    from xuance.torch.agents import BaseCallback
    from torch.utils.tensorboard import SummaryWriter

    class MyCallback(BaseCallback):
    "The customized callback."
    def __init__(self, config):
        super(MyCallback, self).__init__()
        log_dir = os.path.join(os.getcwd(), config.log_dir, 'callback_info')
        create_directory(log_dir)
        self.writer = SummaryWriter(log_dir)

    def on_train_episode_info(self, *args, **kwargs):
        "Visualize the additional information about the environment on Tensorboard."
        infos = kwargs['infos']
        env_id = kwargs['env_id']
        step = kwargs['current_step']
        self.writer.add_scalars('environment_information/info_1', {f"env-{env_id}": infos[env_id]["info_1"]}, step)
        self.writer.add_scalars('environment_information/info_2', {f"env-{env_id}": infos[env_id]["info_2"]}, step)

    Agent = DQN_Agent(config=configs, envs=envs, callback=MyCallback(configs))  # Create a DDPG agent with customized callback.

Full code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The full code for the example can be visited in this link: `https://github.com/agi-brain/xuance/blob/master/examples/new_environments/ddpg_new_env.py <https://github.com/agi-brain/xuance/blob/master/examples/new_environments/ddpg_new_env.py>`_
