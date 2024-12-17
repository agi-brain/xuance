New Algorithm
=========================================================

We allow users create their own customized algorithm outside of the default in XuanCe.

This tutorial walks you through the process of creating, training,
and testing a custom off-policy reinforcement learning (RL) agent using the XuanCe framework.
The demo involves defining a custom policy, learner, and agent while using XuanCe’s modular architecture for RL experiments.

Step 1: Define the Policy Module
-------------------------------------------------------------

The policy is the brain of the agent.
It maps observations to actions, optionally through a value function. Here, we define a custom policy MyPolicy:

.. code-block:: python

    class MyPolicy(nn.Module):
    """
    An example of self-defined policy.

    Args:
        representation (nn.Module): A neural network module responsible for extracting meaningful features from the raw observations provided by the environment.
        hidden_dim (int): Specifies the number of units in each hidden layer, determining the model’s capacity to capture complex patterns.
        n_actions (int): The total number of discrete actions available to the agent in the environment.
        device (torch.device): The calculating device.


    Note: The inputs to the __init__ method are not rigidly defined. You can extend or modify them as needed to accommodate additional settings or configurations specific to your application.
    """

        def __init__(self, representation: nn.Module, hidden_dim: int, n_actions: int, device: torch.device):
            super(MyPolicy, self).__init__()
            self.representation = representation  # Specify the representation.
            self.feature_dim = self.representation.output_shapes['state'][0]  # Dimension of the representation's output.
            self.q_net = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            ).to(device)  # The Q network.
            self.target_q_net = deepcopy(self.q_net)  # Target Q network.

        def forward(self, observation):
            output_rep = self.representation(observation)  # Get the output of the representation module.
            output = self.q_net(output_rep['state'])  # Get the output of the Q network.
            argmax_action = output.argmax(dim=-1)  # Get greedy actions.
            return output_rep, argmax_action, output

        def target(self, observation):
            outputs_target = self.representation(observation)  # Get the output of the representation module.
            Q_target = self.target_q_net(outputs_target['state'])  # Get the output of the target Q network.
            argmax_action = Q_target.argmax(dim=-1)  # Get greedy actions that output by target Q network.
            return outputs_target, argmax_action.detach(), Q_target.detach()

        def copy_target(self):  # Reset the parameters of target Q network as the Q network.
            for ep, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                tp.data.copy_(ep)


Key Points:

- representation module: Extracts state features, decoupling feature engineering from Q-value computation.
- networks: The policy uses a feedforward neural network to calculate actions and estimate Q-values.
- device: The device choice should align with that of the other modules.

Step 2: Define the Learner
-------------------------------------------------------------

The learner manages the policy optimization process,
including computing loss, performing gradient updates, and synchronizing target networks.

.. code-block:: python

    class MyLearner(Learner):
        def __init__(self, config, policy):
            super(MyLearner, self).__init__(config, policy)
            # Build the optimizer.
            self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
            self.loss = nn.MSELoss()  # Build a loss function.
            self.sync_frequency = config.sync_frequency  # The period to synchronize the target network.

        def update(self, **samples):
            info = {}
            self.iterations += 1
            '''Get a batch of training samples.'''
            obs_batch = torch.as_tensor(samples['obs'], device=self.device)
            act_batch = torch.as_tensor(samples['actions'], device=self.device)
            next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
            rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
            ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)

            # Feedforward steps.
            _, _, q_eval = self.policy(obs_batch)
            _, _, q_next = self.policy.target(next_batch)
            q_next_action = q_next.max(dim=-1).values
            q_eval_action = q_eval.gather(-1, act_batch.long().unsqueeze(-1)).reshape(-1)
            target_value = rew_batch + (1 - ter_batch) * self.gamma * q_next_action
            loss = self.loss(q_eval_action, target_value.detach())

            # Backward and optimizing steps.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Synchronize the target network
            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

            # Set the variables you need to observe.
            info.update({'loss': loss.item(),
                         'iterations': self.iterations,
                         'q_eval_action': q_eval_action.mean().item()})

            return info

Key Points:

- optimizer: The pytorch's optimizer should be selected in the __init__ method.
- update: In this method, we can get a batch of samples and use them to calculate loss values and back propagation.
- info: The users can add arbitrarily .

Step 3: Define the Agent
-------------------------------------------------------------

The agent combines the policy, learner, and environment interaction to create a complete RL pipeline.

.. code-block:: python

    class MyAgent(OffPolicyAgent):
        def __init__(self, config, envs):
            super(MyAgent, self).__init__(config, envs)
            self.policy = self._build_policy()  # Build the policy module.
            self.memory = self._build_memory()  # Build the replay buffer.
            REGISTRY_Learners['MyLearner'] = MyLearner  # Registry your pre-defined learner.
            self.learner = self._build_learner(self.config, self.policy)  # Build the learner.

        def _build_policy(self):
            # First create the representation module.
            representation = self._build_representation("Basic_MLP", self.observation_space, self.config)
            # Build your customized policy module.
            policy = MyPolicy(representation, 64, self.action_space.n, self.config.device)
            return policy

Key Points:

- Policy: Build the custom policy and learner defined earlier.
- Memory: Build experience replay to break correlations in training data.
- Learner: Register MyLearner for easy configuration.

Step 4: Build and Run Your Agent.
-------------------------------------------------------------

Finally, we can create the agent and make environments to train the model.

.. code-block:: python

    if __name__ == '__main__':
        config = get_configs(file_dir="./new_rl.yaml")  # Get the config settings from .yaml file.
        config = Namespace(**config)  # Convert the config from dict to argparse.
        envs = make_envs(config)  # Make vectorized environments.
        agent = MyAgent(config, envs)  # Instantiate your pre-build agent class.

        if not config.test_mode:  # Training mode.
            agent.train(config.running_steps // envs.num_envs)  # Train your agent.
            agent.save_model("final_train_model.pth")  # After training, save the model.
        else:  # Testing mode.
            config.parallels = 1  # Test on one environment.
            env_fn = lambda: make_envs(config)  # The method to create testing environment.
            agent.load_model(agent.model_dir_load)  # Load pre-trained model.
            scores = agent.test(env_fn, config.test_episode)  # Test your agent.

        agent.finish()  # Finish the agent.
        envs.close()  # Close the environments.

The source code of this example can be visited at the following link:

`https://github.com/agi-brain/xuance/blob/master/examples/new_algorithm/new_rl.py <https://github.com/agi-brain/xuance/blob/master/examples/new_algorithm/new_rl.py>`_
