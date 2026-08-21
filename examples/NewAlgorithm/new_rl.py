import torch
import torch.nn as nn
from copy import deepcopy
from argparse import Namespace
from xuance.common import load_yaml
from xuance.environment import make_envs
from xuance.torch.agents import OffPolicyAgent
from xuance.torch.learners import Learner, REGISTRY_Learners
from xuance.torch.rl_models.modules import ModelOutput


# Step 1: Create a policy.
class MyModel(nn.Module):
    """
    An example of self-defined policy.

    Args:
        representation (nn.Module): A neural network module responsible for extracting meaningful features from the raw observations provided by the environment.
        hidden_dim (int): Specifies the number of units in each hidden layer, determining the model’s capacity to capture complex patterns.
        n_actions (int): The total number of discrete actions available to the agent in the environment.
        device (torch.device): The calculating device.

    Note: The inputs to the __init__ algo are not rigidly defined. You can extend or modify them as needed to accommodate additional settings or configurations specific to your application.
    """

    def __init__(self, representation: nn.Module, hidden_dim: int, n_actions: int, device: torch.device):
        super(MyModel, self).__init__()
        self.representation = representation  # Specify the representation.
        self.feature_dim = self.representation.output_shapes['state'][0]  # Dimension of the representation's output.
        self.q_net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        ).to(device)  # The Q network.
        self.target_representation = deepcopy(self.representation)
        self.target_q_net = deepcopy(self.q_net)  # Target Q network.

    def forward(self, observation) -> ModelOutput:
        output_rep = self.representation(observation)  # Get the output of the representation module.
        output = self.q_net(output_rep.embeddings)  # Get the output of the Q network.
        argmax_action = output.argmax(dim=-1)  # Get greedy actions.
        return ModelOutput(
            rep_out=output_rep,
            actions=argmax_action,
            values=output
        )

    def act(self, observation):
        return self(observation).actions

    def target(self, observation) -> ModelOutput:
        outputs_target = self.representation(observation)  # Get the output of the representation module.
        Q_target = self.target_q_net(outputs_target.embeddings)  # Get the output of the target Q network.
        argmax_action = Q_target.argmax(dim=-1)  # Get greedy actions that output by target Q network.
        return ModelOutput(
            rep_out=outputs_target,
            actions=argmax_action.detach(),
            values=Q_target.detach()
        )

    def copy_target(self):  # Reset the parameters of target Q network as the Q network.
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            tp.data.copy_(ep)


# Step 2: Create the learner.
class MyLearner(Learner):
    def __init__(self, config, model, callback):
        super(MyLearner, self).__init__(config, model, callback)
        # Build the optimizer.
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate, eps=1e-5)
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
        model_output = self.model(obs_batch)
        q_eval = model_output.values
        target_model_output = self.model.target(next_batch)
        q_next = target_model_output.values
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
            self.model.copy_target()

        # Set the variables you need to observe.
        info.update({'loss': loss.item(),
                     'iterations': self.iterations,
                     'q_eval_action': q_eval_action.mean().item()})

        return info


# Step 3: Create the agent.
class MyAgent(OffPolicyAgent):
    def __init__(self, config, envs, callback=None):
        super(MyAgent, self).__init__(config, envs, callback)
        self.model = self._build_model()  # Build the policy module.
        self.memory = self._build_memory()  # Build the replay buffer.
        REGISTRY_Learners['MyLearner'] = MyLearner  # Registry your pre-defined learner.
        self.learner = self._build_learner(self.config, self.model, self.callback)  # Build the learner.

    def _build_model(self) -> nn.Module:
        # First create the representation module.
        representation = self._build_representation("Basic_MLP", self.observation_space, self.config)
        # Build your custom policy module.
        model = MyModel(representation, 64, self.action_space.n, self.config.device)
        return model


if __name__ == '__main__':
    config = load_yaml(file_dir="new_rl.yaml")  # Get the config settings from .yaml file.
    config = Namespace(**config)  # Convert the config from dict to argparse.
    envs = make_envs(config)  # Make vectorized environments.
    agent = MyAgent(config, envs)  # Instantiate your pre-build agent class.

    if not config.test_mode:  # Training mode.
        agent.train(config.running_steps // envs.num_envs)  # Train your agent.
        agent.save_model("final_train_model.pth")  # After training, save the model.
    else:  # Testing mode.
        config.parallels = 1  # Test on one environment.
        env_fn = lambda: make_envs(config)  # The algo to create testing environment.
        agent.load_model(agent.model_dir_load)  # Load pre-trained model.
        scores = agent.test(env_fn, config.test_episode)  # Test your agent.

    agent.finish()  # Finish the agent.
    envs.close()  # Close the environments.
