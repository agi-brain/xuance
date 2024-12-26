### Deep Q-Network (DQN)

## What is DQN?

Deep Q-Network (DQN) is a foundational algorithm in DRL that integrates Q-learning,
a popular reinforcement learning method, with deep neural networks.
It was first introduced by a team at DeepMind in 2015 and demonstrated remarkable success
in playing Atari games directly from pixel inputs, achieving superhuman performance in many cases.

Q-Learning
............

Q-Learning is a model-free RL algorithm where the agent learns a Q-value function , which estimates the expected cumulative reward of taking an action ￼ in a state ￼ and following the optimal policy thereafter.
	•	The Bellman equation updates the Q-value:
￼
Here, ￼ is the learning rate, ￼ is the reward, ￼ is the discount factor, and ￼ is the next state.
	2.	Deep Neural Networks:
	•	Instead of storing the Q-values in a table (tabular Q-learning), DQN uses a deep neural network to approximate the Q-function, enabling it to handle high-dimensional state spaces like images.
	3.	Key Innovations:
	•	Experience Replay:
	•	A buffer stores the agent’s experiences ￼, and mini-batches of experiences are sampled randomly to train the network. This reduces correlation between consecutive samples and stabilizes training.
	•	Target Network:
	•	A separate target network is used to compute the Q-value targets during training. The target network is updated periodically to reduce oscillations and instability.
	•	Loss Function:
	•	The network is trained using the mean-squared error (MSE) loss between the predicted Q-value and the target:
￼
where ￼, and ￼ are the parameters of the target network.
	4.	Exploration-Exploitation Tradeoff:
	•	DQN uses an ￼-greedy strategy, where the agent explores random actions with probability ￼ and exploits the learned policy otherwise.

DQN Algorithm
	1.	Initialize the replay buffer, main Q-network, and target Q-network.
	2.	For each episode:
	•	Start in an initial state ￼.
	•	For each step in the episode:
	•	Choose an action ￼ using ￼-greedy.
	•	Execute ￼, observe reward ￼ and next state ￼.
	•	Store the experience ￼ in the replay buffer.
	•	Sample a random mini-batch from the replay buffer.
	•	Update the Q-network by minimizing the loss.
	•	Periodically update the target Q-network with the main Q-network weights.
	3.	Repeat until the policy converges or a performance threshold is reached.

Strengths of DQN
	•	Handles high-dimensional input spaces like images.
	•	Stabilizes Q-learning through techniques like experience replay and target networks.
	•	Demonstrated capability to outperform humans in several Atari games.

Limitations and Extensions
	•	DQN can struggle with unstable training, particularly in highly stochastic or complex environments.
	•	Extensions like Double DQN, Dueling DQN, and Prioritized Experience Replay were proposed to address its weaknesses and improve performance.

DQN laid the groundwork for modern DRL methods and continues to inspire new algorithms in the field.

How To Use DQN in XuanCe
''''''''''''''''''''''''''


APIs
'''''''''''''''

PyTorch
...............

.. automodule:: xuance.torch.agents.qlearning_family.dqn_agent
    :members:
    :undoc-members:
    :show-inheritance:

TensorFlow2
...............

.. automodule:: xuance.tensorflow.agents.qlearning_family.dqn_agent
    :members:
    :undoc-members:
    :show-inheritance:

MindSpore
...............

.. automodule:: xuance.mindspore.agents.qlearning_family.dqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
