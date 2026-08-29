import numpy as np
from xuance.tensorflow import tf, Module, ModuleList, Tensor


class ParameterizedDQN(Module):
    def __init__(self,
                 continuous_actor: Module,
                 q_network: Module,
                 **kwargs):
        super().__init__(**kwargs)
        self.continuous_actor = continuous_actor
        self.q_network = q_network
        self.target_continuous_actor = self.continuous_actor.clone(copy_weights=True, trainable=False,
                                                                   name="target_continuous_actor")
        self.target_q_network = self.q_network.clone(copy_weights=True,
                                                     trainable=False, name="target_q_network")

    def Atarget(self, observations: Tensor):
        return self.target_continuous_actor(observations).actions

    def call(self, observations: Tensor):
        return self.continuous_actor(observations).actions

    def Qtarget(self, observations: Tensor, actions: Tensor):
        return self.target_q_network(observations, actions).values

    def Qeval(self, observations: Tensor, actions: Tensor):
        return self.q_network(observations, actions).values

    def Qpolicy(self, observations: Tensor):
        continuous_actions = self.continuous_actor(observations).actions
        policy_q = tf.reduce_sum(self.q_network(observations, continuous_actions).values)
        return policy_q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.continuous_actor.variables, self.target_continuous_actor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.q_network.variables, self.target_q_network.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class MultipassParameterizedDQN(ParameterizedDQN):
    def __init__(self,
                 continuous_actor: Module,
                 q_network: Module,
                 conact_sizes: np.ndarray,
                 **kwargs):
        super().__init__(continuous_actor, q_network, **kwargs)
        self.offsets = conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.num_disact = self.q_network.num_disact

    def Qtarget(self, observations: Tensor, actions: Tensor):
        actions_all = []
        for i in range(self.num_disact):
            start = self.offsets[i]
            end = self.offsets[i + 1]

            action_i = tf.concat([tf.zeros_like(actions[:, :start]),
                                  actions[:, start:end],
                                  tf.zeros_like(actions[:, end:])], axis=-1)
            actions_all.append(action_i)

        actions_input = tf.concat(actions_all, axis=0)
        observations_all = tf.tile(observations, [self.num_disact, 1])

        target_qall = self.target_q_network(observations_all, actions_input).values
        target_qall = tf.reshape(target_qall, [self.num_disact, -1, self.num_disact])

        Q = tf.stack([target_qall[i, :, i] for i in range(self.num_disact)], axis=-1)

        return Q

    def Qeval(self, observations: Tensor, actions: Tensor):
        actions_all = []
        for i in range(self.num_disact):
            start = self.offsets[i]
            end = self.offsets[i + 1]

            action_i = tf.concat([tf.zeros_like(actions[:, :start]),
                                  actions[:, start:end],
                                  tf.zeros_like(actions[:, end:])], axis=-1)
            actions_all.append(action_i)

        actions_input = tf.concat(actions_all, axis=0)
        observations_all = tf.tile(observations, [self.num_disact, 1])

        eval_qall = self.q_network(observations_all, actions_input).values
        eval_qall = tf.reshape(eval_qall, [self.num_disact, -1, self.num_disact])

        Q = tf.stack([eval_qall[i, :, i] for i in range(self.num_disact)], axis=-1)

        return Q

    def Qpolicy(self, observations: Tensor):
        conact = self.continuous_actor(observations).actions
        Q = self.Qeval(observations, conact)
        return Q


class SplitParameterisedDQN(MultipassParameterizedDQN):
    def __init__(self,
                 continuous_actor: Module,
                 q_network: ModuleList,
                 conact_sizes: np.ndarray,
                 num_disact: int,
                 **kwargs):
        super(MultipassParameterizedDQN, self).__init__(continuous_actor, q_network, **kwargs)
        self.offsets = conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.num_disact = num_disact

    def Qtarget(self, observations: Tensor, actions: Tensor):
        target_Q_list = []
        for i in range(self.num_disact):
            conact = actions[:, self.offsets[i]:self.offsets[i + 1]]
            target_q = self.target_q_network[i](observations, conact).values
            target_Q_list.append(tf.expand_dims(target_q, axis=-1))
        target_Q_value = tf.concat(target_Q_list, axis=1)
        return target_Q_value

    def Qeval(self, observations: Tensor, actions: Tensor):
        Q = []
        for i in range(self.num_disact):
            conact = actions[:, self.offsets[i]:self.offsets[i + 1]]
            eval_q = self.q_network[i](observations, conact).values
            Q.append(tf.expand_dims(eval_q, axis=-1))
        Q = tf.concat(Q, axis=1)
        return Q

    def Qpolicy(self, observations: Tensor):
        conacts = self.continuous_actor(observations).actions
        Q = self.Qeval(observations, conacts)
        return Q
