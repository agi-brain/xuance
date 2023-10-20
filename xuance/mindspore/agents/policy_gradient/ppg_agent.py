from xuance.mindspore.agents import *
from xuance.mindspore.utils.distributions import CategoricalDistribution


class PPG_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: VecEnv,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.nenvs = envs.num_envs
        self.nsteps = config.nsteps
        self.nminibatch = config.nminibatch
        self.policy_nepoch = config.policy_nepoch
        self.value_nepoch = config.value_nepoch
        self.aux_nepoch = config.aux_nepoch
        self.render = config.render

        self.gamma = config.gamma
        self.lam = config.lam
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {"old_dist": None}

        writer = SummaryWriter(config.logdir)
        memory = DummyOnPolicyBuffer(self.observation_space,
                                     self.action_space,
                                     self.representation_info_shape,
                                     self.auxiliary_info_shape,
                                     self.nenvs,
                                     self.nsteps,
                                     self.nminibatch,
                                     self.gamma,
                                     self.lam)
        learner = PPG_Learner(policy,
                             optimizer,
                             scheduler,
                             writer,
                             config.modeldir,
                             config.ent_coef,
                             config.clip_range,
                             config.kl_beta)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(PPG_Agent, self).__init__(envs, policy, memory, learner, writer, config.logdir, config.modeldir)

    def _process_observation(self, observations):
        if self.use_obsnorm:
            if isinstance(self.observation_space, Dict):
                for key in self.observation_space.spaces.keys():
                    observations[key] = np.clip(
                        (observations[key] - self.obs_rms.mean[key]) / (self.obs_rms.std[key] + EPS),
                        -self.obsnorm_range, self.obsnorm_range)
            else:
                observations = np.clip((observations - self.obs_rms.mean) / (self.obs_rms.std + EPS),
                                       -self.obsnorm_range, self.obsnorm_range)
            return observations
        return observations

    def _process_reward(self, rewards):
        if self.use_rewnorm:
            std = np.clip(self.ret_rms.std, 0.1, 100)
            return np.clip(rewards / std, -self.rewnorm_range, self.rewnorm_range)
        return rewards

    def _action(self, obs):
        states, act_probs, vs, _ = self.policy(ms.Tensor(obs))
        # acts = self.policy.actor.sample(act_probs).asnumpy()
        dists = CategoricalDistribution(self.action_space.n)
        dists.set_param(act_probs)
        acts = dists.stochastic_sample().asnumpy()
        vs = vs.asnumpy()
        if context._get_mode() == 0:
            return {"state": states[0].asnumpy()}, acts, vs, split_distributions(dists)
        else:
            for key in states.keys():
                states[key] = states[key].asnumpy()
            return states, acts, vs, split_distributions(dists)

    def train(self, train_steps=10000, load_model=None):
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)

        obs = self.envs.reset()
        for step in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts, rets, dists = self._action(obs)
            next_obs, rewards, dones, infos = self.envs.step(acts)
            if self.render: self.envs.render()
            self.memory.store(obs, acts, self._process_reward(rewards), rets, dones, states, {"old_dist": dists})
            if self.memory.full:
                _, _, vals, _ = self._action(self._process_observation(next_obs))
                for i in range(self.nenvs):
                    self.memory.finish_path(rewards[i], i)
                
                # policy update
                # update_type: 0-policy_update, 1-critic_update, 2-auxiliary_update
                for _ in range(self.nminibatch * self.policy_nepoch):
                    obs_batch, act_batch, ret_batch, adv_batch, _, aux_batch = self.memory.sample()
                    self.learner.update(obs_batch, act_batch, ret_batch, adv_batch, aux_batch['old_dist'], 0)
                # critic update
                for _ in range(self.nminibatch * self.value_nepoch):
                    obs_batch, act_batch, ret_batch, adv_batch, _, aux_batch = self.memory.sample()
                    self.learner.update(obs_batch, act_batch, ret_batch, adv_batch, aux_batch['old_dist'], 1)
                    
                # update old_prob
                buffer_obs = self.memory.observations
                buffer_act = self.memory.actions
                _,new_dists,_,_ = self.policy(ms.Tensor(buffer_obs))
                new_dist = CategoricalDistribution(self.action_space.n)
                new_dist.set_param(new_dists)
                self.memory.auxiliary_infos['old_dist'] = split_distributions(new_dist)
                for _ in range(self.nminibatch * self.aux_nepoch):
                    obs_batch, act_batch, ret_batch, adv_batch, _, aux_batch = self.memory.sample()
                    self.learner.update(obs_batch, act_batch, ret_batch, adv_batch, aux_batch['old_dist'], 2)
                
                self.memory.clear()
            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            for i in range(self.nenvs):
                if dones[i] == True:
                    self.ret_rms.update(returns[i:i + 1])
                    self.memory.finish_path(0, i)
                    self.writer.add_scalars("returns-episode", {"env-%d" % i: scores[i]}, episodes[i])
                    self.writer.add_scalars("returns-step", {"env-%d" % i: scores[i]}, step)
                    scores[i] = 0
                    returns[i] = 0
                    episodes[i] += 1

            if step % 50000 == 0 or step == train_steps - 1:
                self.save_model()
                np.save(self.modeldir + "/obs_rms.npy",
                        {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

    def test(self, test_steps=10000, load_model=None):
        self.load_model(self.modeldir)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)

        obs = self.envs.reset()
        for _ in tqdm(range(test_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts, rets, logps = self._action(obs)
            next_obs, rewards, dones, infos = self.envs.step(acts)
            self.envs.render()
            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            for i in range(self.nenvs):
                if dones[i] == True:
                    scores[i], returns[i] = 0, 0
    
    def evaluate(self):
        pass
