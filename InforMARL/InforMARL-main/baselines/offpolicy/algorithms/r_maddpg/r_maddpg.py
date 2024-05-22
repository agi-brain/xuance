import torch
import numpy as np
import copy
import itertools
from baselines.offpolicy.utils.util import huber_loss, mse_loss, to_torch
from baselines.offpolicy.utils.popart import PopArt
from baselines.offpolicy.algorithms.base.trainer import Trainer


class R_MADDPG(Trainer):
    def __init__(
        self,
        args,
        num_agents,
        policies,
        policy_mapping_fn,
        device=None,
        episode_length=None,
        actor_update_interval=1,
    ):
        """
        Trainer class for recurrent MADDPG/MATD3. See parent class for more information.
        :param episode_length: (int) maximum length of an episode.
        :param actor_update_interval: (int) number of critic updates to perform between every update to the actor.
        """
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta
        self.tpdv = dict(dtype=torch.float32, device=device)

        if episode_length is None:
            self.episode_length = self.args.episode_length
        else:
            self.episode_length = episode_length

        self.num_agents = num_agents
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {
            policy_id: sorted(
                [
                    agent_id
                    for agent_id in range(self.num_agents)
                    if self.policy_mapping_fn(agent_id) == policy_id
                ]
            )
            for policy_id in self.policies.keys()
        }
        if self.use_popart:
            self.value_normalizer = {
                policy_id: PopArt(1) for policy_id in self.policies.keys()
            }
        self.actor_update_interval = actor_update_interval
        self.num_updates = {p_id: 0 for p_id in self.policy_ids}
        self.use_same_share_obs = self.args.use_same_share_obs

    def get_update_info(self, update_policy_id, obs_batch, act_batch, avail_act_batch):
        """
        Form centralized observation and action info.
        :param update_policy_id: (str) id of policy being updated.
        :param obs_batch: (np.ndarray) batch of observation sequences sampled from buffer.
        :param act_batch: (np.ndarray) batch of action sequences sampled from buffer.
        :param avail_act_batch: (np.ndarray) batch of available action sequences sampled from buffer. None if environment does not limit actions.

        :return cent_act_sequence_critic: (np.ndarray) batch of centralized action sequences for critic input.
        :return act_sequences: (list) list of action sequences corresponding to each agent.
        :return act_sequence_replace_ind_start: (int) index of act_sequences from which to replace actions for actor update.
        :return cent_nact_sequence: (np.ndarray) batch of centralize next step action sequences.
        """
        act_sequences = []
        nact_sequences = []
        act_sequence_replace_ind_start = None
        # iterate through policies to get the target acts and other centralized info
        ind = 0
        for p_id in self.policy_ids:
            policy = self.policies[p_id]
            if p_id == update_policy_id:
                # where to start replacing actor actions from during actor update
                act_sequence_replace_ind_start = ind
            num_pol_agents = len(self.policy_agents[p_id])
            act_sequences.append(list(act_batch[p_id]))
            batched_obs_seq = np.concatenate(obs_batch[p_id], axis=1)
            # same with buffer actions and available actions
            batched_act_seq = np.concatenate(act_batch[p_id], axis=1)
            if avail_act_batch[p_id] is not None:
                batched_avail_act_seq = np.concatenate(avail_act_batch[p_id], axis=1)
            else:
                batched_avail_act_seq = None
            total_batch_size = batched_obs_seq.shape[1]
            batch_size = total_batch_size // num_pol_agents
            # no gradient tracking is necessary for target actions
            with torch.no_grad():
                # step target actor through the first actions
                if isinstance(policy.act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(policy.act_dim))
                else:
                    sum_act_dim = policy.act_dim
                batched_prev_act_seq = np.concatenate(
                    (
                        np.zeros((1, total_batch_size, sum_act_dim), dtype=np.float32),
                        batched_act_seq[:-1],
                    )
                )
                pol_nact_seq, _, _ = policy.get_actions(
                    batched_obs_seq,
                    batched_prev_act_seq,
                    policy.init_hidden(-1, total_batch_size),
                    available_actions=batched_avail_act_seq,
                    use_target=True,
                )
                # remove the first timestep for next actions
                pol_nact_seq = pol_nact_seq[1:]
                # separate the actions into individual agent actions
                agent_nact_seqs = pol_nact_seq.cpu().split(split_size=batch_size, dim=1)
            # cat to form centralized next step action
            nact_sequences.append(torch.cat(agent_nact_seqs, dim=-1))
            # increase ind by number agents just processed
            ind += num_pol_agents
        # form centralized observations and actions by concatenating
        # flatten list of lists
        act_sequences = list(itertools.chain.from_iterable(act_sequences))
        cent_act_sequence_critic = np.concatenate(act_sequences, axis=-1)
        cent_nact_sequence = np.concatenate(nact_sequences, axis=-1)

        return (
            cent_act_sequence_critic,
            act_sequences,
            act_sequence_replace_ind_start,
            cent_nact_sequence,
        )

    def train_policy_on_batch(self, update_policy_id, batch):
        """See parent class."""
        if self.use_same_share_obs:
            return self.shared_train_policy_on_batch(update_policy_id, batch)
        else:
            return self.cent_train_policy_on_batch((update_policy_id, batch))

    def shared_train_policy_on_batch(self, update_policy_id, batch):
        """Training function when all agents share the same centralized observation. See train_policy_on_batch."""
        # unpack the batch
        (
            obs_batch,
            cent_obs_batch,
            act_batch,
            rew_batch,
            dones_batch,
            dones_env_batch,
            avail_act_batch,
            importance_weights,
            idxes,
        ) = batch

        train_info = {}

        update_actor = (
            self.num_updates[update_policy_id] % self.actor_update_interval == 0
        )

        # number of agents controlled by update policy
        num_update_agents = len(self.policy_agents[update_policy_id])
        update_policy = self.policies[update_policy_id]
        batch_size = obs_batch[update_policy_id].shape[2]
        total_batch_size = batch_size * num_update_agents
        pol_act_dim = (
            int(sum(update_policy.act_dim))
            if isinstance(update_policy.act_dim, np.ndarray)
            else update_policy.act_dim
        )

        rew_sequence = to_torch(rew_batch[update_policy_id][0]).to(**self.tpdv)
        # use numpy
        env_done_sequence = to_torch(dones_env_batch[update_policy_id]).to(**self.tpdv)
        # mask the Q and target Q sequences with shifted dones (assume the first obs in episode is valid)
        first_step_dones = torch.zeros(
            (1, env_done_sequence.shape[1], env_done_sequence.shape[2])
        ).to(**self.tpdv)
        next_steps_dones = env_done_sequence[: self.episode_length - 1, :, :]
        curr_env_dones = torch.cat((first_step_dones, next_steps_dones), dim=0)

        # last time step does not matter for current observations
        cent_obs_sequence = cent_obs_batch[update_policy_id][:-1]
        # first time step does not matter for next step observations
        cent_nobs_sequence = cent_obs_batch[update_policy_id][1:]

        # group data from agents corresponding to one policy into one larger batch
        pol_agents_obs_seq = np.concatenate(obs_batch[update_policy_id], axis=1)[:-1]
        if avail_act_batch[update_policy_id] is not None:
            pol_agents_avail_act_seq = np.concatenate(
                avail_act_batch[update_policy_id], axis=1
            )[:-1]
        else:
            pol_agents_avail_act_seq = None
        pol_prev_buffer_act_seq = np.concatenate(
            (
                np.zeros((1, total_batch_size, pol_act_dim), dtype=np.float32),
                np.concatenate(act_batch[update_policy_id][:, :-1], axis=1),
            )
        )

        # get centralized sequence information
        (
            cent_act_sequence_buffer,
            act_sequences,
            act_sequence_replace_ind_start,
            cent_nact_sequence,
        ) = self.get_update_info(
            update_policy_id, obs_batch, act_batch, avail_act_batch
        )

        # Critic update:
        predicted_Q_sequences, _ = update_policy.critic(
            cent_obs_sequence,
            cent_act_sequence_buffer,
            update_policy.init_hidden(-1, batch_size),
        )

        # iterate over time to get target Qs since the history at each step should be formed from the buffer sequence
        next_Q_sequence = []
        # detach gradients since no gradients go through target critic
        with torch.no_grad():
            target_critic_rnn_state = update_policy.init_hidden(-1, batch_size)
            for t in range(self.episode_length):
                # update the RNN states based on the buffer sequence
                _, target_critic_rnn_state = update_policy.target_critic(
                    cent_obs_sequence[t],
                    cent_act_sequence_buffer[t],
                    target_critic_rnn_state,
                )
                # get the Q value using the next action taken by the target actor, but don't store the RNN state
                next_Q_ts, _ = update_policy.target_critic(
                    cent_nobs_sequence[t],
                    cent_nact_sequence[t],
                    target_critic_rnn_state,
                )
                next_Q_t = torch.cat(next_Q_ts, dim=-1)
                # take min to prevent overestimation bias
                next_Q_t, _ = torch.min(next_Q_t, dim=-1, keepdim=True)
                next_Q_sequence.append(next_Q_t)

        # stack over time
        next_Q_sequence = torch.stack(next_Q_sequence)

        # mask the next step Qs and form targets; use the env dones as the mask since reward can accumulate even after 1 agent dies
        next_Q_sequence = (1 - env_done_sequence) * next_Q_sequence

        if self.use_popart:
            target_Q_sequence = rew_sequence + self.args.gamma * self.value_normalizer[
                update_policy_id
            ].denormalize(next_Q_sequence)
            nodones_target_Q_sequence = target_Q_sequence[curr_env_dones == 0]
            target_Q_sequence[curr_env_dones == 0] = self.value_normalizer[
                update_policy_id
            ](nodones_target_Q_sequence)
        else:
            target_Q_sequence = rew_sequence + self.args.gamma * next_Q_sequence

        predicted_Q_sequences = [
            Q_seq * (1 - curr_env_dones) for Q_seq in predicted_Q_sequences
        ]
        target_Q_sequence = target_Q_sequence * (1 - curr_env_dones)
        # make sure to detach the targets! Loss is MSE loss, but divide by the number of unmasked elements
        # Mean bellman error for each timestep
        errors = [Q_seq - target_Q_sequence.detach() for Q_seq in predicted_Q_sequences]
        if self.use_per:
            importance_weights = to_torch(importance_weights).to(**self.tpdv)
            # prioritized experience replay
            if self.use_huber_loss:
                per_batch_critic_loss = [
                    huber_loss(error, self.huber_delta).sum(dim=0).flatten()
                    for error in errors
                ]
            else:
                per_batch_critic_loss = [
                    mse_loss(error).sum(dim=0).flatten() for error in errors
                ]
            # weight each loss element by their importance sample weight
            importance_weight_critic_loss = [
                loss * importance_weights for loss in per_batch_critic_loss
            ]
            critic_loss = [
                loss.sum() / (1 - curr_env_dones).sum()
                for loss in importance_weight_critic_loss
            ]
            critic_loss = torch.stack(critic_loss).sum(dim=0)

            # new priorities are a combination of the maximum TD error across sequence and the mean TD error across sequence
            td_errors = [error.abs().cpu().detach().numpy() for error in errors]
            new_priorities = [
                (
                    (1 - self.args.per_nu) * td_error.mean(axis=0)
                    + self.args.per_nu * td_error.max(axis=0)
                ).flatten()
                + self.per_eps
                for td_error in td_errors
            ]
            new_priorities = np.stack(new_priorities).mean(axis=0) + self.per_eps
        else:
            if self.use_huber_loss:
                critic_loss = [
                    huber_loss(error, self.huber_delta).sum()
                    / (1 - curr_env_dones).sum()
                    for error in errors
                ]
            else:
                critic_loss = [
                    mse_loss(error).sum() / (1 - curr_env_dones).sum()
                    for error in errors
                ]
            critic_loss = torch.stack(critic_loss).sum(dim=0)
            new_priorities = None

        update_policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            update_policy.critic.parameters(), self.args.max_grad_norm
        )
        update_policy.critic_optimizer.step()

        train_info["critic_loss"] = critic_loss
        train_info["critic_grad_norm"] = critic_grad_norm

        if update_actor:
            # Actor update
            # freeze Q-networks
            for p in update_policy.critic.parameters():
                p.requires_grad = False

            agent_Q_sequences = []
            # formulate mask to determine how to combine actor output actions with batch output actions
            mask_temp = []
            for p_id in self.policy_ids:
                if isinstance(self.policies[p_id].act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(self.policies[p_id].act_dim))
                else:
                    sum_act_dim = self.policies[p_id].act_dim
                for _ in self.policy_agents[p_id]:
                    mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))

            masks = []
            done_mask = []
            # need to iterate through agents, but only formulate masks at each step
            for i in range(num_update_agents):
                curr_mask_temp = copy.deepcopy(mask_temp)
                curr_mask_temp[act_sequence_replace_ind_start + i] = np.ones(
                    pol_act_dim, dtype=np.float32
                )
                curr_mask_vec = np.concatenate(curr_mask_temp)
                # expand this mask into the proper size
                curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
                masks.append(curr_mask)

                # now collect agent dones
                # ! use numpy
                agent_done_sequence = to_torch(dones_batch[update_policy_id][i])
                agent_first_step_dones = torch.zeros(
                    (1, agent_done_sequence.shape[1], agent_done_sequence.shape[2])
                )
                agent_next_steps_dones = agent_done_sequence[
                    : self.episode_length - 1, :, :
                ]
                curr_agent_dones = torch.cat(
                    (agent_first_step_dones, agent_next_steps_dones), dim=0
                )
                done_mask.append(curr_agent_dones)
            # cat masks and form into torch tensors
            mask = to_torch(np.concatenate(masks)).to(**self.tpdv)
            done_mask = torch.cat(done_mask, dim=1).to(**self.tpdv)

            # get all the actions from actor, with gumbel softmax to differentiate through the samples
            policy_act_seq, _, _ = update_policy.get_actions(
                pol_agents_obs_seq,
                pol_prev_buffer_act_seq,
                update_policy.init_hidden(-1, total_batch_size),
                available_actions=pol_agents_avail_act_seq,
                use_gumbel=True,
            )

            # separate the output into individual agent act sequences
            agent_actor_seqs = policy_act_seq.split(split_size=batch_size, dim=1)
            # convert act sequences to torch, formulate centralized buffer action, and repeat as done above
            act_sequences = list(
                map(lambda arr: to_torch(arr).to(**self.tpdv), act_sequences)
            )

            actor_cent_acts = copy.deepcopy(act_sequences)
            for i in range(num_update_agents):
                actor_cent_acts[act_sequence_replace_ind_start + i] = agent_actor_seqs[
                    i
                ]
            # cat these along final dim to formulate centralized action and stack copies of the batch so all agents can be updated
            actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat(
                (1, num_update_agents, 1)
            )

            batch_cent_acts = torch.cat(act_sequences, dim=-1).repeat(
                (1, num_update_agents, 1)
            )
            # also repeat the cent obs
            stacked_cent_obs_seq = np.tile(cent_obs_sequence, (1, num_update_agents, 1))
            critic_rnn_state = update_policy.init_hidden(-1, total_batch_size)

            # iterate through timesteps and and get Q values to form actor loss
            for t in range(self.episode_length):
                # get Q values at timestep t with the replaced actions
                replaced_cent_act_batch = (
                    mask * actor_cent_acts[t] + (1 - mask) * batch_cent_acts[t]
                )
                # get Q values at timestep but don't store the new RNN state
                Q_t, _ = update_policy.critic(
                    stacked_cent_obs_seq[t], replaced_cent_act_batch, critic_rnn_state
                )
                Q_t = Q_t[0]
                # update the RNN state by stepping the RNN through with buffer sequence
                _, critic_rnn_state = update_policy.critic(
                    stacked_cent_obs_seq[t], batch_cent_acts[t], critic_rnn_state
                )
                agent_Q_sequences.append(Q_t)
            # stack over time
            agent_Q_sequences = torch.stack(agent_Q_sequences)
            # mask at the places where agents were terminated in env
            agent_Q_sequences = agent_Q_sequences * (1 - done_mask)

            actor_loss = (-agent_Q_sequences).sum() / (1 - done_mask).sum()

            update_policy.critic_optimizer.zero_grad()
            update_policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                update_policy.actor.parameters(), self.args.max_grad_norm
            )
            update_policy.actor_optimizer.step()

            # unfreeze the Q networks
            for p in update_policy.critic.parameters():
                p.requires_grad = True

            train_info["actor_grad_norm"] = actor_grad_norm
            train_info["actor_loss"] = actor_loss
        train_info["update_actor"] = update_actor

        self.num_updates[update_policy_id] += 1
        return train_info, new_priorities, idxes

    def cent_train_policy_on_batch(self, update_policy_id, batch):
        """Training function when each agent has its own centralized observation. See train_policy_on_batch."""
        # unpack the batch
        (
            obs_batch,
            cent_obs_batch,
            act_batch,
            rew_batch,
            dones_batch,
            dones_env_batch,
            avail_act_batch,
            importance_weights,
            idxes,
        ) = batch

        train_info = {}
        update_actor = (
            self.num_updates[update_policy_id] % self.actor_update_interval == 0
        )

        # obs_batch: dict mapping policy id to batches where each batch is shape (# agents, ep_len, batch_size, obs_dim)
        update_policy = self.policies[update_policy_id]
        batch_size = obs_batch[update_policy_id].shape[2]
        pol_act_dim = (
            int(sum(update_policy.act_dim))
            if isinstance(update_policy.act_dim, np.ndarray)
            else update_policy.act_dim
        )
        num_update_agents = len(self.policy_agents[update_policy_id])
        total_batch_size = batch_size * num_update_agents

        rew_sequence = to_torch(rew_batch[update_policy_id][0]).to(**self.tpdv)
        env_done_sequence = to_torch(dones_env_batch[update_policy_id]).to(**self.tpdv)
        cent_obs_sequence = cent_obs_batch[update_policy_id][:-1]
        cent_nobs_sequence = cent_obs_batch[update_policy_id][1:]
        dones_sequence = dones_batch[update_policy_id]
        # get centralized sequence information
        (
            cent_act_sequence_buffer,
            act_sequences,
            act_sequence_replace_ind_start,
            cent_nact_sequence,
        ) = self.get_update_info(
            update_policy_id, obs_batch, act_batch, avail_act_batch
        )

        # combine all agents data into one array/tensor by stacking along batch dim; easier to process

        all_agent_cent_obs = np.concatenate(cent_obs_sequence, axis=1)
        all_agent_cent_nobs = np.concatenate(cent_nobs_sequence, axis=1)
        all_agent_dones = np.concatenate(dones_sequence, axis=1)

        pol_agents_obs_seq = np.concatenate(obs_batch[update_policy_id], axis=1)[:-1]
        pol_prev_buffer_act_seq = np.concatenate(
            (
                np.zeros((1, total_batch_size, pol_act_dim), dtype=np.float32),
                np.concatenate(act_batch[update_policy_id][:, :-1], axis=1),
            )
        )
        if avail_act_batch[update_policy_id] is not None:
            pol_agents_avail_act_seq = np.concatenate(
                avail_act_batch[update_policy_id], axis=1
            )[:-1]
        else:
            pol_agents_avail_act_seq = None

        # since this is same for each agent, just repeat when stacking
        all_agent_cent_act_buffer = np.tile(
            cent_act_sequence_buffer, (1, num_update_agents, 1)
        )
        all_agent_cent_nact = np.tile(cent_nact_sequence, (1, num_update_agents, 1))
        all_env_dones = env_done_sequence.repeat(1, num_update_agents, 1)
        all_agent_rewards = rew_sequence.repeat(1, num_update_agents, 1)
        first_step_dones = torch.zeros(
            (1, all_env_dones.shape[1], all_env_dones.shape[2])
        ).to(**self.tpdv)
        next_steps_dones = all_env_dones[:-1, :, :]
        curr_env_dones = torch.cat((first_step_dones, next_steps_dones), dim=0)

        predicted_Q_sequences, _ = update_policy.critic(
            all_agent_cent_obs,
            all_agent_cent_act_buffer,
            update_policy.init_hidden(-1, total_batch_size),
        )
        # iterate over time to get target Qs since history at each step should be formed from the buffer sequence
        next_Q_sequence = []
        # don't track gradients for target computation
        with torch.no_grad():
            target_critic_rnn_state = update_policy.init_hidden(-1, total_batch_size)
            for t in range(self.episode_length):
                # update the RNN states based on the buffer sequence
                _, target_critic_rnn_state = update_policy.target_critic(
                    all_agent_cent_obs[t],
                    all_agent_cent_act_buffer[t],
                    target_critic_rnn_state,
                )
                # get the next value using next action taken by the target actor, but don't store the RNN state
                next_Q_ts, _ = update_policy.target_critic(
                    all_agent_cent_nobs[t],
                    all_agent_cent_nact[t],
                    target_critic_rnn_state,
                )
                next_Q_t = torch.cat(next_Q_ts, dim=-1)
                next_Q_t, _ = torch.min(next_Q_t, dim=-1, keepdim=True)
                next_Q_sequence.append(next_Q_t)
        # stack over time
        next_Q_sequence = torch.stack(next_Q_sequence)
        next_Q_sequence = (1 - all_env_dones) * next_Q_sequence
        if self.use_popart:
            target_Q_sequence = (
                all_agent_rewards
                + self.args.gamma
                * self.value_normalizer[update_policy_id].denormalize(next_Q_sequence)
            )
            nodones_target_Q_sequence = target_Q_sequence[curr_env_dones == 0]
            target_Q_sequence[curr_env_dones == 0] = self.value_normalizer[
                update_policy_id
            ](nodones_target_Q_sequence)
        else:
            target_Q_sequence = all_agent_rewards + self.args.gamma * next_Q_sequence

        predicted_Q_sequences = [
            Q_seq * (1 - curr_env_dones) for Q_seq in predicted_Q_sequences
        ]
        target_Q_sequence = target_Q_sequence * (1 - curr_env_dones)

        if self.use_value_active_masks:
            curr_agent_dones = to_torch(all_agent_dones).to(**self.tpdv)
            predicted_Q_sequences = [
                Q_seq * (1 - curr_agent_dones) for Q_seq in predicted_Q_sequences
            ]
            target_Q_sequence = target_Q_sequence * (1 - curr_agent_dones)

        # make sure to detach the targets! Loss is MSE loss, but divide by the number of unmasked elements
        # Mean bellman error for each timestep
        errors = [Q_seq - target_Q_sequence.detach() for Q_seq in predicted_Q_sequences]
        if self.use_per:
            agent_importance_weights = np.tile(importance_weights, num_update_agents)
            agent_importance_weights = to_torch(agent_importance_weights).to(
                **self.tpdv
            )
            if self.use_huber_loss:
                per_batch_critic_loss = [
                    huber_loss(error, self.huber_delta).sum(dim=0).flatten()
                    for error in errors
                ]
            else:
                per_batch_critic_loss = [
                    mse_loss(error).sum(dim=0).flatten() for error in errors
                ]
            agent_importance_weight_critic_loss = [
                loss * agent_importance_weights for loss in per_batch_critic_loss
            ]

            if self.use_value_active_masks:
                critic_loss = [
                    loss.sum() / ((1 - curr_env_dones) * (1 - curr_agent_dones)).sum()
                    for loss in agent_importance_weight_critic_loss
                ]
            else:
                critic_loss = [
                    loss.sum() / (1 - curr_env_dones).sum()
                    for loss in agent_importance_weight_critic_loss
                ]

            td_errors = [error.abs().cpu().detach().numpy() for error in errors]
            agent_new_priorities = [
                (
                    (1 - self.args.per_nu) * td_error.mean(axis=0)
                    + self.args.per_nu * td_error.max(axis=0)
                ).flatten()
                for td_error in td_errors
            ]
            new_priorities = np.stack(agent_new_priorities).mean(axis=0) + self.per_eps
        else:
            if self.use_huber_loss:
                if self.use_value_active_masks:
                    critic_loss = [
                        huber_loss(error, self.huber_delta).sum()
                        / ((1 - curr_env_dones) * (1 - curr_agent_dones)).sum()
                        for error in errors
                    ]
                else:
                    critic_loss = [
                        huber_loss(error, self.huber_delta).sum()
                        / (1 - curr_env_dones).sum()
                        for error in errors
                    ]
            else:
                if self.use_value_active_masks:
                    critic_loss = [
                        mse_loss(error).sum()
                        / ((1 - curr_env_dones) * (1 - curr_agent_dones)).sum()
                        for error in errors
                    ]
                else:
                    critic_loss = [
                        mse_loss(error).sum() / (1 - curr_env_dones).sum()
                        for error in errors
                    ]
            critic_loss = torch.stack(critic_loss).sum(dim=0)
            new_priorities = None

        update_policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            update_policy.critic.parameters(), self.args.max_grad_norm
        )
        update_policy.critic_optimizer.step()
        train_info["critic_loss"] = critic_loss
        train_info["critic_grad_norm"] = critic_grad_norm

        if update_actor:
            # actor update: can form losses for each agent that the update policy controls
            # freeze Q-networks
            for p in update_policy.critic.parameters():
                p.requires_grad = False

            agent_Q_sequences = []
            # formulate mask to determine how to combine actor output actions with batch output actions
            mask_temp = []
            for p_id in self.policy_ids:
                if isinstance(self.policies[p_id].act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(self.policies[p_id].act_dim))
                else:
                    sum_act_dim = self.policies[p_id].act_dim
                for _ in self.policy_agents[p_id]:
                    mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))

            masks = []
            done_mask = []
            # need to iterate through agents, but only formulate masks at each step
            for i in range(num_update_agents):
                curr_mask_temp = copy.deepcopy(mask_temp)
                # set the mask to 1 at locations where the action should come from the actor output
                curr_mask_temp[act_sequence_replace_ind_start + i] = np.ones(
                    pol_act_dim, dtype=np.float32
                )
                curr_mask_vec = np.concatenate(curr_mask_temp)
                # expand this mask into the proper size
                curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
                masks.append(curr_mask)

                # now collect agent dones
                if self.use_value_active_masks:
                    agent_done_sequence = to_torch(dones_batch[update_policy_id][i])
                    done_mask.append(agent_done_sequence)
                else:
                    agent_done_sequence = to_torch(dones_batch[update_policy_id][i])
                    agent_first_step_dones = torch.zeros(
                        (1, agent_done_sequence.shape[1], agent_done_sequence.shape[2])
                    )
                    agent_next_steps_dones = agent_done_sequence[
                        : self.episode_length - 1, :, :
                    ]
                    curr_agent_dones = torch.cat(
                        (agent_first_step_dones, agent_next_steps_dones), dim=0
                    )
                    done_mask.append(curr_agent_dones)

            # cat masks and form into torch tensors
            mask = to_torch(np.concatenate(masks)).to(**self.tpdv)
            done_mask = torch.cat(done_mask, dim=1).to(**self.tpdv)

            # get all the actions from actor, with gumbel softmax to differentiate through the samples
            policy_act_seq, _, _ = update_policy.get_actions(
                pol_agents_obs_seq,
                pol_prev_buffer_act_seq,
                update_policy.init_hidden(-1, total_batch_size),
                available_actions=pol_agents_avail_act_seq,
                use_gumbel=True,
            )
            # separate the output into individual agent act sequences
            agent_actor_seqs = policy_act_seq.split(split_size=batch_size, dim=1)
            # convert act sequences to torch, formulate centralized buffer action, and repeat as done above
            act_sequences = list(
                map(lambda arr: to_torch(arr).to(**self.tpdv), act_sequences)
            )

            actor_cent_acts = copy.deepcopy(act_sequences)
            for i in range(num_update_agents):
                actor_cent_acts[act_sequence_replace_ind_start + i] = agent_actor_seqs[
                    i
                ]
            # cat these along final dim to formulate centralized action and stack copies of the batch so all agents can be updated
            actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat(
                (1, num_update_agents, 1)
            )

            batch_cent_acts = torch.cat(act_sequences, dim=-1).repeat(
                (1, num_update_agents, 1)
            )
            # also repeat the cent obs
            critic_rnn_state = update_policy.init_hidden(-1, total_batch_size)

            # iterate through timesteps and and get Q values to form actor loss
            for t in range(self.episode_length):
                # get Q values at timestep t with the replaced actions
                replaced_cent_act_batch = (
                    mask * actor_cent_acts[t] + (1 - mask) * batch_cent_acts[t]
                )
                # get Q values at timestep but don't store the new RNN state
                Q_t, _ = update_policy.critic(
                    all_agent_cent_obs[t], replaced_cent_act_batch, critic_rnn_state
                )
                Q_t = Q_t[0]
                # update the RNN state by stepping the RNN through with buffer sequence
                _, critic_rnn_state = update_policy.critic(
                    all_agent_cent_obs[t], batch_cent_acts[t], critic_rnn_state
                )
                agent_Q_sequences.append(Q_t)
            # stack over time
            agent_Q_sequences = torch.stack(agent_Q_sequences)
            # mask at the places where agents were terminated in env
            agent_Q_sequences = agent_Q_sequences * (1 - done_mask)

            actor_loss = (-agent_Q_sequences).sum() / (1 - done_mask).sum()

            update_policy.critic_optimizer.zero_grad()
            update_policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                update_policy.actor.parameters(), self.args.max_grad_norm
            )
            update_policy.actor_optimizer.step()

            # unfreeze the Q networks
            for p in update_policy.critic.parameters():
                p.requires_grad = True

            train_info["actor_grad_norm"] = actor_grad_norm
            train_info["actor_loss"] = actor_loss

        self.num_updates[update_policy_id] += 1
        return train_info, new_priorities, idxes

    def prep_training(self):
        """See parent class."""
        for policy in self.policies.values():
            policy.actor.train()
            policy.critic.train()
            policy.target_actor.train()
            policy.target_critic.train()

    def prep_rollout(self):
        """See parent class."""
        for policy in self.policies.values():
            policy.actor.eval()
            policy.critic.eval()
            policy.target_actor.eval()
            policy.target_critic.eval()
