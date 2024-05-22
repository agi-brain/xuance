import argparse
import torch
import copy
from baselines.offpolicy.algorithms.mqmix.algorithm.mq_mixer import M_QMixer
from baselines.offpolicy.algorithms.mvdn.algorithm.mvdn_mixer import M_VDNMixer
from baselines.offpolicy.utils.util import soft_update, huber_loss, mse_loss, to_torch
import numpy as np
from baselines.offpolicy.utils.popart import PopArt


class M_QMix:
    def __init__(
        self,
        args: argparse.Namespace,
        num_agents: int,
        policies: dict,
        policy_mapping_fn,
        device: torch.device = torch.device("cuda:0"),
        vdn: bool = False,
    ):
        """
        Trainer class for QMix with MLP policies. See parent class for more information.
        :param vdn: (bool) whether the algorithm in use is VDN.
        """
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = self.args.lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay

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

        multidiscrete_list = None
        if any(
            [
                isinstance(policy.act_dim, np.ndarray)
                for policy in self.policies.values()
            ]
        ):
            # multidiscrete
            multidiscrete_list = [
                len(self.policies[p_id].act_dim) * len(self.policy_agents[p_id])
                for p_id in self.policy_ids
            ]

        # mixer network
        if vdn:
            self.mixer = M_VDNMixer(
                args,
                self.num_agents,
                self.policies["policy_0"].central_obs_dim,
                self.device,
                multidiscrete_list=multidiscrete_list,
            )
        else:
            self.mixer = M_QMixer(
                args,
                self.num_agents,
                self.policies["policy_0"].central_obs_dim,
                self.device,
                multidiscrete_list=multidiscrete_list,
            )

        # target policies/networks
        self.target_policies = {
            p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids
        }
        self.target_mixer = copy.deepcopy(self.mixer)

        # collect all trainable parameters: each policy parameters, and the mixer parameters
        self.parameters = []
        for policy in self.policies.values():
            self.parameters += policy.parameters()
        self.parameters += self.mixer.parameters()

        self.optimizer = torch.optim.Adam(
            params=self.parameters, lr=self.lr, eps=self.opti_eps
        )

        if args.use_double_q:
            print("double Q learning will be used")

    def train_policy_on_batch(self, batch, use_same_share_obs):
        """See parent class."""
        # unpack the batch
        (
            obs_batch,
            cent_obs_batch,
            act_batch,
            rew_batch,
            nobs_batch,
            cent_nobs_batch,
            dones_batch,
            dones_env_batch,
            valid_transition_batch,
            avail_act_batch,
            navail_act_batch,
            importance_weights,
            idxes,
        ) = batch

        if use_same_share_obs:
            cent_obs_batch = to_torch(cent_obs_batch[self.policy_ids[0]])
            cent_nobs_batch = to_torch(cent_nobs_batch[self.policy_ids[0]])
        else:
            choose_agent_id = 0
            cent_obs_batch = to_torch(
                cent_obs_batch[self.policy_ids[0]][choose_agent_id]
            )
            cent_nobs_batch = to_torch(
                cent_nobs_batch[self.policy_ids[0]][choose_agent_id]
            )

        dones_env_batch = to_torch(dones_env_batch[self.policy_ids[0]]).to(**self.tpdv)

        # individual agent q values: each element is of shape (batch_size, 1)
        agent_qs = []
        agent_next_qs = []

        for p_id in self.policy_ids:
            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            # get data related to the policy id
            rewards = to_torch(rew_batch[p_id][0]).to(**self.tpdv)
            curr_obs_batch = to_torch(obs_batch[p_id])
            curr_act_batch = to_torch(act_batch[p_id]).to(**self.tpdv)
            curr_nobs_batch = to_torch(nobs_batch[p_id])

            # stacked_obs_batch size : [agent_num*batch_size, obs_shape]
            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2)
            stacked_obs_batch = torch.cat(list(curr_obs_batch), dim=-2)
            stacked_nobs_batch = torch.cat(list(curr_nobs_batch), dim=-2)

            if navail_act_batch[p_id] is not None:
                curr_navail_act_batch = to_torch(navail_act_batch[p_id])
                stacked_navail_act_batch = torch.cat(
                    list(curr_navail_act_batch), dim=-2
                )
            else:
                stacked_navail_act_batch = None

            # curr_obs_batch size : agent_num*batch_size*obs_shape
            batch_size = curr_obs_batch.shape[1]

            pol_all_q_out = policy.get_q_values(stacked_obs_batch)

            if isinstance(pol_all_q_out, list):
                # multidiscrete case
                ind = 0
                Q_per_part = []
                for i in range(len(policy.act_dim)):
                    curr_stacked_act_batch = stacked_act_batch[
                        :, ind : ind + policy.act_dim[i]
                    ]
                    curr_stacked_act_batch_ind = curr_stacked_act_batch.max(dim=-1)[1]
                    curr_all_q_out = pol_all_q_out[i]
                    curr_pol_q_out = torch.gather(
                        curr_all_q_out, 1, curr_stacked_act_batch_ind.unsqueeze(dim=-1)
                    )
                    Q_per_part.append(curr_pol_q_out)
                    ind += policy.act_dim[i]
                Q_combined_parts = torch.cat(Q_per_part, dim=-1)
                pol_agents_q_outs = Q_combined_parts.split(
                    split_size=batch_size, dim=-2
                )
            else:
                # get the q values associated with the action taken acording ot the batch
                stacked_act_batch_ind = stacked_act_batch.max(dim=-1)[1]
                # pol_q_outs : batch_size * 1
                pol_q_outs = torch.gather(
                    pol_all_q_out, 1, stacked_act_batch_ind.unsqueeze(dim=-1)
                )
                # separate into agent q sequences for each agent,
                # then cat along the final dimension to prepare for mixer input
                pol_agents_q_outs = pol_q_outs.split(split_size=batch_size, dim=-2)

            agent_qs.append(torch.cat(pol_agents_q_outs, dim=-1))

            with torch.no_grad():
                if self.args.use_double_q:
                    # actions come from live q; get the q values for the final nobs
                    pol_next_qs = policy.get_q_values(stacked_nobs_batch)

                    if type(pol_next_qs) == list:
                        # multidiscrete case
                        assert (
                            stacked_navail_act_batch is None
                        ), "Available actions not supported for multidiscrete"
                        pol_nacts = []
                        for i in range(len(pol_next_qs)):
                            curr_next_q = pol_next_qs[i]
                            pol_curr_nacts = curr_next_q.max(dim=-1)[1]
                            pol_nacts.append(pol_curr_nacts)
                        # pol_nacts = np.concatenate(pol_nacts, axis=-1)
                        targ_pol_next_qs = target_policy.get_q_values(
                            stacked_nobs_batch, action_batch=pol_nacts
                        )
                    else:
                        # mask out the unavailable actions
                        if stacked_navail_act_batch is not None:
                            pol_next_qs[stacked_navail_act_batch == 0.0] = -1e10
                        # greedily choose actions which maximize the
                        # q values and convert these actions to onehot
                        pol_nacts = pol_next_qs.max(dim=-1)[1]
                        # q values given by target but evaluated at actions taken by live
                        targ_pol_next_qs = target_policy.get_q_values(
                            stacked_nobs_batch, action_batch=pol_nacts
                        )
                else:
                    # just choose actions from target policy
                    _, targ_pol_next_qs = target_policy.get_actions(
                        stacked_nobs_batch,
                        available_actions=stacked_navail_act_batch,
                        t_env=None,
                        explore=False,
                    )
                    targ_pol_next_qs = targ_pol_next_qs.max(dim=-1)[0]
                    targ_pol_next_qs = targ_pol_next_qs.unsqueeze(-1)
                # separate the next qs into sequences for each agent
                pol_agents_nq_sequence = targ_pol_next_qs.split(
                    split_size=batch_size, dim=-2
                )
            # cat target qs along the final dim
            agent_next_qs.append(torch.cat(pol_agents_nq_sequence, dim=-1))
        # combine the agent q value sequences to feed into mixer networks
        agent_qs = torch.cat(agent_qs, dim=-1)
        agent_next_qs = torch.cat(agent_next_qs, dim=-1)

        curr_Q_tot = self.mixer(agent_qs, cent_obs_batch).squeeze(-1)
        next_step_Q_tot = self.target_mixer(agent_next_qs, cent_nobs_batch).squeeze(-1)

        # all agents must share reward, so get the reward sequence for an agent
        # form bootstrapped targets
        if self.use_popart:
            Q_tot_targets = rewards + (
                1 - dones_env_batch
            ) * self.args.gamma * self.value_normalizer[p_id].denormalize(
                next_step_Q_tot
            )
            Q_tot_targets = self.value_normalizer[p_id](Q_tot_targets)
        else:
            Q_tot_targets = (
                rewards + (1 - dones_env_batch) * self.args.gamma * next_step_Q_tot
            )

        # loss is MSE Bellman Error
        error = curr_Q_tot - Q_tot_targets.detach()
        if self.use_per:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).flatten()
            else:
                loss = mse_loss(error).flatten()
            loss = (loss * to_torch(importance_weights).to(**self.tpdv)).mean()
            # new priorities are a combination of the maximum TD error
            # across sequence and the mean TD error across sequence
            new_priorities = error.abs().cpu().detach().numpy().flatten() + self.per_eps
        else:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).mean()
            else:
                loss = mse_loss(error).mean()
            new_priorities = None

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.args.max_grad_norm
        )
        self.optimizer.step()

        train_info = {}
        train_info["loss"] = loss
        train_info["grad_norm"] = grad_norm
        train_info["Q_tot"] = curr_Q_tot.mean()

        return train_info, new_priorities, idxes

    def hard_target_updates(self):
        """Hard update the target networks."""
        # print("hard update targets")
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(self.policies[policy_id])
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_target_updates(self):
        """Soft update the target networks."""
        for policy_id in self.policy_ids:
            soft_update(
                self.target_policies[policy_id], self.policies[policy_id], self.tau
            )
        if self.mixer is not None:
            soft_update(self.target_mixer, self.mixer, self.tau)

    def prep_training(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.train()
            self.target_policies[p_id].q_network.train()
        self.mixer.train()
        self.target_mixer.train()

    def prep_rollout(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.eval()
            self.target_policies[p_id].q_network.eval()
        self.mixer.eval()
        self.target_mixer.eval()
