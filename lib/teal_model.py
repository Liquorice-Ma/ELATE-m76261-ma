import pickle
import time
import json
import sys
import os
from tqdm import tqdm
from networkx.readwrite import json_graph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .teal_actor import TealActor
from .teal_env import TealEnv
from .utils import print_


class Teal():
    def __init__(self, teal_env, teal_actor, lr, early_stop):
        """Initialize Teal model.

        Args:
            teal_env: teal environment
            teal_actor: teal actor
            lr: learning rate
            early_stop: whether to early stop
        """

        self.env = teal_env
        self.actor = teal_actor
        self.leo_mode = teal_env.leo_mode

        # init optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # early stop when val result no longer changes
        self.early_stop = early_stop
        if self.early_stop:
            self.val_reward = []

    def train(self, num_epoch, batch_size, num_sample):
        """Train Teal model.

        Args:
            num_epoch: number of training epoch
            batch_size: batch size
            num_sample: number of samples in COMA reward
        """

        for epoch in range(num_epoch):

            self.env.reset('train')

            ids = range(self.env.idx_start, self.env.idx_stop)
            loop_obj = tqdm(
                [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)],
                desc=f"Training epoch {epoch}/{num_epoch}: ")

            for idx in loop_obj:
                loss = 0
                for _ in idx:
                    torch.cuda.empty_cache()

                    obs = self.env.get_obs()

                    if self.leo_mode:
                        loss += self._train_step_leo(obs, num_sample)
                    else:
                        loss += self._train_step_teal(obs, num_sample)

                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

            # early stop
            if self.early_stop:
                self.val()
                if len(self.val_reward) > 20 and abs(
                        sum(self.val_reward[-20:-10])/10
                        - sum(self.val_reward[-10:])/10) < 0.0001:
                    break
        self.actor.save_model()

    def _train_step_teal(self, obs, num_sample):
        """Original TEAL training step."""
        raw_action, log_probability = self.actor.evaluate(obs)
        reward, info = self.env.step(raw_action, num_sample=num_sample)
        return -(log_probability * reward).mean()

    def _train_step_leo(self, obs, num_sample):
        """ELATE LEO training step with counterfactual MARL (Eq. 9-10).

        Policy gradient: ∇_θ J(θ) = E_π[Σ_k ∇_θ log π_θ(a_k|s_k) * A_k]
        where A_k is the counterfactual advantage from Eq. 9.
        """
        if self.env.num_elephant == 0:
            self.env._next_obs()
            return torch.tensor(0.0, requires_grad=True).to(self.env.device)

        # Get action from policy (with gradient)
        feature = obs.reshape(-1, 1)
        mean, std = self.actor.forward(feature)

        from torch.distributions.normal import Normal
        distribution = Normal(mean, std)
        sample = distribution.rsample()
        log_probability = distribution.log_prob(sample).sum(axis=-1)

        # Compute counterfactual advantage (Eq. 9)
        # R(s, a) - (1/N_s) * Σ R(s, (a_{-k}, a'_m))
        with torch.no_grad():
            # Current joint action reward
            action = self.env._transform_leo_action(sample.detach())
            current_mlu = self.env._compute_mlu(action)
            current_reward = -current_mlu

            # Counterfactual baseline per agent
            num_agents = self.env.num_elephant
            advantage = torch.zeros(num_agents).to(self.env.device)

            for k in range(num_agents):
                cf_total = 0.0
                for _ in range(num_sample):
                    # Resample agent k's action from current policy
                    cf_sample = sample.clone()
                    cf_action_k = distribution.rsample()[k]
                    cf_sample[k] = cf_action_k

                    cf_action = self.env._transform_leo_action(cf_sample)
                    cf_mlu = self.env._compute_mlu(cf_action)
                    cf_total += -cf_mlu

                baseline = cf_total / max(num_sample, 1)
                advantage[k] = current_reward - baseline

        # Policy gradient loss (Eq. 10)
        loss = -(log_probability * advantage).sum()

        # Step environment
        self.env._next_obs()
        return loss

    def val(self):
        """Validating Teal model."""

        self.actor.eval()
        self.env.reset('val')

        rewards = 0
        for idx in range(self.env.idx_start, self.env.idx_stop):

            problem_dict = self.env.render()
            obs = self.env.get_obs()
            raw_action = self.actor.act(obs)
            reward, info = self.env.step(raw_action)

            if self.leo_mode:
                rewards += reward.item() if torch.is_tensor(reward) else reward
            else:
                rewards += reward.item()/problem_dict['total_demand']\
                    if self.env.obj == 'total_flow' else reward.item()

        self.val_reward.append(
            rewards/(self.env.idx_stop - self.env.idx_start))

    def test(self, num_admm_step, output_header, output_csv, output_dir):
        """Test Teal model.

        Args:
            num_admm_step: number of ADMM steps
            output_header: header of the output csv
            output_csv: name of the output csv
            output_dir: directory to save output solution
        """

        self.actor.eval()
        self.env.reset('test')

        with open(output_csv, "a") as results:
            print_(",".join(output_header), file=results)

            runtime_list, obj_list = [], []
            loop_obj = tqdm(
                range(self.env.idx_start, self.env.idx_stop),
                desc="Testing: ")

            for idx in loop_obj:

                problem_dict = self.env.render()
                obs = self.env.get_obs()

                start_time = time.time()
                raw_action = self.actor.act(obs)
                runtime = time.time() - start_time

                reward, info = self.env.step(
                    raw_action, num_admm_step=num_admm_step)

                runtime += info.get('runtime', 0)
                runtime_list.append(runtime)

                if self.leo_mode:
                    obj_val = reward.item() if torch.is_tensor(reward) \
                        else reward
                    obj_list.append(-obj_val)  # MLU (positive)
                else:
                    obj_list.append(
                        reward.item()/problem_dict['total_demand']
                        if self.env.obj == 'total_flow' else reward.item())

                loop_obj.set_postfix({
                    'runtime': '%.4f' % (sum(runtime_list)/len(runtime_list)),
                    'obj': '%.4f' % (sum(obj_list)/len(obj_list)),
                    })

                # save solution matrix
                sol_mat = info.get('sol_mat')
                if sol_mat is not None:
                    torch.save(sol_mat, os.path.join(
                        output_dir,
                        "{}-{}-{}-teal_objective-{}_{}-paths_"
                        "edge-disjoint-{}_dist-metric-{}_sol-mat.pt".format(
                            problem_dict['problem_name'],
                            problem_dict['traffic_model'],
                            problem_dict['traffic_seed'],
                            problem_dict['obj'],
                            problem_dict['num_path'],
                            problem_dict['edge_disjoint'],
                            problem_dict['dist_metric'])))

                PLACEHOLDER = ",".join("{}" for _ in output_header)
                result_line = PLACEHOLDER.format(
                    problem_dict['problem_name'],
                    problem_dict['num_node'],
                    problem_dict['num_edge'],
                    problem_dict['traffic_seed'],
                    problem_dict['scale_factor'],
                    problem_dict['traffic_model'],
                    problem_dict['total_demand'],
                    "ELATE" if self.leo_mode else "Teal",
                    problem_dict['num_path'],
                    problem_dict['edge_disjoint'],
                    problem_dict['dist_metric'],
                    problem_dict['obj'],
                    reward,
                    runtime)
                print_(result_line, file=results)
