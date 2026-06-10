import pickle
import json
import os
import math
import time
import random
from itertools import product

from networkx.readwrite import json_graph
import networkx as nx

import torch
import torch_scatter
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from .config import TOPOLOGIES_DIR
from .ADMM import ADMM
from .path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
from .flow_identification import identify_elephant_mice


class TealEnv(object):

    def __init__(
            self, obj, topo, problems,
            num_path, edge_disjoint, dist_metric, rho,
            train_size, val_size, test_size, num_failure, device,
            leo_mode=False, lam=5.0,
            raw_action_min=-10.0, raw_action_max=10.0):
        """Initialize Teal environment.

        Args:
            obj: objective
            topo: topology JSON filename
            problems: problem list
            num_path: number of paths per demand
            edge_disjoint: whether edge-disjoint paths
            dist_metric: distance metric for shortest paths
            rho: hyperparameter for the augmented Lagrangian
            train_size: train start index, stop index
            val_size: val start index, stop index
            test_size: test start index, stop index
            num_failure: number of link failures in testing
            device: device id
            leo_mode: whether to use ELATE mode
            lam: lambda threshold for elephant/mice separation
            raw_action_min: min value when clamp raw action
            raw_action_max: max value when clamp raw action
        """

        self.obj = obj
        self.topo = topo
        self.problems = problems
        self.num_path = num_path
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric
        self.leo_mode = leo_mode

        self.train_start, self.train_stop = train_size
        self.val_start, self.val_stop = val_size
        self.test_start, self.test_stop = test_size
        self.num_failure = num_failure
        self.device = device

        self.raw_action_min = raw_action_min
        self.raw_action_max = raw_action_max

        if self.leo_mode:
            self._init_leo(lam, rho)
        else:
            self._init_teal(rho)

        self.reset('train')

    def _init_teal(self, rho):
        """Initialize in original TEAL mode."""
        self.G = self._read_graph_json(self.topo)
        self.capacity = torch.FloatTensor(
            [float(c_e) for u, v, c_e in self.G.edges.data('capacity')])
        self.num_edge_node = len(self.G.edges)
        self.num_path_node = self.num_path * self.G.number_of_nodes()\
            * (self.G.number_of_nodes()-1)
        self.edge_index, self.edge_index_values, self.p2e = \
            self.get_topo_matrix(self.topo, self.num_path,
                                self.edge_disjoint, self.dist_metric)

        self.ADMM = ADMM(
            self.p2e, self.num_path, self.num_path_node,
            self.num_edge_node, rho, self.device)

    def _init_leo(self, lam, rho):
        """Initialize in ELATE mode.

        Reads topology from JSON (same as TEAL), then sets up
        elephant/mice flow separation infrastructure.
        """
        self.lam = lam

        # Read topology from JSON
        self.G = self._read_graph_json(self.topo)
        self.num_nodes = self.G.number_of_nodes()
        self.capacity = torch.FloatTensor(
            [float(c_e) for u, v, c_e in self.G.edges.data('capacity')])
        self.num_edge_node = len(self.G.edges)

        # edge index lookup
        self.edge2idx = {edge: idx for idx, edge in enumerate(self.G.edges)}

        # pre-compute shortest paths for all pairs (used for mice routing)
        print("Pre-computing shortest paths for mice flow routing...")
        self.shortest_paths = dict(nx.all_pairs_shortest_path(self.G))
        print(f"Topology: {self.num_nodes} nodes, {self.num_edge_node} edges")

        # pre-compute k-shortest paths for elephant flows
        print("Pre-computing candidate paths...")
        self.all_path_dict = self._compute_all_paths()
        print(f"Path dict size: {len(self.all_path_dict)}")

        # dynamic state (set per observation)
        self.elephant_indices = None
        self.mice_edge_flow = None
        self.residual_capacity = None
        self.num_path_node = None
        self.num_elephant = None
        self.edge_index = None
        self.edge_index_values = None
        self.p2e = None

    def _compute_all_paths(self):
        """Pre-compute k-shortest paths for all node pairs."""
        path_dict = {}
        G = graph_copy_with_edge_weights(self.G, self.dist_metric)
        for s_k in G.nodes:
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(
                    G, s_k, t_k, self.num_path, self.edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                # pad to num_path if not enough
                if len(paths_no_cycles) < self.num_path:
                    paths_no_cycles = [paths_no_cycles[0]] * \
                        (self.num_path - len(paths_no_cycles)) + paths_no_cycles
                elif len(paths_no_cycles) > self.num_path:
                    paths_no_cycles = paths_no_cycles[:self.num_path]
                path_dict[(s_k, t_k)] = paths_no_cycles
        return path_dict

    def reset(self, mode='test'):
        """Reset the initial conditions in the beginning."""

        if mode == 'train':
            self.idx_start, self.idx_stop = self.train_start, self.train_stop
        elif mode == 'test':
            self.idx_start, self.idx_stop = self.test_start, self.test_stop
        else:
            self.idx_start, self.idx_stop = self.val_start, self.val_stop
        self.idx = self.idx_start
        self.obs = self._read_obs()

    def get_obs(self):
        """Return observation."""
        return self.obs

    def _read_obs(self):
        """Return observation from files."""
        topo, topo_fname, tm_fname = self.problems[self.idx]
        with open(tm_fname, 'rb') as f:
            tm = pickle.load(f)

        if self.leo_mode:
            return self._build_leo_obs(tm)
        else:
            return self._build_teal_obs(tm)

    def _build_teal_obs(self, tm):
        """Build observation for original TEAL mode."""
        tm_tensor = torch.FloatTensor(
            [[ele]*self.num_path for i, ele in enumerate(tm.flatten())
                if i % len(tm) != i//len(tm)]).flatten()
        obs = torch.concat([self.capacity, tm_tensor]).to(self.device)
        if self.num_failure > 0 and self.idx_start == self.test_start:
            idx_failure = torch.tensor(
                random.sample(range(self.num_edge_node),
                self.num_failure)).to(self.device)
            obs[idx_failure] = 0
        return obs

    def _build_leo_obs(self, tm):
        """Build observation for ELATE mode.

        1. Separate elephant/mice flows (Algorithm 1)
        2. Route mice on shortest paths -> residual capacity
        3. Build bipartite graph for elephant flows only
        4. Return [residual_capacity, elephant_demands]
        """
        tm_tensor = torch.FloatTensor(tm)

        # Step 1: Identify elephant and mice flows
        F_e, F_m, elephant_mask = identify_elephant_mice(tm_tensor, self.lam)

        # Step 2: Get elephant flow pairs
        elephant_pairs = []
        elephant_demands = []
        n = tm_tensor.shape[0]
        for i in range(n):
            for j in range(n):
                if elephant_mask[i, j]:
                    elephant_pairs.append((i, j))
                    elephant_demands.append(F_e[i, j].item())

        self.num_elephant = len(elephant_pairs)
        self.elephant_pairs = elephant_pairs
        self.elephant_demands_raw = torch.FloatTensor(elephant_demands)

        # Step 3: Route mice flows on shortest paths, compute residual capacity
        mice_edge_flow = torch.zeros(self.num_edge_node)
        for i in range(n):
            for j in range(n):
                if i == j or F_m[i, j] == 0:
                    continue
                if i in self.shortest_paths and j in self.shortest_paths[i]:
                    path = self.shortest_paths[i][j]
                    flow = F_m[i, j].item()
                    for u, v in zip(path[:-1], path[1:]):
                        if (u, v) in self.edge2idx:
                            mice_edge_flow[self.edge2idx[(u, v)]] += flow

        self.mice_edge_flow = mice_edge_flow.to(self.device)
        self.residual_capacity = (self.capacity - mice_edge_flow).clamp(min=0)
        self.residual_capacity = self.residual_capacity.to(self.device)

        # Step 4: Build bipartite graph for elephant flows
        if self.num_elephant > 0:
            self._build_elephant_bipartite(elephant_pairs)

            elephant_tm = torch.FloatTensor(
                [[d]*self.num_path for d in elephant_demands]).flatten()
            self.num_path_node = self.num_elephant * self.num_path
        else:
            self.num_path_node = 0
            elephant_tm = torch.FloatTensor([])
            self.p2e = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            self.edge_index = torch.zeros(
                (2, 0), dtype=torch.long).to(self.device)
            self.edge_index_values = torch.FloatTensor([]).to(self.device)

        obs = torch.concat([self.residual_capacity, elephant_tm]).to(self.device)
        return obs

    def _build_elephant_bipartite(self, elephant_pairs):
        """Build bipartite graph for elephant flows with augmented bi-adjacency.

        Implements Eq. 7: Ā_t = [[I_{|E|}, A_t], [A_t^T, I_{|P|}]]
        """
        src, dst = [], []
        path_i = 0
        edge_num = self.num_edge_node

        for (s, d) in elephant_pairs:
            paths = self.all_path_dict.get((s, d), [])
            for path in paths:
                for u, v in zip(path[:-1], path[1:]):
                    if (u, v) in self.edge2idx:
                        src.append(edge_num + path_i)
                        dst.append(self.edge2idx[(u, v)])
                path_i += 1

        if len(src) == 0:
            self.p2e = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            self.edge_index = torch.zeros(
                (2, 0), dtype=torch.long).to(self.device)
            self.edge_index_values = torch.FloatTensor([]).to(self.device)
            return

        total_nodes = edge_num + path_i

        # Bipartite edges (path <-> edge) + identity (self-loops)
        all_src = src + dst + list(range(total_nodes))
        all_dst = dst + src + list(range(total_nodes))

        # D^(-0.5) * A * D^(-0.5) normalization
        node2degree = {}
        for u, v in zip(all_src, all_dst):
            node2degree[u] = node2degree.get(u, 0) + 1
            node2degree[v] = node2degree.get(v, 0) + 1

        edge_index_values = torch.tensor(
            [1.0 / math.sqrt(node2degree.get(u, 1) * node2degree.get(v, 1))
             for u, v in zip(all_src, all_dst)]).to(self.device)
        edge_index = torch.tensor(
            [all_src, all_dst], dtype=torch.long).to(self.device)

        # p2e: path-to-edge mapping (for flow computation)
        p2e = torch.tensor([src, dst], dtype=torch.long).to(self.device)
        p2e[0] -= edge_num

        self.edge_index = edge_index
        self.edge_index_values = edge_index_values
        self.p2e = p2e

    def _next_obs(self):
        """Return next observation."""
        self.idx += 1
        if self.idx == self.idx_stop:
            self.idx = self.idx_start
        self.obs = self._read_obs()
        return self.obs

    def render(self):
        """Return a dictionary for the details of the current problem."""

        topo, topo_fname, tm_fname = self.problems[self.idx]

        if self.leo_mode:
            total_demand = self.elephant_demands_raw.sum().item() if \
                self.num_elephant > 0 else 0
            problem_dict = {
                'problem_name': topo,
                'obj': 'min_max_link_util',
                'tm_fname': tm_fname.split('/')[-1],
                'num_node': self.num_nodes,
                'num_edge': self.num_edge_node,
                'num_path': self.num_path,
                'edge_disjoint': self.edge_disjoint,
                'dist_metric': self.dist_metric,
                'traffic_model': tm_fname.split('/')[-2],
                'traffic_seed': int(tm_fname.split('_')[-3]),
                'scale_factor': float(tm_fname.split('_')[-2]),
                'total_demand': total_demand,
                'num_elephant': self.num_elephant,
            }
        else:
            problem_dict = {
                'problem_name': topo,
                'obj': self.obj,
                'tm_fname': tm_fname.split('/')[-1],
                'num_node': self.G.number_of_nodes(),
                'num_edge': self.G.number_of_edges(),
                'num_path': self.num_path,
                'edge_disjoint': self.edge_disjoint,
                'dist_metric': self.dist_metric,
                'traffic_model': tm_fname.split('/')[-2],
                'traffic_seed': int(tm_fname.split('_')[-3]),
                'scale_factor': float(tm_fname.split('_')[-2]),
                'total_demand': self.obs[
                    -self.num_path_node::self.num_path].sum().item(),
            }
        return problem_dict

    def step(self, raw_action, num_sample=0, num_admm_step=0):
        """Return the reward of current action."""
        info = {}

        if self.leo_mode:
            reward, info = self._step_leo(raw_action, num_sample)
        else:
            if self.idx_start == self.train_start:
                reward = self.take_action(raw_action, num_sample)
            else:
                start_time = time.time()
                action = self.transform_raw_action(raw_action)
                if self.obj == 'total_flow':
                    action = self.ADMM.tune_action(
                        self.obs, action, num_admm_step)
                    action = self.round_action(action)
                info['runtime'] = time.time() - start_time
                info['sol_mat'] = self.extract_sol_mat(action)
                reward = self.get_obj(action)

        self._next_obs()
        return reward, info

    def _step_leo(self, raw_action, num_sample):
        """Step in ELATE mode."""
        info = {}

        if self.num_elephant == 0:
            info['runtime'] = 0.0
            info['sol_mat'] = None
            mlu = (self.mice_edge_flow / self.capacity.to(self.device)).max()
            return -mlu, info

        if self.idx_start == self.train_start:
            reward = self._compute_leo_reward(raw_action, num_sample)
        else:
            start_time = time.time()
            action = self._transform_leo_action(raw_action)
            info['runtime'] = time.time() - start_time
            info['sol_mat'] = self.extract_sol_mat(action) if \
                self.p2e.shape[1] > 0 else None
            reward = -self._compute_mlu(action)

        return reward, info

    def _transform_leo_action(self, raw_action):
        """Transform raw action to flow allocation (softmax split ratios)."""
        raw_action = torch.clamp(
            raw_action, min=self.raw_action_min, max=self.raw_action_max)
        split_ratios = torch.softmax(
            raw_action.reshape(-1, self.num_path), dim=-1)
        demands = self.obs[-self.num_path_node::self.num_path]
        action = (split_ratios * demands[:, None]).flatten()
        return action

    def _compute_mlu(self, action):
        """Compute MLU (Eq. 1): η = max_e (flow_on_e / c(e))."""
        if self.p2e.shape[1] == 0:
            edge_flow = self.mice_edge_flow
        else:
            elephant_edge_flow = torch_scatter.scatter(
                action[self.p2e[0]], self.p2e[1],
                dim_size=self.num_edge_node)
            edge_flow = self.mice_edge_flow + elephant_edge_flow

        mlu = (edge_flow / self.capacity.to(self.device)).max()
        return mlu

    def _compute_leo_reward(self, raw_action, num_sample):
        """Compute counterfactual MARL reward (Eq. 9).

        A_k = R(s,a) - (1/N_s) * Σ R(s, (a_{-k}, a'_m))
        """
        action = self._transform_leo_action(raw_action)
        current_mlu = self._compute_mlu(action)
        current_reward = -current_mlu

        advantage = torch.zeros(self.num_elephant).to(self.device)

        for k in range(self.num_elephant):
            counterfactual_rewards = 0.0
            for _ in range(num_sample):
                cf_raw_action = raw_action.clone()
                cf_raw_action[k] = torch.FloatTensor(
                    self.num_path).uniform_(
                    self.raw_action_min, self.raw_action_max).to(self.device)
                cf_action = self._transform_leo_action(cf_raw_action)
                cf_mlu = self._compute_mlu(cf_action)
                counterfactual_rewards += -cf_mlu

            baseline = counterfactual_rewards / num_sample
            advantage[k] = current_reward - baseline

        return advantage

    # ===== Original TEAL methods (unchanged) =====

    def get_obj(self, action):
        """Return objective (TEAL mode)."""
        if self.obj == 'total_flow':
            return action.sum(axis=-1)
        elif self.obj == 'min_max_link_util':
            return (torch_scatter.scatter(
                action[self.p2e[0]], self.p2e[1]
                )/self.obs[:-self.num_path_node]).max()

    def transform_raw_action(self, raw_action):
        """Return network flow allocation as action (TEAL mode)."""
        raw_action = torch.clamp(
            raw_action, min=self.raw_action_min, max=self.raw_action_max)

        if self.obj == 'min_max_link_util':
            raw_action = torch.softmax(raw_action, dim=-1)
        else:
            raw_action = raw_action.exp()
            raw_action = raw_action/(1+raw_action.sum(axis=-1)[:, None])

        raw_action = raw_action.flatten() * self.obs[-self.num_path_node:]
        return raw_action

    def round_action(
            self, action, round_demand=True, round_capacity=True,
            num_round_iter=2):
        """Return rounded action (TEAL mode)."""

        demand = self.obs[-self.num_path_node::self.num_path]
        capacity = self.obs[:-self.num_path_node]

        if round_demand:
            action = action.reshape(-1, self.num_path)
            ratio = action.sum(-1) / demand
            action[ratio > 1, :] /= ratio[ratio > 1, None]
            action = action.flatten()

        if round_capacity:
            path_flow = action
            path_flow_allocated_total = torch.zeros(path_flow.shape)\
                .to(self.device)
            for round_iter in range(num_round_iter):
                edge_flow = torch_scatter.scatter(
                    path_flow[self.p2e[0]], self.p2e[1])
                util = 1 + (edge_flow/capacity-1).relu()
                util = torch_scatter.scatter(
                    util[self.p2e[1]], self.p2e[0], reduce="max")
                path_flow_allocated = path_flow/util
                path_flow_allocated_total += path_flow_allocated
                if round_iter != num_round_iter - 1:
                    capacity = (capacity - torch_scatter.scatter(
                        path_flow_allocated[self.p2e[0]], self.p2e[1])).relu()
                    path_flow = path_flow - path_flow_allocated
            action = path_flow_allocated_total

        return action

    def take_action(self, raw_action, num_sample):
        """Return approximate reward for action (TEAL mode)."""

        path_flow = self.transform_raw_action(raw_action)
        edge_flow = torch_scatter.scatter(path_flow[self.p2e[0]], self.p2e[1])
        util = edge_flow/self.obs[:-self.num_path_node]

        distribution = Uniform(
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_min,
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_max)
        reward = torch.zeros(self.num_path_node//self.num_path).to(self.device)

        if self.obj == 'total_flow':

            util, path_bottleneck = torch_scatter.scatter_max(
                util[self.p2e[1]], self.p2e[0])
            path_bottleneck = self.p2e[1][path_bottleneck]

            coef = path_flow/util**2
            coef[util < 1] = 0
            coef = torch_scatter.scatter(
                coef, path_bottleneck).reshape(-1, 1)

            bottleneck_p2e = torch.sparse_coo_tensor(
                self.p2e, (1/self.obs[:-self.num_path_node])[self.p2e[1]],
                [self.num_path_node, self.num_edge_node])

            for _ in range(num_sample):
                sample = distribution.rsample()

                delta_path_flow = self.transform_raw_action(sample) - path_flow
                reward += -(delta_path_flow/(1+(util-1).relu()))\
                    .reshape(-1, self.num_path).sum(-1)

                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                delta_util = torch.sparse.mm(delta_path_flow, bottleneck_p2e)
                reward += torch.sparse.mm(delta_util, coef).flatten()

        elif self.obj == 'min_max_link_util':

            max_util_edge = util.argmax()

            max_util_paths = torch.zeros(self.num_path_node).to(self.device)
            max_util_paths[self.p2e[0, self.p2e[1] == max_util_edge]] =\
                1/self.obs[max_util_edge]

            for _ in range(num_sample):
                sample = distribution.rsample()

                delta_path_flow = self.transform_raw_action(sample) - path_flow
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                reward += torch.sparse.mm(
                    delta_path_flow, max_util_paths.reshape(-1, 1)).flatten()

        return reward/num_sample

    def _read_graph_json(self, topo):
        """Return network topo from json file."""
        assert topo.endswith(".json")
        with open(os.path.join(TOPOLOGIES_DIR, topo)) as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)

    def path_full_fname(self, topo, num_path, edge_disjoint, dist_metric):
        """Return full name of the topology path."""
        return os.path.join(
            TOPOLOGIES_DIR, "paths", "path-form",
            "{}-{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl".format(
                topo, num_path, edge_disjoint, dist_metric))

    def get_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary."""
        self.path_fname = self.path_full_fname(
            topo, num_path, edge_disjoint, dist_metric)
        print("Loading paths from pickle file", self.path_fname)
        try:
            with open(self.path_fname, 'rb') as f:
                path_dict = pickle.load(f)
                print("path_dict size:", len(path_dict))
                return path_dict
        except FileNotFoundError:
            print("Creating paths {}".format(self.path_fname))
            path_dict = self.compute_path(
                topo, num_path, edge_disjoint, dist_metric)
            print("Saving paths to pickle file")
            with open(self.path_fname, "wb") as w:
                pickle.dump(path_dict, w)
        return path_dict

    def compute_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary through computation."""
        path_dict = {}
        G = graph_copy_with_edge_weights(self.G, dist_metric)
        for s_k in G.nodes:
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(G, s_k, t_k, num_path, edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                path_dict[(s_k, t_k)] = paths_no_cycles
        return path_dict

    def get_regular_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary with the same number of paths per demand."""
        path_dict = self.get_path(topo, num_path, edge_disjoint, dist_metric)
        for (s_k, t_k) in path_dict:
            if len(path_dict[(s_k, t_k)]) < self.num_path:
                path_dict[(s_k, t_k)] = [
                    path_dict[(s_k, t_k)][0] for _
                    in range(self.num_path - len(path_dict[(s_k, t_k)]))]\
                    + path_dict[(s_k, t_k)]
            elif len(path_dict[(s_k, t_k)]) > self.num_path:
                path_dict[(s_k, t_k)] = path_dict[(s_k, t_k)][:self.num_path]
        return path_dict

    def get_topo_matrix(self, topo, num_path, edge_disjoint, dist_metric):
        """Return matrices related to topology (TEAL mode)."""

        path_dict = self.get_regular_path(
            topo, num_path, edge_disjoint, dist_metric)

        edge2idx_dict = {edge: idx for idx, edge in enumerate(self.G.edges)}
        node2degree_dict = {}
        edge_num = len(self.G.edges)

        src, dst, path_i = [], [], 0
        for s in range(len(self.G)):
            for t in range(len(self.G)):
                if s == t:
                    continue
                for path in path_dict[(s, t)]:
                    for (u, v) in zip(path[:-1], path[1:]):
                        src.append(edge_num+path_i)
                        dst.append(edge2idx_dict[(u, v)])

                        if src[-1] not in node2degree_dict:
                            node2degree_dict[src[-1]] = 0
                        node2degree_dict[src[-1]] += 1
                        if dst[-1] not in node2degree_dict:
                            node2degree_dict[dst[-1]] = 0
                        node2degree_dict[dst[-1]] += 1
                    path_i += 1

        edge_index_values = torch.tensor(
            [1/math.sqrt(node2degree_dict[u]*node2degree_dict[v])
                for u, v in zip(src+dst, dst+src)]).to(self.device)
        edge_index = torch.tensor(
            [src+dst, dst+src], dtype=torch.long).to(self.device)
        p2e = torch.tensor([src, dst], dtype=torch.long).to(self.device)
        p2e[0] -= len(self.G.edges)

        return edge_index, edge_index_values, p2e

    def extract_sol_mat(self, action):
        """Return sparse solution matrix."""
        sol_mat_index = torch.stack([
            self.p2e[0] % self.num_path,
            torch.div(self.p2e[0], self.num_path, rounding_mode='floor'),
            self.p2e[1]])

        sol_mat = torch.sparse_coo_tensor(
            sol_mat_index,
            action[self.p2e[0]],
            (self.num_path,
                self.num_path_node//self.num_path,
                self.num_edge_node))
        sol_mat = torch.sparse.sum(sol_mat, [0])

        return sol_mat
