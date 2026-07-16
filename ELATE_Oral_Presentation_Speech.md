## Oral Presentation Speech — ELATE

**Estimated duration: ~10 minutes at a moderate speaking pace (~140 words/min)**

---

### Opening & Motivation (~2 min)

Good morning/afternoon, everyone. Thank you for being here. My name is [Speaker Name], from Beijing University of Posts and Telecommunications. Today, I am excited to present our work titled "ELATE: Exploring Elephant Flow-aware Learning-Accelerated Traffic Engineering in Large-Scale LEO Constellations."

Let me begin with the big picture. Large-scale low earth orbit constellations — such as SpaceX's Starlink and Amazon's Project Kuiper — are rapidly reshaping global communication. With thousands of satellites orbiting the Earth, they provide low-latency, high-bandwidth connectivity to users worldwide, including underserved and remote regions. As these constellations scale up, however, a fundamental challenge emerges: how do we efficiently manage the enormous and ever-changing traffic flowing through these networks?

This is precisely the domain of Traffic Engineering, or TE. The goal of TE is to optimize how traffic is distributed across network links — minimizing congestion, reducing latency, and preventing bottlenecks. In terrestrial networks, this is already a hard problem. But in LEO constellations, it becomes significantly harder for two reasons. First, the topology is highly dynamic — satellites move at roughly 27,000 kilometers per hour, so the network graph changes continuously. Second, the scale is massive — we are talking about thousands of nodes and links, with traffic demands that fluctuate rapidly over time.

Traditional optimization approaches, such as linear programming, simply cannot keep up. They are computationally expensive and struggle to scale. Even recent advances using segment routing or machine learning often fail to strike the right balance between effective load balancing and the speed of decision-making that LEO networks demand.

This is the gap our work aims to fill.

---

### Problem Formulation (~1.5 min)

Let me formalize the problem briefly. We model the LEO constellation as a graph, where each satellite is a node and each inter-satellite link is an edge. In our study, we use a 2-D grid topology — each satellite connects to its neighbors within the same orbit and to the nearest satellites in adjacent orbits. Importantly, although the satellites themselves are moving, their relative positions in this grid remain invariant over time, which gives us a stable structure to work with.

Our objective is to minimize the Maximum Link Utilization, or MLU. MLU is defined as the highest ratio of traffic load to link capacity across all links in the network. Intuitively, a lower MLU means traffic is more evenly distributed, and the network is less congested.

The decision variables are the flow splitting ratios — for each traffic demand, we decide what fraction of its volume goes along each available shortest path. The constraints ensure that the total load on each link does not exceed its capacity scaled by the MLU, and that all traffic demand is fully routed.

Now, the challenge is that when you have thousands of demands, each with multiple candidate paths, the solution space explodes. This is where our key insight comes in.

---

### The ELATE Framework (~4 min)

Our framework, ELATE, is built on a simple but powerful observation about satellite traffic: it exhibits a heavy-tailed distribution. A small number of large flows — which we call "elephant flows" — dominate network congestion, while a massive number of small flows — "mice flows" — contribute only marginally to bandwidth consumption. ELATE exploits this structure through a three-stage pipeline.

**Stage 1: Elephant Flow Identification.**

In the first stage, we partition all traffic demands into elephant flows and mice flows using a statistical thresholding strategy. Specifically, we compute the mean and standard deviation of the traffic matrix, and set a threshold at mean plus lambda times the standard deviation. Flows exceeding this threshold are classified as elephant flows; the rest are mice flows.

All mice flows are then statically routed along their shortest paths, and their cumulative traffic is subtracted from the link capacities. This gives us a residual capacity topology, and dramatically reduces the solution space — because now we only need to optimize the routing of the identified elephant flows, which are far fewer in number.

To further accelerate path computation, we exploit the translation invariance of the LEO ring-like grid. Instead of running expensive all-pairs shortest path searches online, we pre-compute paths from a virtual origin to all possible offsets and cache them. For any specific source-destination pair, the actual paths are obtained by a simple spatial shift. This reduces the per-flow path computation from O of V squared to O of V, which is a significant speedup.

**Stage 2: GNN-Derived Elephant Flow Embedding.**

In the second stage, we need a good state representation that captures the dynamic LEO topology and the interactions between flows and links. Standard node-centric graph neural networks are not well-suited here, because the core routing challenge is about path-link interference — which links are shared by which paths, and where the bottlenecks are.

To address this, we model the network as a time-dependent bipartite graph. One set of nodes represents the active links, and the other set represents the candidate paths. The edges capture which links are traversed by which paths. We then apply a GNN with interleaved message-passing layers on this bipartite structure.

The key insight is that one direction of message passing aggregates the congestion states of all links along a path — giving us an end-to-end path quality measure. The other direction aggregates the traffic capacity of all paths traversing a link — capturing the bottleneck effect. This alternating message passing naturally encodes the network dynamics into rich embeddings.

Additionally, to enforce the flow conservation constraint — that the splitting ratios for all paths of a given demand must sum to one — we interleave DNN layers after each GNN layer. These DNN layers coordinate across all candidate paths of each elephant flow, ensuring physically valid routing decisions.

**Stage 3: Multi-Agent Reinforcement Learning.**

In the third and final stage, we map these learned embeddings into actual routing actions using a multi-agent reinforcement learning framework. Here, each elephant flow is modeled as an independent agent. At each decision epoch, each agent observes its local state — formed by its candidate path embeddings — and selects a continuous flow-splitting action from a shared stochastic policy network.

All agents are trained under the centralized training with decentralized execution paradigm, using the negative MLU as the global reward. To address the credit assignment problem — that is, determining which agent's action actually helped or hurt — we adopt a counterfactual multi-agent algorithm. The key domain insight is that traffic engineering is essentially a one-step process: today's routing decisions do not affect tomorrow's traffic demands. This allows us to use the immediate global reward directly, and approximate the counterfactual baseline via Monte Carlo sampling. The resulting advantage function robustly quantifies whether each agent's chosen action outperforms its average behavior, leading to stable and efficient policy updates.

The entire framework — GNN, DNN, and policy network — is trained end-to-end via multi-agent policy gradients. Once training is complete, the inference is purely feedforward, which is what gives ELATE its remarkable speed at deployment time.

---

### Simulation Results (~2 min)

Now let me share the key results. We evaluated ELATE on the Starlink Phase-I Shell-I constellation, comprising 1,584 satellites across 72 orbits with 22 satellites per orbit. Traffic was generated using the random Gravity model, with city selection probabilities proportional to either GDP or population, giving us 100 GDP-based and 100 population-based traffic matrices for robust evaluation.

We compared ELATE against five representative baselines: Shortest Path routing, Shortest Path with Equal Splitting, Linear Programming solved by Gurobi, Local Search, and our prior work CESLP which uses distributed cluster-based optimization.

In terms of load balancing performance, ELATE consistently achieves the lowest normalized MLU across both traffic types. Specifically, ELATE achieves a median MLU of approximately 0.53, yielding an 8 percent improvement over the closest baseline CESLP, and up to 29 percent improvement over simple shortest path routing. These results hold consistently across both GDP-based and population-based traffic matrices.

But where ELATE truly shines is in computational efficiency. ELATE averages just 10.57 milliseconds per traffic matrix on an NVIDIA RTX 4090 GPU. In contrast, shortest path algorithms take about 115 to 123 milliseconds, linear programming takes 34.76 seconds, and local search exceeds 100 seconds. Even CESLP, which leverages GPU acceleration, is bottlenecked by its iterative LP sub-problems and requires up to 60 seconds. This means ELATE achieves between 11 times and 3,288 times speedup over the baselines — a truly dramatic improvement that brings real-time, millisecond-scale traffic engineering within reach for large-scale LEO constellations.

We also conducted a thorough ablation study, which confirmed that every component of ELATE is essential. Removing the elephant-mice flow separation causes the neural network to be overwhelmed by state complexity. Replacing our edge-centric GNN with a standard node-based GNN leads to severe performance degradation, especially under widely distributed population-based traffic. And replacing MARL with surrogate loss minimization consistently degrades load balancing, confirming that reinforcement learning is more robust for directly optimizing the non-differentiable MLU objective.

---

### Conclusion (~0.5 min)

To conclude, ELATE is a novel, highly scalable traffic engineering framework designed specifically for the extreme dynamics of large-scale LEO constellations. By exploiting the heavy-tailed nature of satellite traffic, modeling path-link interference through a bipartite graph neural network, and leveraging multi-agent reinforcement learning for adaptive routing, ELATE achieves superior load balancing with 8 to 29 percent lower MLU, while delivering 11 to 3,288 times computational speedups — enabling millisecond-scale flow allocation that is well-suited for the rapid re-optimization demands of next-generation LEO networks.

Thank you very much for your attention. I would be happy to take any questions.
