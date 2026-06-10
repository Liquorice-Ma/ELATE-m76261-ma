import networkx as nx


def generate_leo_grid(N=72, M=22, capacity=1.0):
    """Generate a +Grid LEO constellation topology.

    Each satellite (x, y) connects to 4 neighbors via ISLs:
    - Intra-orbit: (x, y±1 mod M)
    - Inter-orbit: (x±1 mod N, y)

    Args:
        N: number of orbits
        M: number of satellites per orbit
        capacity: uniform ISL capacity

    Returns:
        G: NetworkX DiGraph with node/edge attributes
        node_to_coord: dict mapping node index to (orbit, slot) tuple
        coord_to_node: dict mapping (orbit, slot) to node index
    """
    G = nx.DiGraph()
    node_to_coord = {}
    coord_to_node = {}

    for x in range(N):
        for y in range(M):
            node_id = x * M + y
            G.add_node(node_id)
            node_to_coord[node_id] = (x, y)
            coord_to_node[(x, y)] = node_id

    for x in range(N):
        for y in range(M):
            src = coord_to_node[(x, y)]
            # intra-orbit links
            dst_y_plus = coord_to_node[(x, (y + 1) % M)]
            dst_y_minus = coord_to_node[(x, (y - 1) % M)]
            # inter-orbit links
            dst_x_plus = coord_to_node[((x + 1) % N, y)]
            dst_x_minus = coord_to_node[((x - 1) % N, y)]

            for dst in [dst_y_plus, dst_y_minus, dst_x_plus, dst_x_minus]:
                if not G.has_edge(src, dst):
                    G.add_edge(src, dst, capacity=capacity)

    return G, node_to_coord, coord_to_node
