import networkx as nx
from itertools import islice


def compute_offset(src, dst, N, M):
    """Compute relative offset between two satellites (Eq. 5).

    Args:
        src: (x_i, y_i) coordinates of source satellite
        dst: (x_j, y_j) coordinates of destination satellite
        N: number of orbits
        M: satellites per orbit

    Returns:
        (dx, dy): offset tuple with wrap-around
    """
    dx = (dst[0] - src[0]) % N
    dy = (dst[1] - src[1]) % M
    return (dx, dy)


def precompute_offset_paths(G, N, M, num_path, coord_to_node):
    """Pre-compute shortest paths from origin (0,0) to all unique offsets.

    Exploits translational invariance of +Grid topology.
    Only needs to compute paths for O(|V|) offsets instead of O(|V|^2) pairs.

    Args:
        G: LEO grid graph
        N: number of orbits
        M: satellites per orbit
        num_path: number of candidate paths per offset
        coord_to_node: mapping from (orbit, slot) to node index

    Returns:
        offset_paths: dict mapping (dx, dy) -> list of paths (as coord lists)
    """
    origin = coord_to_node[(0, 0)]
    offset_paths = {}

    for dx in range(N):
        for dy in range(M):
            if dx == 0 and dy == 0:
                continue
            dst = coord_to_node[(dx, dy)]
            try:
                paths = list(islice(
                    nx.shortest_simple_paths(G, origin, dst, weight=None),
                    num_path))
            except nx.NetworkXNoPath:
                paths = []

            if len(paths) < num_path and len(paths) > 0:
                paths = [paths[0]] * (num_path - len(paths)) + paths

            offset_paths[(dx, dy)] = paths

    return offset_paths


def instantiate_paths(offset_paths, src_coord, dst_coord, N, M,
                      coord_to_node, node_to_coord):
    """Instantiate paths for a specific source-destination pair (Eq. 6).

    P_k = {(p + v_i) mod (N, M) | p ∈ Ω_{Δ_{i,j}}}

    Args:
        offset_paths: pre-computed paths from origin
        src_coord: (x_i, y_i) of source
        dst_coord: (x_j, y_j) of destination
        N: number of orbits
        M: satellites per orbit
        coord_to_node: mapping from coords to node id
        node_to_coord: mapping from node id to coords

    Returns:
        paths: list of paths (each path is a list of node indices)
    """
    offset = compute_offset(src_coord, dst_coord, N, M)
    if offset not in offset_paths:
        return []

    base_paths = offset_paths[offset]
    shifted_paths = []

    for path in base_paths:
        shifted_path = []
        for node_id in path:
            coord = node_to_coord[node_id]
            new_coord = ((coord[0] + src_coord[0]) % N,
                         (coord[1] + src_coord[1]) % M)
            shifted_path.append(coord_to_node[new_coord])
        shifted_paths.append(shifted_path)

    return shifted_paths


def compute_all_elephant_paths(elephant_pairs, N, M, num_path,
                               offset_paths, coord_to_node, node_to_coord):
    """Compute paths for all elephant flow pairs.

    Args:
        elephant_pairs: list of (src_node, dst_node) for elephant flows
        N, M: grid dimensions
        num_path: paths per demand
        offset_paths: pre-computed offset paths
        coord_to_node, node_to_coord: coordinate mappings

    Returns:
        path_dict: dict mapping (src, dst) -> list of paths
    """
    path_dict = {}
    for (src, dst) in elephant_pairs:
        src_coord = node_to_coord[src]
        dst_coord = node_to_coord[dst]
        paths = instantiate_paths(
            offset_paths, src_coord, dst_coord, N, M,
            coord_to_node, node_to_coord)
        path_dict[(src, dst)] = paths
    return path_dict
