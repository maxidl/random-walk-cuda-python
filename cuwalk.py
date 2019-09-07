import cudf
import cugraph
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np


@cuda.jit
def generate_walks(start_nodes, out, offsets, indices, rng_states):
    thread_id = cuda.grid(1)
    if thread_id < start_nodes.size:  # Check array boundaries
        start_node = start_nodes[thread_id]
        out[thread_id][0] = start_node
        curr_node = start_node
        for i in range(1, out.shape[1]):  # starts at 1 due to first walk element being the start node
            # get neighbors
            if curr_node == -1:
                next_node = curr_node
            else:
                start_idx = offsets[curr_node]
                end_idx = offsets[curr_node + 1]
                neighbors = indices[start_idx:end_idx]
                num_neighbors = len(neighbors)
                if num_neighbors > 0:
                    rand_float = xoroshiro128p_uniform_float32(rng_states, thread_id)
                    choice = int(rand_float * num_neighbors)
                    next_node = neighbors[choice]
                else:
                    next_node = -1
            out[thread_id][i] = next_node
            curr_node = next_node


def main():
    """
    Just a short test driver
    """
    import time
    # gdf = cudf.read_csv('zachary.ssv', header=None, sep=' ', dtype=['int32', 'int32'])
    gdf = cudf.read_csv('Flickr-labelled.edgelist', header=None, sep=' ', dtype=['int32', 'int32'])
    gdf.columns = ['src', 'dest']
    gdf = gdf.sort_values(by='src')
    G = cugraph.Graph()
    G.add_edge_list(gdf['src'], gdf['dest'])
    adj_list = G.view_adj_list()
    offsets, indices = adj_list[0], adj_list[1]
    offsets, indices = cuda.to_device(offsets), cuda.to_device(indices)
    nodes = gdf['src'].unique().values

    walk_length = 80
    walks_per_node = 100
    start_nodes = np.hstack([nodes] * walks_per_node)
    out = np.full_like(start_nodes, -2).repeat(walk_length).reshape(start_nodes.shape[0], walk_length)
    print(out.nbytes / 1e9)
    threads_per_block = 64
    blocks_per_grid = (start_nodes.size + (threads_per_block - 1)) // threads_per_block
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)
    start_time = time.time()
    generate_walks[blocks_per_grid, threads_per_block](start_nodes, out, offsets, indices, rng_states)
    generation_time = time.time() - start_time
    print(out)
    print(f'Generated {out.shape[0]} random walks of length {out.shape[1]}')
    print(f'Generation time: {generation_time}')


if __name__ == '__main__':
    main()
