from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
from collections import defaultdict

from vis import *

class TopDownSorting:

    def __init__(self, raw_data: csr_matrix, raw_weights : np.ndarray):
        self.raw_data = raw_data
        self.raw_weights = raw_weights


    def nz_columns(self, A : csr_matrix, row : int):
        assert 0 <= row < A.shape[0]
        return np.array(A.indices[A.indptr[row] : A.indptr[row + 1]])
    

    def expand_edge(self, link : list[int], n_new : int):
        '''
        Takes child edges of one edge in an iteration as links to raw data, 
            then bins & merges & sorts lexicographically. 

        link: indices of edges in the raw data to bin, has length k
        n_new: n of previous * 2

        returns: local csr_matrix (d x n) containing binned & sorted child edges of the input link
                 weights W of length d with d < k
                 new links of length d for each d new edges
        '''
        # Build map {binned vertices in edge} -> frequency
        insertion_map : dict[tuple[int, ...], int] = defaultdict(int)
        weights_map : dict[tuple[int, ...], float] = defaultdict(float)
        link_map : dict[tuple[int, ...], list[int]] = defaultdict(list[int]) # Keep track of which raw edges correspond to subedges
        for raw_edge in link:
            # Bin raw edges in this link according to new resolution n
            n_old = self.raw_data.shape[1]
            bin_size = n_old / n_new
            cols = tuple(np.unique(
                np.floor(self.nz_columns(self.raw_data, raw_edge) / bin_size).astype(int)
            ))
            insertion_map[cols] += 1
            weights_map[cols] += self.raw_weights[raw_edge]
            link_map[cols].append(raw_edge)

        # Build csr matrix from map with d << k
        d = len(insertion_map)
        sub_W = np.zeros(d)
        sub_links : list[list[int]] = [[] for _ in range(d)]
        sub_csr_rows = []
        sub_csr_cols_all = []   

        for i, (cols, _) in enumerate(insertion_map.items()):
            sub_csr_rows.extend([i] * len(cols))
            sub_csr_cols_all.extend(cols)
            sub_W[i] = weights_map[cols]
            sub_links[i] = link_map[cols]
        data = np.ones(len(sub_csr_rows))
        sub_csr = csr_matrix((data, (sub_csr_rows, sub_csr_cols_all)), shape=(d, n_new))


        # Sort lexicographically by vertex position
        def lex_key(A: csr_matrix, i: int):
            start, end = A.indptr[i], A.indptr[i + 1]
            cols = A.indices[start:end]
            return (cols[0], cols[-1])
        
        order = sorted(range(d), key = lambda i: lex_key(sub_csr, i))
        sub_csr = sub_csr[order]
        sub_W = sub_W[order]
        sub_links = [sub_links[i] for i in order]

        return sub_csr, sub_W, sub_links



    def sort_at_resolution(self, n : float, prev_links : list[list[int]]):
        '''
        Main function for computing H & W at each resolution given the previous smaller one. 

        n: previous resolution
        prev_links: list of size m, maps to all raw edges aggregated into each row of prev_H
                    In the actual program, these will point to locations to a raw datafile on disk,
                    or uniqueptrs if we can spare the memory to load all of a SPRITE instance.  

        returns: higher resolution incidence H, edge weights W, and new links
        '''
        n_new = int(n * 2)
        m = len(prev_links)

        stack_H = []
        stack_W = []
        combined_links = []
        for prev_edge in range(m):
            link = prev_links[prev_edge]
            sub_H, sub_W, sub_links = self.expand_edge(link, n_new)
            stack_H.append(sub_H)
            stack_W.append(sub_W)
            combined_links.extend(sub_links)

        combined_H = sp.vstack(stack_H)
        combined_W = np.hstack(stack_W)
        print(f'Links at n={n} : {combined_links}')

        return combined_H, combined_W, combined_links
    
    
    def build_cache(self, resolution_range : tuple[int, int]):
        '''
        Main function for computing all resultions by iteratively calling sort_at_resolution. 

        resolution_range: (min #bins, max #bins). Should be powers of 2. 
        returns: list of tuples: [(H, W, links), ...]
        '''

        min_n, max_n = resolution_range

        # Starting link: one 'super edge' linked to everything in data. 
        # This is the most convenient thing to do for now. 
        m = len(self.raw_weights)
        prev_links : list[list[int]] = [list(range(m))]
        output = []

        current_n = min_n / 2
        while current_n < max_n:
            cached = self.sort_at_resolution(current_n, prev_links)
            H, _, combined_links = cached
            _, current_n = H.shape
            prev_links = combined_links
            output.append(cached)

        return output
