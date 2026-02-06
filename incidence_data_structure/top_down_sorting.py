from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
from collections import defaultdict

class TopDownSorting:

    def __init__(self, raw_data: csr_matrix, raw_weights : np.ndarray):
        self.raw_data = raw_data
        self.raw_weights = raw_weights


    def nz_columns(self, A : csr_matrix, row : int):
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
        k = len(link)

        # Build map {binned vertices in edge} -> frequency
        insertion_map : map[tuple, int] = defaultdict(int)
        weights_map : map[tuple, float] = defaultdict(float)
        link_map : map[tuple, list[int]] = defaultdict(list[int]) # Keep track of which raw edges correspond to subedges
        for raw_edge in link:
            # Bin raw edges in this link according to new resolution n
            cols = tuple(np.floor(self.nz_columns(self.raw_data, raw_edge) / n_new).astype(int))
            insertion_map[cols] += 1
            weights_map[cols] += self.raw_data[raw_edge]
            link_map[cols].append(raw_edge)

        # Build csr matrix from map with d << k
        d = len(insertion_map)
        sub_csr = csr_matrix((d, n_new))
        sub_W = np.zeros(d)
        sub_links : list[list[int]] = [[] for _ in range(d)]

        for i, (cols, _) in enumerate(insertion_map.items()):
            sub_csr[i, list(cols)] = 1
            sub_W[i] = weights_map[cols]
            sub_links[i] = sub_links[cols]

        # Sort lexicographically by vertex position
        def lex_key(A: csr_matrix, i: int):
            start, end = A.indptr[i], A.indptr[i + 1]
            cols = A.indices[start:end]
            return (cols[0], cols[-1])
        
        order = sorted(range(d), key = lambda i: lex_key(sub_csr, i))
        sub_csr = sub_csr[order]
        sub_W = sub_W[order]
        sub_links = sub_links[order]

        return sub_csr, sub_W, sub_links



    def sort_at_resolution(self, m : int, n : int, prev_links : list[list[int]]):
        '''
        Main function for computing H & W at each resolution given the previous smaller one. 

        m: previous # edges
        n: previous resolution
        prev_links: list of size m, maps to all raw edges aggregated into each row of prev_H
                    In the actual program, these will point to locations to a raw datafile on disk,
                    or uniqueptrs if we can spare the memory to load all of a SPRITE instance.  

        returns: higher resolution incidence H, edge weights W, and new links
        '''
        n_new = n * 2

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

        return combined_H, combined_W, combined_links