import os
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
import pickle
import time

from joblib import Parallel, delayed
from tqdm import tqdm

from visualization import histogram, powerfit
from config import *

from data_generation import DataGenerator
from chiadrop_generation import ChiaDropGenerator

from .format_SPRITE import *

from preprocessed_data import PreprocessedData

def process_genomic_dist_histogram(H : csr_matrix, title=""):
    A = H @ H.T # clique exp
    A.setdiag(0)
    A.eliminate_zeros()
    N = A.shape[0]
    A_arr = A.toarray()

    X = np.arange(1, N)
    Y = np.array([np.mean(np.diag(A_arr, k=x)) for x in X])
    powerfit(X, Y, 
        xlabel=f'Genomic dist in bins (bin_size={int(CONFIG.bin_size * 1e-3)}kb)',
        ylabel='Incidence diag mean',
        title=title
    )

class Preprocessing:
    def __init__(self, clique_adjust, name=""):
        """
        Input:      clique_adjust: If true, take the square root of the weights to prepare for clique expansion correctly. 
                    genomic_range: If not none, limit range to genomic_range = [start, end] in bp. 
                    name: Title for visualization
        """
        self.ignore_bin_range = False
        self.clique_adjust = clique_adjust
        self.name = name
        self.file_name = ''
        self.data_type = ''
        self.non_rep = ''

        self.visualize_edge_counts = CONFIG.cfg['visualization']['visualize_edge_histogram']
        self.visualize_edge_order_fits = CONFIG.cfg['visualization']['visualize_edge_order_fits']
        self.visualize_genomic_dist = CONFIG.cfg['visualization']['visualize_genomic_dist']

        self.bin_range = None
        if CONFIG.genomic_range is not None:
            self.bin_range = [int(r / CONFIG.bin_size) for r in CONFIG.genomic_range] # convert from bp to bin_size

    
    def cache_fname(self, file_path : str):
        """
        Function to generate cache filenames so that runs with different parameters have different caches.
        """
        def to_kb(x):
            return f"{int(x/1e3)}kb"
        gr = None
        if CONFIG.genomic_range is not None:
            gr = f"{to_kb(CONFIG.genomic_range[0])},{to_kb(CONFIG.genomic_range[1])}"
        out = f"{os.path.basename(file_path)};bs={to_kb(CONFIG.bin_size)};gr={gr};chr={CONFIG.cfg['chr']}"

        ES = CONFIG.EDGE_SIZE_FILTER
        if ES != [2, 100]:
            out += f";esf=[{ES[0]},{ES[1]}]"
        return out
            

    @abstractmethod
    def process_hyperedge(self, hyperedge, CONFIG, **kwargs):
        """
        Implement this for parsing a df row into a hyperedge
        """
        pass

    @abstractmethod
    def preprocess(self, filename, proportion:Optional[float]=None, only_statistics=False, test_noise_level=None, test_noise_func=None):
        """
        Implement this for reading in the file and constructing hypergraph(s)
        """
        pass

    def preprocess_restore_index(self, index, *args, **kwargs):
        """
        Sometimes we preprocess several reps at once (e.g. in SPRITE). To save memory (for delayed preprocessing), 
        we need to store specific replicates using *index* from disk using this rather than all of them. 
        """
        return self.preprocess(*args, **kwargs)


    def encode_incidence(self, freq_map_out, CONFIG, only_statistics=False, binding_intensity=None, timing=False) -> PreprocessedData:
        """
        Encode the hypergraph as an incidence matrix H. The incidence matrix H has shape (n, |E|) 
        whose value at (v,e) is w(e)/|e| where w(e) is the edge frequency. We also determine weight matrix W and degree matrix D. 
        Input:      freq_map_out: should be a tuple of (freq_map, num_v, num_e) from calling encode_freq_map
                    only_statistics: If true, do not find H and only calculate statistics
                    binding_intensity: optional ChiaDrop binding intensity
        Returns:    The incidence H (n, |E|) .
        """
        if timing:
            TIME_start = time.perf_counter()

        # Obtain freq map [list of vertex bins] -> [frequency (int)] to determine n and |E|
        freq_map, num_v, num_e = freq_map_out

        # Record edge histogram
        hyperedge_sizes = []
        for e, (hyperedge, count) in enumerate(freq_map.items()):
            hyperedge_sizes.append(len(hyperedge))
        hyperedge_sizes_map = Counter(hyperedge_sizes) # Encodes [size(int)] -> [frequency (int)]
        if only_statistics: return PreprocessedData()

        # Visualize edge histogram
        if self.visualize_edge_counts:
            X = np.array(list(hyperedge_sizes_map.keys()))
            Y = np.array(list(hyperedge_sizes_map.values()))
            powerfit(X, Y, 
                xlabel=f'Hyperedge Order',
                ylabel='Frequency',
                title=f'Edge orders : {self.file_name}',
                dofit=self.visualize_edge_order_fits
            )

        # Dump edge histogram
        if CONFIG.cfg['data']['Data_Generator']['dump_edge_histogram']:
            with open(CONFIG.cfg['data']['Data_Generator']['dump_edge_histogram_path'], 'wb') as f:
                pickle.dump(hyperedge_sizes_map, f)

        if timing: 
            TIME_edge_hist = time.perf_counter()
            CONFIG.logger.info(f"\t\tEdge histogram time : {TIME_edge_hist - TIME_start}")

        H = lil_matrix((num_v, num_e))
        W = np.zeros(shape=(num_e,), dtype=int)
        edge_orders = np.zeros(shape=(num_e,))

        for e, (hyperedge, count) in enumerate(freq_map.items()):
            for v in hyperedge:
                H[v, e] = 1 # create 0/1 incidence matrix
            edge_orders[e] = len(hyperedge)
            W[e] = int(count)
        H_csr = csr_matrix(H)

        if timing: 
            TIME_inc = time.perf_counter()
            CONFIG.logger.info(f"\t\tH construction time : {TIME_inc - TIME_edge_hist}")

        if self.visualize_genomic_dist:
            process_genomic_dist_histogram(H_csr, title=f'Genomic distance freq for \n{self.file_name}')

        out = PreprocessedData(H_csr, hyperedge_sizes_map, binding_intensity, W, edge_orders)
        out.set_label(self.data_type, self.non_rep)
        return out
    
    def restrict_bin_range(self):
        return self.bin_range is not None and not self.ignore_bin_range
    
    def vertex_offset(self):
        return self.bin_range[0] if self.restrict_bin_range() else 0

    def encode_freq_map(self, df, CONFIG, **kwargs):
        """
        Encode the hypergraph as a map where we record frequencies (weights) as values.
        Input:      A dataframe. 
        Returns:    The map [list of vertex bins] -> [frequency (int)], 
                    number of vertices num_v, 
                    number of edges num_e .
        """
        print(f'bin_range={self.bin_range}')
        restrict_br = self.restrict_bin_range()
        vertex_offset = self.vertex_offset()

        hypergraph = {}
        num_reads = 0
        for row in df.itertuples():
            hyperedge = self.process_hyperedge(row, CONFIG, **kwargs)
            if restrict_br:
                # Restrict to bin range
                hyperedge = frozenset([loci-vertex_offset for loci in hyperedge if loci>self.bin_range[0] and loci<self.bin_range[1]])
            hyperedge = hyperedge_filter(hyperedge, CONFIG) # limit edge size, etc
            if hyperedge is None: continue
            if not (hyperedge in hypergraph.keys()): 
                hypergraph[hyperedge] = 0
            hypergraph[hyperedge] += 1
            num_reads += 1

        if restrict_br:
            num_v = self.bin_range[1] - self.bin_range[0] + 1
        else:
            loci = [v for e in hypergraph.keys() for v in e]
            num_v = int(max(loci) + 1) # Find maximum genomic range in terms of bins
        num_e = int(len(hypergraph))

        CONFIG.logger.info(f'# vertices : {num_v}')
        CONFIG.logger.info(f'# reads    : {num_reads}')
        CONFIG.logger.info(f'# edges    : {num_e}')

        return hypergraph, num_v, num_e
    

def encode_incidence_noise(freq_map_out, CONFIG, only_statistics, test_noise_level, test_noise_func, encode_incidence_func, 
                           binding_intensity=None, timing=False) -> PreprocessedData | list[PreprocessedData]:
    """
    For noise experiments, we have a common way of adding noise across input types
    freq_map_out: tuple of input freq_map, num bins, num edges
    """
    if len(test_noise_level) > 0 and test_noise_func is not None:
        freq_map, nbins, nume = freq_map_out
        out_list = []
        for level in test_noise_level:
            freq_map_noisy = test_noise_func(freq_map, level)
            nume_noisy = sum(list(freq_map_noisy.values()))
            out_list.append(encode_incidence_func(
                (freq_map_noisy, nbins, nume_noisy), 
                CONFIG, 
                only_statistics=only_statistics, 
                binding_intensity=binding_intensity
            ))
        return out_list
    else:
        out = encode_incidence_func(
            freq_map_out, 
            CONFIG, 
            only_statistics=only_statistics, 
            binding_intensity=binding_intensity,
            timing=timing
        )
        return out

    

class Preprocessing_ChIADrop(Preprocessing):

    def cache_fname(self, file_path : str):
        prev = super().cache_fname(file_path)
        prev += f";BI_{CONFIG.cfg['data_adjustment']['apply_binding_intensity']}"
        return prev

    def process_hyperedge(self, hyperedge, CONFIG, **kwargs):
        """
        Input:      raw ChIA-Drop hyperedge string A->B; C->D; ... in bp units (e.g. chr1:34466183-34466883;chr1:34680765-34681694)
        Returns:    A tuple of integers representing the bin location of each vertex. Specifically, 
                    we find hyperedge m(A,B) ; m(C,D) in [bin_size] units where m() is the midpoint. 
                    TODO : Optionally bin based on ChIA-Drop protein binding peak centers. 
        """
        hyperedge = hyperedge.List_of_frag_coord.split(";") 
        def process_vertex(v): 
            ranges = v.split(":")[1].split("-")
            return int((int(ranges[0]) + int(ranges[1])) // (2 * CONFIG.bin_size))
        return frozenset([process_vertex(v) for v in hyperedge])
    
    def process_bedgraph_row(self, row, CONFIG):
        """
        Input:      raw bedgraph string. E.g. chr1	267480	267485	7
        Returns:    Relative bin location and integral intensity
        """
        bin_location = int((int(row.loci1) + int(row.loci2)) // (2 * CONFIG.bin_size)) - self.vertex_offset()
        return bin_location, row.intensity

    
    def preprocess(self, file_name, BI_file_name, proportion:Optional[float]=None, only_statistics=False, test_noise_level=[], test_noise_func=None,
                   rep_string='', non_rep_string='') -> PreprocessedData:
        """
        Function that converts the raw data for a single ChIA-DROP into a binned hypergraph.
        Input:      proportion: If defined, subsample edges at uniform random. 
                    only_statistics: If true, do not find H and only calculate statistics
        Returns:    The binned hypergraph incidence matrix H. None if the file doesn't exist
        """
        self.data_type = "ChIA-Drop GM12878"
        self.non_rep = non_rep_string
        self.file_name = file_name
        load_cache = CONFIG.cfg['data']['ChIA_Drop']['load_cache']
        save_cache = CONFIG.cfg['data']['ChIA_Drop']['save_cache']
        force_cache = CONFIG.cfg['data']['ChIA_Drop']['force_cache']
        cache_path = CONFIG.cfg['data']['ChIA_Drop']['cache_path'].format(fname=self.cache_fname(file_name))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if load_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
            
        elif force_cache and os.path.exists(file_name):
            CONFIG.logger.error(f'Cache not found: {cache_path}')
            exit(1)

        if not os.path.exists(file_name) or not os.path.exists(BI_file_name):
            return None
        df = pd.read_csv(file_name, sep="\t")
        df = df[df['decis1'] == 'PASS'] # Subselect PASS
        if proportion is not None:
            df = df.sample(frac=proportion, random_state=42)

        freq_map_out = self.encode_freq_map(df, CONFIG)

        # Read binding intensity
        _, num_v, _ = freq_map_out
        binding_intensity = np.zeros(shape=(num_v,))
        BI_df = pd.read_csv(BI_file_name, sep="\t", header=None)
        BI_df.columns = ["chr", "loci1", "loci2", "intensity"]
        for row in BI_df.itertuples():
            if row.chr != f'chr{CONFIG.cfg["chr"]}': continue
            loci, intensity = self.process_bedgraph_row(row, CONFIG)
            if 0 <= loci < num_v:
                binding_intensity[loci] = max(binding_intensity[loci], intensity) # use maximum rather than mean

        H_info = encode_incidence_noise(
            freq_map_out=freq_map_out,
            CONFIG=CONFIG,
            only_statistics=only_statistics,
            test_noise_level=test_noise_level,
            test_noise_func=test_noise_func,
            encode_incidence_func=self.encode_incidence,
            binding_intensity=binding_intensity
        )
        if len(test_noise_level) == 0:
            H_info.set_rep(rep_string)
        if save_cache and len(test_noise_level) == 0:
            with open(cache_path, 'wb') as f:
                pickle.dump(H_info, f)
        return H_info
    
# Function to handle multiprocessing over replicates on SPRITE
def _process_rep_SPRITE(rep, df, proportion, encode_freq_map_fn, encode_incidence_fn, 
                        CONFIG, format, rep_type, only_statistics, rep_data, file_name,
                        test_noise_level, test_noise_func, timing):
    if timing:
        TIME_processstart = time.perf_counter()

    CONFIG.logger.info(f"Preprocessing SPRITE {file_name}: rep{rep}")
    if rep_type == "letter":
        CONFIG.logger.info("Filtering rep (letter)")
        df_rep = df[df['id'].apply(lambda id: format.rep_selector(id) == rep)]
    elif rep_type == "finder":
        CONFIG.logger.info("Filtering rep (fastq finder)")
        df_rep = df[df['id'].str.split('.').str[0].isin(rep_data[rep])] # HFFc6 formatting
    else:
        df_rep = df

    if proportion is not None and proportion < 1:
        df_rep = df_rep.sample(frac=proportion, random_state=42)

    if timing:
        TIME_filter_rep = time.perf_counter()
        CONFIG.logger.info(f"\t\tFiltering replicates time : {TIME_filter_rep - TIME_processstart}")
    CONFIG.logger.info("Encoding freq map...")
    freq_map_out = encode_freq_map_fn(df_rep, CONFIG, format=format)

    if timing:
        TIME_freqmap = time.perf_counter()
        CONFIG.logger.info(f"\t\tFinding freq map time : {TIME_freqmap - TIME_filter_rep}")
    res = encode_incidence_noise(
        freq_map_out=freq_map_out,
        CONFIG=CONFIG,
        only_statistics=only_statistics,
        test_noise_level=test_noise_level,
        test_noise_func=test_noise_func,
        encode_incidence_func=encode_incidence_fn,
        timing=timing
    )
    if len(test_noise_level) <= 1:
        res.set_rep(rep)
    return rep, res

class Preprocessing_SPRITE(Preprocessing):
    def process_hyperedge(self, hyperedge, CONFIG, format):
        """
        Input:      raw SPRITE hyperedge string A,C,... in bp units (e.g. 76406674,76408474)
        Returns:    A tuple of integers representing the bin location of each vertex. 
        """
        return format.cluster_selector(hyperedge, CONFIG)
    
    def preprocess_restore_index(self, index, *args, **kwargs):
        out_map = self.preprocess(*args, **kwargs)
        return out_map[index]

        
    def preprocess(self, file_name, proportion:Optional[float]=None, only_statistics=False, 
                   test_noise_level=[], test_noise_func=None, timing=False, non_rep_string='') -> dict[str, PreprocessedData]:
        self.data_type = "SPRITE"
        self.non_rep = non_rep_string
        if timing: TIME_init = time.perf_counter()
        self.file_name = file_name
        # Cache
        load_cache = CONFIG.cfg['data']['SPRITE']['load_cache']
        force_cache = CONFIG.cfg['data']['SPRITE']['force_cache']
        save_cache = CONFIG.cfg['data']['ChIA_Drop']['save_cache']
        cache_path = CONFIG.cfg['data']['SPRITE']['cache_path'].format(fname=self.cache_fname(file_name))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if load_cache and os.path.exists(cache_path):
            CONFIG.logger.info(f'Loading cache: {cache_path}')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        elif force_cache and os.path.exists(file_name):
            CONFIG.logger.error(f'Cache not found: {cache_path}')
            exit(1)
            
        # file name
        cfg_sub = CONFIG.cfg['data']['SPRITE']['data'][file_name]
        fname_map = cfg_sub['fname']

        # load formatting functions
        format:Format = eval(cfg_sub['format'])()

        # Read raw data file
        out_map = dict()
        CONFIG.logger.info(f"rep fname list: {fname_map}")
        for fname_id, fname in fname_map.items():
            CONFIG.logger.info(f"Reading {fname}...")
            if not os.path.exists(fname): 
                CONFIG.logger.error(f"File not found: {fname}")
                return None
            if timing:
                TIME_start = time.perf_counter()
                CONFIG.logger.info(f"\tSPRITE preprocess start time : {TIME_start - TIME_init}")

            df:pd.DataFrame = format.read_csv(fname)

            if timing:
                TIME_readCSV = time.perf_counter()
                CONFIG.logger.info(f"\tRead CSV time : {TIME_readCSV - TIME_start}")

            CONFIG.logger.info(f"Subselect chr...")
            df = format.subselect_chr(df) # Subselect chr1

            if timing:
                TIME_chr = time.perf_counter()
                CONFIG.logger.info(f"\tSubselect chr1 time : {TIME_chr - TIME_readCSV}")

            # Replicate filtering modes
            # optionally use DPM6[...] letter id as rep, otherwise we use single rep
            rep_type = cfg_sub['rep_type']
            rep_data = None
            if rep_type == "letter":
                CONFIG.logger.info("Finding letter replicates...")
                replicates = sorted(list({format.rep_selector(row.id) for row in df.itertuples()}))

            # optionally split file into replicates based on fastq data - needed for HFFc6
            elif rep_type == "finder": 
                CONFIG.logger.info("Using fastq data to find replicates...")
                with open(cfg_sub['rep_finder'], 'rb') as f:
                    rep_data = pickle.load(f)
                replicates = list(rep_data.keys())

            else:
                CONFIG.logger.info("Assigning single replicate...")
                replicates = [fname_id]
            CONFIG.logger.info(f"Found replicates: {replicates}")

            if timing:
                TIME_find_reps = time.perf_counter()
                CONFIG.logger.info(f"\tFinding replicates time : {TIME_find_reps - TIME_chr}")

            # TODO : split up many tasks among a single job to minimize copying DF between processes
            parallel = False
            if parallel:
                results = Parallel(n_jobs=-1)(
                    delayed(_process_rep_SPRITE)(rep, df, proportion, self.encode_freq_map, 
                                                    self.encode_incidence, CONFIG, format, rep_type, 
                                                    only_statistics, rep_data, file_name,
                                                    test_noise_level, test_noise_func, timing)
                    for rep in tqdm(replicates, desc="Processing replicates")
                )
            else:
                results = []
                for rep in tqdm(replicates, desc="Processing replicates"):
                    res = _process_rep_SPRITE(
                        rep, df, proportion, self.encode_freq_map,
                        self.encode_incidence, CONFIG, format, rep_type,
                        only_statistics, rep_data, file_name,
                        test_noise_level, test_noise_func, timing
                    )
                    results.append(res)
            results_dict = dict(results) # rep -> PreprocessedData
            if len(fname_map.items())>1:
                results_dict = {str(k)+','+str(fname_id):v for k,v in results_dict.items()}
            out_map.update(results_dict)

        if save_cache and len(test_noise_level) == 0:
            with open(cache_path, 'wb') as f:
                pickle.dump(out_map, f)
        return out_map



class Preprocessing_DATA_GENERATOR(Preprocessing):
    def __init__(self, clique_adjust, name=""):
        super().__init__(clique_adjust, name)
        self.config_name = 'Data_Generator'
        self.short_name = 'DG'
        self.data_type = "DG"
        self.gen_class = DataGenerator

    def preprocess(self, dg:DataGenerator, group, noise_override=None, test_noise_level=[], test_noise_func=None, rep_string='', non_rep_string=''):
        self.non_rep = non_rep_string
        self.ignore_bin_range = True
        freq_map_out = dg.generate(group, noise_override) # self.freq_map, num_bins, num_e
        res = encode_incidence_noise(
            freq_map_out=freq_map_out,
            CONFIG=CONFIG,
            only_statistics=False,
            test_noise_level=test_noise_level,
            test_noise_func=test_noise_func,
            encode_incidence_func=self.encode_incidence,
        )
        res.set_rep(rep_string)
        return res
    
class Preprocessing_CHIADROP_GENERATOR(Preprocessing_DATA_GENERATOR):
    def __init__(self, clique_adjust, name=""):
        super().__init__(clique_adjust, name)
        self.config_name = 'ChiaDrop_Generator'
        self.short_name = 'CDG'
        self.data_type = "CDG"
        self.gen_class = ChiaDropGenerator

    def preprocess(self, dg:ChiaDropGenerator, group, noise_override=None, test_noise_level=[], test_noise_func=None, rep_string='', non_rep_string=''):
        self.non_rep = non_rep_string
        self.ignore_bin_range = True
        freq_map_out, binding_intensity = dg.generate(group, noise_override)
        res =  encode_incidence_noise(
            freq_map_out=freq_map_out,
            CONFIG=CONFIG,
            only_statistics=False,
            test_noise_level=test_noise_level,
            test_noise_func=test_noise_func,
            encode_incidence_func=self.encode_incidence,
            binding_intensity=binding_intensity,
        )
        res.set_rep(rep_string)
        return res