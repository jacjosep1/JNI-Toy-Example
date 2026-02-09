
import pickle
import random
import numpy as np
from scipy.sparse import csr_matrix, hstack
from itertools import combinations

from config import *
# high_order_size = int(cfg['data']['Data_Generator']['high_order_size'])

# Util functions

def weighted_random(map):
    """
    Helper function to generate random key from a map [keys] -> [weights].
    """
    return random.choices(list(map.keys()), weights=list(map.values()), k=1)[0]

def powerset(lst):
    result = []
    for r in range(2, len(lst)+1):
        result.extend(combinations(lst, r))
    return result

# Random noise functions

def random_gaussian(edge, num_bins, section='Data_Generator'):
    """
    Use this function to variate the position of loops, stripes, etc...
    """
    sigma = float(CONFIG.cfg['data'][section]['nonrep_settings']['random_sigma'])
    order = len(edge)
    if sigma > 0 and order > 0:
        Z = np.random.multivariate_normal(
            mean=np.zeros((order,)), 
            cov=np.eye(N=order)*sigma
        )
        Z = Z.round().astype(int)
    else:
        Z = np.zeros((order,))
    return np.clip(edge + Z, 0, num_bins-1)
    
# Main data generation

class DataGenerator:

    class GroupParameters:
        """
        Class to store generated parameters per non-replicate group. Replicates belong to the same group. 
        """
        @staticmethod
        def gen_pois(cfg):
            num_bins = int(cfg['num_bins'])
            num_poi_range = [int(x) for x in cfg['num_poi']]

            num_poi = random.randint(*num_poi_range)
            pois = random.sample(range(1, num_bins-1), k=num_poi)
            
            def add_poi(p):
                if p not in pois: pois.append(p)
            add_poi(0)
            add_poi(num_bins-1)
            pois.sort()
            return pois

        @classmethod
        def static_init(cls, cfg):
            cls.cfg = cfg
            randomize_poi = cls.cfg['nonrep_settings']['randomize_POI']
            # Persists across groups
            if not randomize_poi:
                cls.pois = cls.gen_pois(cls.cfg)

        def called_generate(self):
            self.current_call_count += 1

        def __init__(self, group, num_groups, dg, cfg):
            self.current_call_count = 0
            self.cfg = cfg
            high_order_size = int(self.cfg['high_order_size'])
            randomize_poi = self.cfg['nonrep_settings']['randomize_POI']
            random_TAD_prob =        float(self.cfg['nonrep_settings']['random_TAD_prob'])
            random_loop_prob =       float(self.cfg['nonrep_settings']['random_loop_prob'])
            random_stripe_prob =     float(self.cfg['nonrep_settings']['random_stripe_prob'])
            random_loop_TAD_prob =   float(self.cfg['nonrep_settings']['random_loop_TAD_prob'])
            random_stripe_TAD_prob = float(self.cfg['nonrep_settings']['random_stripe_TAD_prob'])
            random_dense_prob =      float(self.cfg['nonrep_settings']['random_dense_prob'])

            # Different across groups
            if randomize_poi:
                self.pois = self.gen_pois(self.cfg)
            
            # Structure gen
            structure_dist = {
                dg.generate_TAD :           random_TAD_prob,
                dg.generate_loop :          random_loop_prob,
                dg.generate_stripe :        random_stripe_prob,
                dg.generate_loop_TAD :      random_loop_TAD_prob,
                dg.generate_stripe_TAD :    random_stripe_TAD_prob,
                dg.generate_dense :         random_dense_prob,
            }
            self.structures = []
            self.large_edges = [] # Only use for generate_dense
            for i, (loci1, loci2) in enumerate(zip(self.pois, self.pois[1:])):
                self.structures.append(weighted_random(structure_dist))
                if random_dense_prob > 0.0:
                    sample_space = range(loci1, loci2+1)
                    edge = random.sample(sample_space, k=min(high_order_size, len(sample_space)))
                    self.large_edges.append(edge)

    def __init__(self, num_groups, cfg_subsection, param_cls=GroupParameters):
        self.cfg = cfg_subsection
        noise_level = float(self.cfg['noise_level'])
        with open(self.cfg['dump_edge_histogram_path'], 'rb') as f:
            self.edge_histogram = pickle.load(f)     
        param_cls.static_init(self.cfg)
        self.params = [param_cls(group, num_groups, self, self.cfg) for group in range(num_groups)]  
        self.noise_level = noise_level
        self.TAD_density = float(self.cfg['TAD_density'])


    def generate(self, group, noise_override=None):
        """
        Generates a replicate in a particular replicate group.
        Input       Group: different groups are non-replicates. Same group should be replicates. 
                    noise_override: If not none, override the noise level in [0, 1]
        Returns:    Tuple of (freq_map, num_v, num_e)
        """
        num_bins = int(self.cfg['num_bins'])
        high_order_noise = float(self.cfg['rep_settings']['high_order_noise'])

        if noise_override is not None:
            self.noise_level = noise_override
        self.freq_map = {}
        params:DataGenerator.GroupParameters = self.params[group]

        for i, (loci1, loci2) in enumerate(zip(params.pois, params.pois[1:])):
            params.structures[i](loci1=loci1, loci2=loci2, struct_index=i, params=params)

        # Additive noise
        num_e = sum(self.freq_map.values())
        num_noise_edges = int(num_e * self.noise_level)
        self.add_random_region(0, num_bins-1, num_noise_edges)

        # High order noise
        if high_order_noise > 0.0:
            num_e = sum(self.freq_map.values())
            N = int(num_e * high_order_noise)
            RNG = random.Random(42 + params.current_call_count)
            self.add_unsupported_high_order(0, num_bins-1, N, RNG=RNG)

        self.shuffle_edges()
        self.randomize_seq_depth()
        num_e = len(self.freq_map.values())
        params.called_generate()
        CONFIG.logger.info(f"Generated with num_v={num_bins} num_e={num_e}")
        return self.freq_map, num_bins, num_e
    

    def randomize_seq_depth(self):
        variation = float(self.cfg['rep_settings']['seq_depth_variation'])
        if variation > 0:
            seq_depth_adjustment = np.random.uniform(1-variation, 1) # pick a sequence depth
            for edge, freq in list(self.freq_map.items()):
                new_freq = np.random.binomial(n=freq, p=seq_depth_adjustment)
                if new_freq > 0:
                    self.freq_map[edge] = new_freq
                else:
                    del self.freq_map[edge]
    

    def shuffle_edges(self):
        items = list(self.freq_map.items())
        random.shuffle(items)
        self.freq_map = dict(items)
    

    def gen_edge_size(self, bound, high_order=False):
        high_order_size = int(self.cfg['high_order_size'])
        if high_order:
            return high_order_size
        hist = {2:1.}
        hist.update({
            e:f
            for e,f in self.edge_histogram.items() 
            if e <= bound and e >= 2
        })
        return weighted_random(hist)
    

    def generate_TAD(self, loci1, loci2, **kwargs):
        num_edges = int(self.TAD_density * (loci2 - loci1) ** 1.5 * (1-self.noise_level))
        self.add_random_region(loci1, loci2, num_edges)


    def generate_loop(self, loci1, loci2, **kwargs):
        num_bins = int(self.cfg['num_bins'])

        num_edges = int(self.TAD_density * (loci2 - loci1) * (1-self.noise_level))
        edge = np.array([loci1, loci2])
        for _ in range(num_edges):
            self.add_edge(random_gaussian(edge, num_bins))


    def generate_stripe(self, loci1, loci2, **kwargs):
        num_bins = int(self.cfg['num_bins'])

        num_edges = int(self.TAD_density * (loci2 - loci1) ** 1.4 * (1-self.noise_level))
        for _ in range(num_edges):
            edge = np.array([np.random.randint(loci1, loci2), loci1])
            self.add_edge(random_gaussian(edge, num_bins))

    def generate_loop_TAD(self, loci1, loci2, **kwargs):
        """
        Generates a loop and a TAD in the same region.
        """
        self.generate_loop(loci1, loci2, **kwargs)
        self.generate_TAD(loci1, loci2, **kwargs)

    def generate_stripe_TAD(self, loci1, loci2, **kwargs):
        """
        Generates a stripe and a TAD in the same region.
        """
        self.generate_stripe(loci1, loci2, **kwargs)
        self.generate_TAD(loci1, loci2, **kwargs)


    def generate_dense(self, loci1, loci2, struct_index, params : GroupParameters):
        num_edges = int(self.TAD_density * (loci2 - loci1) * (1-self.noise_level))
        edge = params.large_edges[struct_index]
        P = powerset(edge)
        P = random.sample(P, k=min(num_edges, len(P)))
        for subset in P:
            self.add_edge(subset)
            

    def add_edge(self, edge):
        if len(edge) > 1:
            edge = frozenset(edge)
            self.freq_map[edge] = self.freq_map.get(edge, 0) + 1


    def add_random_region(self, loci1, loci2, num_edges):
        sample_space = range(loci1, loci2+1)
        for _ in range(num_edges):
            k = self.gen_edge_size(loci2 - loci1)
            edge = random.sample(sample_space, k=k)
            self.add_edge(edge)


    def add_unsupported_high_order(self, loci1, loci2, num_edges, RNG:random.Random):
        sample_space = range(loci1, loci2+1)
        k = self.gen_edge_size(loci2 - loci1, high_order=True)
        groups = int(self.cfg['rep_settings']['high_order_noise_groups'])

        edge_map = [RNG.sample(sample_space, k=k) for _ in range(groups)]
        for i in range(num_edges):
            self.add_edge(edge_map[i % groups])
    
