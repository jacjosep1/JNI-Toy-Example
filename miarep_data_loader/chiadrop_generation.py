
import random
import numpy as np
from scipy.sparse import csr_matrix, hstack

from config import *
from .data_generation import DataGenerator, random_gaussian, weighted_random

def gaussian(xs, mu=0, sigma=1., str=1.):
    return np.exp(-((xs - mu)**2) / (2 * sigma**2)) * str

class ChiaDropGenerator(DataGenerator):

    class ChiaDropGroupParameters(DataGenerator.GroupParameters):
        def __init__(self, group, num_groups, dg, cfg):
            self.pois = self.gen_pois(cfg)

            # Uniform random peak size
            str_min, str_max = tuple([float(e) for e in cfg['nonrep_settings']['peak_size']])
            self.pois_base_strength = [random.random()*(str_max-str_min)+str_min for _ in range(len(self.pois))]
            convergent_prop = float(cfg['nonrep_settings']['convergent_peaks_proportion'])
            self.pois_convergent   = [(random.random() < convergent_prop) for _ in range(len(self.pois))]


    def __init__(self, num_groups, cfg_subsection, param_cls=ChiaDropGroupParameters):
        super().__init__(num_groups, cfg_subsection, param_cls)


    def generate(self, group, noise_override=None):
        section = 'ChiaDrop_Generator'
        num_bins = int(self.cfg['num_bins'])
        if noise_override is not None:
            self.noise_level = noise_override
        self.freq_map = {}
        params:ChiaDropGenerator.ChiaDropGroupParameters = self.params[group]

        # Generate peak strengths with random perturbations
        sigma = float(self.cfg['rep_settings']['real_peaks']['str_sigma'])
        strengths = [np.random.normal(base_str, sigma) for base_str in params.pois_base_strength]
        num_real_peaks = len(strengths)
        
        # Center gaussian at each peak with sigma based on real data. 
        sigma = float(self.cfg['rep_settings']['real_peaks']['width'])
        intensities = np.zeros(shape=(num_bins,))
        xs = np.arange(num_bins)
        for loc, str in zip(params.pois, strengths):
            intensities += gaussian(xs, loc, sigma, str)

        # Generate uncorrelated small noisy peaks on binding strength.
        str_mean = float(self.cfg['rep_settings']['random_peaks']['str'])
        str_sigma = float(self.cfg['rep_settings']['random_peaks']['str_sigma'])
        width_sigma = float(self.cfg['rep_settings']['random_peaks']['width'])
        mean_dist = float(self.cfg['rep_settings']['random_peaks']['mean_dist'])

        num_to_generate = int(num_bins / mean_dist)
        for _ in range(num_to_generate):
            str = np.random.normal(str_mean, str_sigma)
            loc = np.random.randint(num_bins)
            intensities += gaussian(xs, loc, width_sigma, str)

        # Generate uniform noise on binding strength. 
        background_mean = float(self.cfg['rep_settings']['background_gaussian_noise']['mean'])
        backgroud_sigma = float(self.cfg['rep_settings']['background_gaussian_noise']['sigma'])
        intensities += np.random.normal(loc=background_mean, scale=backgroud_sigma, size=(num_bins,))

        # Gen edges between actual peaks, only generate up to certain distance
        max_dist = int(self.cfg['rep_settings']['real_edges']['max_dist'])
        density = float(self.cfg['rep_settings']['real_edges']['density'])
        proportion_spanning = float(self.cfg['rep_settings']['real_edges']['proportion_spanning'])

        total_incidences = density * num_real_peaks * proportion_spanning
        while total_incidences > 0:
            edge_order = self.gen_edge_size(100, False)
            # select starting real peak (center)
            poi_sample_space = []
            center_index = random.randint(0, len(params.pois)-1)
            center_poi = params.pois[center_index]

            # find neighbor real peaks < max_dist/2 away
            for poi, convergent in zip(params.pois, params.pois_convergent):
                if np.abs(poi - center_poi)*2 < max_dist:
                    poi_sample_space.append((poi, convergent))

            # select edge_order-1 of them
            if len(poi_sample_space) > 0:
                # Spanning edge
                added_edge = dict(random.sample(poi_sample_space, k=min(edge_order, len(poi_sample_space)-1)))
                added_edge_pois = list(added_edge.keys())
                added_edge_pois.sort()
                self.add_edge(random_gaussian(added_edge_pois, num_bins, section))
                total_incidences -= len(added_edge_pois)

                # Filling edge
                for i, (loci_L, loci_R) in enumerate(zip(added_edge_pois, added_edge_pois[1:])):
                    convergent = added_edge[loci_L]
                    if not convergent: continue
                    num_to_generate = int(.5 / proportion_spanning)
                    def gen_filling(fixed : int):
                        for _ in range(num_to_generate):
                            filling_edge_order = self.gen_edge_size(np.abs(loci_L - loci_R), False)
                            edge = [fixed]+[random.randint(loci_L, loci_R) for _ in range(filling_edge_order-1)]
                            self.add_edge(random_gaussian(edge, num_bins, section))
                    gen_filling(loci_L) # Left aligned edges
                    gen_filling(loci_R) # Right aligned edges

        # Generate uncorrelated noisy edges. 
        num_e = sum(self.freq_map.values())
        num_noise_edges = int(num_e * self.noise_level)
        self.add_random_region(0, num_bins-1, num_noise_edges)

        self.shuffle_edges()
        self.randomize_seq_depth()
        num_e = sum(self.freq_map.values())
        CONFIG.logger.info(f'Generated ChIA-Drop with |E|={num_e}')
        return (self.freq_map, num_bins, num_e), np.abs(intensities)

