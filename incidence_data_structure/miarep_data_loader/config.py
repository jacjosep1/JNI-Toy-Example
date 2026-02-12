
import yaml
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import ast
import logging
import pickle

def deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if (
            k in dst
            and isinstance(dst[k], dict)
            and isinstance(v, dict)
        ):
            deep_update(dst[k], v)
        else:
            dst[k] = v

class RepMethod:
    def __init__(self, fn, on_exp):
        self.fn = fn
        self.on_exp = on_exp
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

class Config:
    """
    Object for storing global information specified by the main config file. 
    """

    
    def divide_by_edge_order(self):
        exp = self.cfg['data_adjustment']['expansion']
        return exp != 'Baseline'
    

    def get_genomic_range_string(self):
        if self.gr_enabled:
            GR1 = f'{self.genomic_range[0]/1e6}Mb'
            GR2 = f'{self.genomic_range[1]/1e6}Mb'
            GR = f'{GR1}-{GR2}'
        else:
            GR = 'entire'
        return f'chr{self.cfg["chr"]}:{GR}'

    def save_results(self, results):
        if self.run_dir is None: return
        save_dir = os.path.join(self.run_dir, f'results_{self.cfg["name"]}.pkl')
        self.logger.info(f"Saving results for {self.cfg['name']}...")
        with open(save_dir, 'wb') as f:
            pickle.dump(results, f)
        

    def initialize(self, args):
        self.cfg = dict()
        cfg_names = { # Maps filename to top-level category in the final config dict
            'main': [],
            'data': ['data'],
            'data_gen': ['data'],
            'rep_methods': ['rep_methods'],
            'exp_methods': ['expansion_methods'],
            'noise_methods': ['noise_methods'],
            'learning': [],
        }
        for fname, dict_path in cfg_names.items():
            with open(f'miarep_data_loader/config/{fname}.yaml', 'r') as f:
                sub_cfg = yaml.safe_load(f)
            d:dict = self.cfg
            for k in dict_path:
                d = d.setdefault(k, {})
            d.update(sub_cfg)

        # Parse cfg modifications in arguments
        for arg in args:
            if "=" not in arg: continue
            key_path, value = arg.split("=", 1)
            keys = key_path.split(".")
            d = self.cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            try:
                v = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                v = value
            d[keys[-1]] = v

        # Apply baseline overrides
        baseline_name : str = self.cfg['baseline']
        if baseline_name != 'none':
            with open(f'config/baseline/{baseline_name}.yaml', 'r') as f:
                updated_cfg = yaml.safe_load(f)
            deep_update(d, updated_cfg)

        self.bin_size = int(float(self.cfg['bin_size'])) # accept scientific notation
        self.gr_enabled = self.cfg['genomic_range_enabled']
        self.genomic_range=[int(float(e)) for e in self.cfg['genomic_range']] if self.gr_enabled else None
        self.EDGE_SIZE_FILTER = [int(e) for e in self.cfg['data_adjustment']['hyperedge_range']]
        self.do_logging = self.cfg['do_logging']
        self.out_dir = 'out'
        self.run_dir = None
        if self.do_logging:
            self.run_dir = os.path.join(self.out_dir, f"runs/{self.cfg['name']}_{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}")
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, 'config.yaml'), "w") as f:
                yaml.safe_dump(self.cfg, f, sort_keys=False)
            #shutil.copytree('config', os.path.join(self.run_dir, 'config'))

            file_handler = logging.FileHandler(os.path.join(self.run_dir, 'log.txt'))
            console_handler = logging.StreamHandler()
            file_handler.setLevel(logging.INFO)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    def get_state(self):
        return self.__dict__.copy()
    def update_state(self, state : dict):
        self.__dict__.update(state)

CONFIG=Config()