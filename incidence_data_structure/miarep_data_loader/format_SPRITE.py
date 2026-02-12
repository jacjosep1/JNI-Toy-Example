
from abc import abstractmethod
import pandas as pd

from .config import *

# This file contains all special string parsing needed for different SPRITE .clusters formats for our data. 

class Format:
    # Abstract class to parent different SPRITE formatting
    @abstractmethod
    def rep_selector(self, id)->str: # Filters id letter in row (should be replicates)
        pass
    @abstractmethod
    def read_csv(self, fname)->pd.DataFrame:
        pass
    @abstractmethod
    def subselect_chr(self, df)->pd.DataFrame: # returns new df that filters out chr
        pass
    @abstractmethod
    def cluster_selector(self, hyperedge, CONFIG)->frozenset[int]: # returns list of clusters
        pass

class Format_GM12878(Format):
    # Example: chr10	SPRITE-100000000-DPM6A10.NYBot10_Stg.Odd2Bo10.Even2Bo16.Odd2Bo10.10-AAAAAA-K00384-HA-1-0	76406674,76408474
    def rep_selector(self, id):
        return id.split(".")[0].split("-")[2][4]
    
    def read_csv(self, fname):
        df = pd.read_csv(fname, sep='\t', header=None)
        df.columns = ["chr", "id", "coords"]
        return df
    
    def subselect_chr(self, df): 
        return df[df['chr'] == "chr"+CONFIG.cfg['chr']]
    
    def cluster_selector(self, hyperedge, CONFIG):
        hyperedge = hyperedge.coords.split(",") 
        return frozenset([int(int(v) // CONFIG.bin_size) for v in hyperedge])
    
class Format_HFFc6(Format):
    # Example: DPM5bot10_B2.NYBot37_Stg.Odd2Bo64.Even2Bo91.Odd2Bo96	chr4:5805500	chr4:6304500
    def rep_selector(self, id):
        return id.split("_")[1][0]
    def read_csv(self, fname):
        with open(fname, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
        #lines = ['DPM5bot10_B2.NYBot37_Stg.Odd2Bo64.Even2Bo91.Odd2Bo96	chr4:5805500	chr4:6304500']
        df = pd.DataFrame({'raw': lines})
        df['id'] = df['raw'].str.split('\t').str[0]
        return df
    
    def subselect_chr(self, df): 
        target = "chr" + CONFIG.cfg['chr']

        parts = df['raw'].str.split('\t', expand=True)
        chr_cols = parts.iloc[:, 1:].apply(lambda c: c.str.split(':').str[0])

        mask = (chr_cols == target).all(axis=1)

        return df[mask]
    
    def cluster_selector(self, hyperedge, CONFIG):
        hyperedge = list(hyperedge.raw.split('\t')[1:])
        return frozenset([int(int(v.split(':')[1]) // CONFIG.bin_size) for v in hyperedge])
    
class Format_H1_hESC(Format_HFFc6):
    # Example: DPM5bot11_C2_Diff.YbotE39.Odd2Bo11.Even2Bo63.Odd2Bo93.4dn-10	DNA[+]_chr20:39924182-39924321	DNA[+]_chr20:39924182-39924323
    def subselect_chr(self, df):
        target = "chr" + CONFIG.cfg['chr']

        parts = df['raw'].str.split('\t', expand=True)

        chr_cols = (
            parts.iloc[:, 1:]
            .apply(lambda c: c.str.split('_').str[1].str.split(':').str[0])
        )

        mask = (chr_cols == target).all(axis=1)

        return df[mask]
    
    def cluster_selector(self, hyperedge, CONFIG):
        hyperedge = list(hyperedge.raw.split('\t')[1:]) # hyperedge is a df row
        return frozenset([int(
            (
                int(v.split(':')[1].split('-')[0]) +    # First location
                int(v.split(':')[1].split('-')[1])      # Second location
            ) // (2 * CONFIG.bin_size) # midpoint
        ) for v in hyperedge])
    
