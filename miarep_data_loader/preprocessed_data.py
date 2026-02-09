from scipy.sparse import csr_matrix

class PreprocessedData:

    def __init__(self, H=csr_matrix([]), hyperedge_sizes_map=None, binding_intensity=None, W=None, edge_orders=None):
        self.H = H
        self.hyperedge_sizes_map = hyperedge_sizes_map
        self.binding_intensity = binding_intensity
        self.W = W
        self.edge_orders = edge_orders
        self.A = None

        self.data_type = ""
        self.non_rep = ""
        self.rep = ""

        self.preprocessing_object = None
        self.preprocessing_args = ()
        self.preprocessing_kwargs = dict()
        self.preprocessing_index = None

    def set_label(self, data_type, non_rep):
        self.data_type = data_type
        self.non_rep = non_rep

    def set_rep(self, rep):
        self.rep = rep

    def get_short_label(self):
        return f"{self.data_type} [{self.non_rep}] [rep{self.rep}]"
    
    def get_data(self):
        return self.H, self.W, self.edge_orders, self.binding_intensity
    
    def restore_preprocessing(self, CONFIG):
        """
       Re-preprocess the memory heavy data from disk. 
        """
        delay_preprocessing = CONFIG.cfg['delay_preprocessing']
        if not delay_preprocessing: return

        assert self.preprocessing_object is not None
        result = self.preprocessing_object.preprocess_restore_index(
            self.preprocessing_index,
            *self.preprocessing_args, 
            **self.preprocessing_kwargs
        )
        self.H, self.W, self.edge_orders, self.binding_intensity = result.H, result.W, result.edge_orders, result.binding_intensity
    
    def free_data(self, CONFIG):
        """
        Clears all memory-heavy data to be restored later using restore_preprocessing
        """
        delay_preprocessing = CONFIG.cfg['delay_preprocessing']
        if not delay_preprocessing: return

        self.H, self.W, self.edge_orders, self.binding_intensity = None, None, None, None
    
    def delay_preprocessing(self, preprocessing, args, kwargs, index=None):
        """
        Call after creating this object to setup delayed loading of data
        """
        self.preprocessing_object = preprocessing
        self.preprocessing_args = args
        self.preprocessing_kwargs = kwargs
        self.preprocessing_index = index