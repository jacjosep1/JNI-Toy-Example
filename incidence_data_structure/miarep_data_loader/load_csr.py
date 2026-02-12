
from .preprocess import *
from .visualization import *

def pack_args(*args, **kwargs):
    return args, kwargs

def load_csr() -> dict[str, dict[str, PreprocessedData]]:
    CONFIG.initialize([])
    proportion = float(CONFIG.cfg['data']['proportion'])
    chr=CONFIG.cfg['chr']

    # SPRITE
    As : dict[str, dict[str, PreprocessedData]] = {}
    if CONFIG.cfg['data']['SPRITE']['enabled']:
        SPRITE_data_list:dict = CONFIG.cfg['data']['SPRITE']['data']
        for file_name in SPRITE_data_list.keys():
            if not SPRITE_data_list[file_name]['enabled']: continue
            preprocessing = Preprocessing_SPRITE(False)
            nonrep_str = f"SPRITE_{file_name}"
            CONFIG.logger.info(f"Preprocessing {nonrep_str}")
            preprocess_args, preprocess_kwargs = pack_args(file_name, proportion, non_rep_string=file_name)
            Hs : dict[str, PreprocessedData] = preprocessing.preprocess(*preprocess_args, **preprocess_kwargs)

            if Hs is not None:
                As[nonrep_str] = {}
                for rep, result in Hs.items():
                    result.delay_preprocessing(preprocessing, preprocess_args, preprocess_kwargs, index=rep)
                    As[nonrep_str][rep] = result

    return As




    

