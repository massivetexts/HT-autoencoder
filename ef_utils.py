import pandas as pd
import numpy as np

def load_tokenref(path, trim_head=200):
    return pd.read_csv(path, names=['token'])[trim_head:].reset_index(drop=True)

def tokenref2dict(tokenref):
    return tokenref.reset_index().set_index('token').to_dict()['index']

def vol2vec(vol, token_dict, shape=None, idf=None, bypage=False, trim_head=200):
    '''
    Convert an EF volume to an array of counts, using ids encoded
    in token_dict as 'word': index. Counts multiplied by
    idf array if supplied.
    '''
    
    volids = []
    if bypage:
        tl = vol.tokenlist(pos=False, case=False)
        groups = tl.groupby(level='page')
    
        all_arr = []
        for page_num, page_tl in groups:
            page_arr = tl_to_arr(page_tl, token_dict, shape, idf)
            if np.sum(page_arr) == 0:
                continue
            all_arr.append(page_arr)
            volids.append(vol.id + str(page_num))
        arr = np.vstack(all_arr)
        
    else:
        tl = vol.tokenlist(pos=False, case=False, pages=False)
        arr = tl_to_arr(tl, token_dict, shape, idf)
        volids.append(vol.id)

    return volids, arr

def tl_to_arr(tl, token_dict, shape=None, idf=None):
    if not shape:
        shape = len(token_dict)
    arr = np.zeros(shape)
        
    tl = (tl.reset_index()[['lowercase', 'count']]
            .rename(columns={'lowercase':'token'})
        )
    for i, row in tl.iterrows():
        if row.token in token_dict:
            token_i = token_dict[row.token]

            arr[token_i] = row['count']
                
    if idf is not None:
        arr = arr * idf    
                
    return arr