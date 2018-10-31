import pandas as pd
import numpy as np

def load_tokenref(path, trim_head=200):
    return pd.read_csv(path, names=['token'])[trim_head:].reset_index(drop=True)

def tokenref2dict(tokenref):
    return tokenref.reset_index().set_index('token').to_dict()['index']

def vol2vec(vol, token_dict, shape=None, idf=None, trim_head=200):
    '''
    Convert an EF volume to an array of counts, using ids encoded
    in token_dict as 'word': index. Counts multiplied by
    idf array if supplied.
    '''
    if not shape:
        shape = len(token_dict)
    arr = np.zeros(shape)

    tl = (vol.tokenlist(pos=False, case=False, pages=False).reset_index()[['lowercase', 'count']]
         .rename(columns={'lowercase':'token'})
     )

    for i, row in tl.iterrows():
        if row.token in token_dict:
            token_i = token_dict[row.token]

            arr[token_i] = row['count']
    
    if idf is not None:
        arr = arr * idf

    return vol.id, arr
