import tensorflow as tf
import pandas as pd
from htrc_features import FeatureReader, utils
import itertools
import glob
from ef_utils import *

ef_root = "data/ef-files/comedy/"
ef_file_paths = glob.glob(ef_root+"/*.bz2")
ef_files = FeatureReader(paths=list(ef_file_paths))

token_ref = load_tokenref('eng-vocab-1.txt.bz2', trim_head=0)

volumes = ef_files.volumes()

i = 0
writer = tf.python_io.TFRecordWriter('data/literature/tfrecords/lit-%d.tfrecord' % int(i/100))

for vol in volumes:
    i += 1
    if i % 100 == 0:
        writer.close()
        writer = tf.python_io.TFRecordWriter('data/literature/tfrecords/lit-%d.tfrecord' % int(i/100))
        
    print(vol.id)
    pages_en = [p for p in vol.pages() if {'en': '1.00'} in p.languages]
    for page in pages_en:
        page_body_tokens = page.tokenlist(section='body', case=False, pos=False)
        token_counts = page_body_tokens.reset_index().drop(['page', 'section'], axis=1)
        token_counts.rename(index=str, columns={'lowercase': 'token'}, inplace=True)
        indexed_token_counts = token_counts.merge(token_index, how='inner', on='token')
            
        token_ids = tf.train.Feature(int64_list=tf.train.Int64List(
                value=list(indexed_token_counts['index'])
            ))

        counts = tf.train.Feature(int64_list=tf.train.Int64List(
                value=list(indexed_token_counts['count'])
            ))
        volid = tf.train.Feature(bytes_list=tf.train.BytesList(value=[vol.id.encode('utf-8')]))
        page_seq = tf.train.Feature(bytes_list=tf.train.BytesList(value=[page.seq.encode('utf-8')]))

        example = tf.train.Example(
            features=tf.train.Features(feature=
                                        {
                                            'page_seq': page_seq,
                                            'volid': volid,
                                            'token_ids': token_ids,
                                            'counts': counts
                                        })
            )

        writer.write(example.SerializeToString())
            
writer.close()