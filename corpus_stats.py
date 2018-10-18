import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import hashlib
from data_utils import sparse_to_dense, tfrecord_schema

def count_docs(filenames):
    N = 0

    for fn in filenames:
        for record in tf.python_io.tf_record_iterator(fn):
            N += 1

    return N

# Get Document Frequency
def document_frequency(dataset):
    def to_dense_ones(sparse):
        dense = tf.sparse_to_dense(sparse.indices, sparse.dense_shape, sparse.values)
        zero = tf.constant(0, dtype=tf.int64)
        where = tf.not_equal(dense, zero)
        return tf.cast(where, 'int32')

    dfdataset = dataset.map(to_dense_ones)
    dfdataset = dfdataset.map(lambda x: tf.reduce_sum(x, axis=0))

    dfdata_iter = dfdataset.make_one_shot_iterator()
    dfdata = dfdata_iter.get_next()

    counter = np.zeros(202498,)
    i = 0

    while True:
        try:
            print(i, end=', ')
            # Sum of 1000 pages
            counter += dfdata.eval()
            i += 1
        except:
            break
            
    return counter

def max_counts(dataset):
    maxdataset = dataset.map(lambda x: tf.sparse_reduce_max_sparse(x, axis=0))
    maxdataset = maxdataset.batch(100)
    maxdataset = maxdataset.map(lambda x: tf.sparse_reduce_max(x, axis=0))

    maxdata_iter = maxdataset.make_one_shot_iterator()
    maxdata = maxdata_iter.get_next()

    counter = np.zeros(202498,)
    i = 0

    while True:
        try:
            print(i, end=', ')
            # Sum of 1000 pages
            maxbatch = maxdata.eval()
            counter = np.vstack([counter, maxbatch]).max(axis=0)
            i += 1
        except:
            break
            
    return counter

def main(filenames, outdir):
    with tf.Session() as sess:
        original_dim = 202498
        name = hashlib.md5(str(filenames).encode('utf-8')).hexdigest()
        sparse_features = tfrecord_schema(original_dim)

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
        # Batch 1000 at a time
        dataset = dataset.batch(1000)
        dataset = dataset.map(lambda x: tf.parse_example(x, features=sparse_features)['sparse'])

        print('counting number of documents')
        N = count_docs(filenames)
        print('total number of documents (pages) is', N)

        print("Counting document frequencies")
        df = document_frequency(dataset)
        print("Done counting document frequencies. Saving...")
        np.save(outdir + '/document-frequency-N%s-%s.npy' % (N, name), df)

        print("Counting max(tf)")
        maxes = max_counts(dataset)
        print("Done counting maxes. Saving...")
        np.save(outdir + '/maxes-N%s-%s.npy' % (N, name), maxes)        
        
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate corpus statistics')
    parser.add_argument('--out-dir', '-o', type=str, default='data/',
                        help='Location to save statistics')
    parser.add_argument('datafiles', type=str, nargs='+',
                        help='Location of TFRecord files.') 

    args = parser.parse_args()
    
    main(args.datafiles, args.out_dir)