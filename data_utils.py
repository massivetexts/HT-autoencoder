import tensorflow as tf
import numpy as np
import glob

# TFRecord definitions
def tfrecord_schema(original_dim=202498):
    return {'sparse': tf.SparseFeature(index_key=['token_ids'],
                                                  value_key='counts',
                                                  dtype=tf.int64,
                                                  size=[original_dim]),
                  "volid": tf.FixedLenFeature((), tf.string, default_value=""),
                  #'page_seq': tf.FixedLenFeature((), tf.string, default_value="")
                  }

def sparse_to_dense(sparse):
    ''' This drops the volid and page seq metadata, only returning the dense vector'''
    dense = tf.sparse_to_dense(sparse.indices, sparse.dense_shape, sparse.values)
    return dense


def get_train_dataset(path, batch_size, n_batches, original_dim=202498, 
                      trim_dim=0, shuffle_buffer=30000, idf_path=None,
                      max_path=None, seed=123456,
                     compression="", repeat=10, trim_head=200):
    '''
    Note: Keep path input and shuffling settings consistent between runs.
    
    trim_head removes the first *n* words of the vocabulary.
    '''
    sparse_features = tfrecord_schema(original_dim)
    if compression == "GZIP":
        path = path + ".gz"
    filenames = glob.glob(path)
    
    np.random.seed(seed=seed)
    np.random.shuffle(filenames)
    
    dataset = tf.data.TFRecordDataset(filenames, compression_type=compression, num_parallel_reads=2)
    dataset = dataset.shuffle(shuffle_buffer, seed=12345, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x: tf.parse_example(x, features=sparse_features)['sparse'])

    trim_end = (trim_dim+trim_head) if trim_dim else original_dim
    dataset = dataset.map(lambda x: tf.sparse_slice(x, [0,trim_head], [batch_size, trim_dim if trim_dim else original_dim]) )
        
    if idf_path:
        idf = np.load(idf_path)[trim_head:trim_end]
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32) * idf)

    if max_path:
        maxarr = np.load(max_path)[trim_head:trim_end]
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32) / maxarr)
        
    dataset = dataset.map(sparse_to_dense)
    dataset = dataset.repeat(repeat)
    
    return dataset

def get_validation_dataset(path, n_pages, original_dim=202498, trim_dim=0,
                           shuffle_buffer=30000, trim_head=200, idf_path=None, seed=123456,
                           max_path=None, compression=""):
    '''
    Note: Keep path input and shuffling settings consistent between runs.
    '''
    
    sparse_features = tfrecord_schema(original_dim)
    if compression == "GZIP":
        path = path + ".gz"
    filenames = glob.glob(path)
    
    np.random.seed(seed=seed)
    np.random.shuffle(filenames)
    
    dataset = tf.data.TFRecordDataset(filenames, compression_type=compression).shuffle(shuffle_buffer, seed=30303)
    dataset = dataset.batch(n_pages).take(1)
    dataset = dataset.map(lambda x: tf.parse_example(x, features=sparse_features)['sparse'])
    
    trim_end = (trim_dim+trim_head) if trim_dim else original_dim
    dataset = dataset.map(lambda x: tf.sparse_slice(x, [0,trim_head], [n_pages, trim_dim if trim_dim else original_dim]) )

    if idf_path:
        idf = np.load(idf_path)[trim_head:trim_end]
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32) * idf)
    
    if max_path:
        maxarr = np.load(max_path)[trim_head:trim_end]
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32) / maxarr)
    
    return dataset