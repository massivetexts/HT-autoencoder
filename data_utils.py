import tensorflow as tf
import numpy as np
import glob

# TFRecord definitions
def tfrecord_schema(original_dim):
    return {'sparse': tf.SparseFeature(index_key=['token_ids'],
                                                  value_key='counts',
                                                  dtype=tf.int64,
                                                  size=[original_dim]),
                  "volid": tf.FixedLenFeature((), tf.string, default_value=""),
                  'page_seq': tf.FixedLenFeature((), tf.int64)
                  }

def sparse_to_dense(sparse):
    ''' This drops the volid and page seq metadata, only returning the dense vector'''
    dense = tf.sparse_to_dense(sparse.indices, sparse.dense_shape, sparse.values)
    return dense


def get_train_dataset(path, batch_size, n_batches, original_dim=202498, 
                      trim_dim=0, shuffle_buffer=20000, idf_path=None):
    '''
    Note: Keep path input and shuffling settings consistent between runs.
    '''
    sparse_features = tfrecord_schema(original_dim)
    filenames = glob.glob(path)
    
    np.random.seed(seed=123456)
    np.random.shuffle(filenames)
    
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
    dataset = dataset.shuffle(shuffle_buffer, seed=12345, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x: tf.parse_example(x, features=sparse_features)['sparse'])

    if trim_dim > 0:
        dataset = dataset.map(lambda x: tf.sparse_slice(x, [0,0], [batch_size, trim_dim]) )
        
    if idf_path:
        idf = np.load(idf_path)[:trim_dim if trim_dim else original_dim]
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32) * idf)
        
    dataset = dataset.take(n_batches).map(sparse_to_dense)
    return dataset

def get_validation_dataset(path, n_pages, original_dim=202498, trim_dim=0,
                           shuffle_buffer=20000, idf_path=None):
    '''
    Note: Keep path input and shuffling settings consistent between runs.
    '''
    
    sparse_features = tfrecord_schema(original_dim)
    filenames = glob.glob(path)
    
    np.random.seed(seed=123456)
    np.random.shuffle(filenames)
    
    dataset = tf.data.TFRecordDataset(filenames).shuffle(shuffle_buffer, seed=30303)
    dataset = dataset.batch(n_pages).take(1)
    dataset = dataset.map(lambda x: tf.parse_example(x, features=sparse_features)['sparse'])
    
    if trim_dim > 0:
        dataset = dataset.map(lambda x: tf.sparse_slice(x, [0,0], [n_pages, trim_dim]) )
        
    if idf_path:
        idf = np.load(idf_path)[:trim_dim if trim_dim else original_dim]
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32) * idf)
    
    return dataset