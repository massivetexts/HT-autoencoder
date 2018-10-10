def main(args):
    import tensorflow as tf
    import numpy as np
    from vae import create_vae
    import time
    import json
    import os
    from data_utils import get_train_dataset, get_validation_dataset

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

    epochs = np.floor(args.n_batches / args.batches_per_epoch).astype(int)
    assert epochs > 0
    assert args.hidden_dim > args.hidden2_dim

    params = "L{}-H{}-G{}-b{}-l{}-N{}-E{}-v{}-V{}".format(args.learning_rate,
                                                      args.hidden_dim,
                                                      args.hidden2_dim,
                                                      args.batch_size,
                                                      args.latent_dim,
                                                      args.n_batches,
                                                      args.batches_per_epoch,
                                                      args.vocab_size,
                                                         args.validation_size)
    
    print("Running with params", params)
    
    # Compile model
    vae = create_vae(args.vocab_size, args.hidden_dim, args.latent_dim,
                     intermediate_dim2=args.hidden2_dim,
                     learning_rate=args.learning_rate, epsilon_std=1.0)

    # Prepare reference to input data
    train_dataset = get_train_dataset(path=args.training_path + "/*.tfrecord", batch_size=args.batch_size, n_batches=args.n_batches)
    val_dataset = get_validation_dataset(path=args.cross_validation_path + "/*.tfrecord", n_pages=args.validation_size)

    # Initialize Variables
    init = tf.global_variables_initializer()
    init.run()

    # Load Cross-Validation data into memory

    val_iter = val_dataset.make_one_shot_iterator()
    val_data = val_iter.get_next()['sparse']
    val_data_dense = tf.sparse_to_dense(val_data.indices,
                                                val_data.dense_shape,
                                                val_data.values).eval()

    # Create an Iterator for the training data
    train_iter = train_dataset.make_one_shot_iterator()
    traindata = train_iter.get_next()
    traindata.set_shape({args.batch_size, args.vocab_size})
    
    start = time.time()

    # Train the autoencoder
    history = vae.fit(traindata,
            shuffle=False,
            epochs=epochs,
            steps_per_epoch=args.batches_per_epoch,
            validation_data=(val_data_dense, None),
            validation_steps = 1
            )

    passed = time.time() - start

    # Save Model
    mfilename = '%s/%s-keras-%s.h5' % (args.model_outdir, 
                                    params, "small" if args.no_optimizer_save else "full")
    vae.save(mfilename, include_optimizer=(not args.no_optimizer_save))

    # Save Run info
    log = dict(history= history.history,
               time= passed,
               date=time.ctime(),
               params=params
    )

    with open(os.path.join(args.log_dir, params + '-log.json'), mode='a') as f:
        json.dump(log, f)
    
    sess.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--batch-size', '-b', type=int, default=256,
                        help='Size of training batches')
    parser.add_argument('--vocab-size', '-v', type=int, default=202498,
                        help='Size of the vocabulary.')
    parser.add_argument('--latent-dim', '-l', type=int, default=200,
                        help='Number of output dimensions.')
    parser.add_argument('--hidden-dim', '-H', type=int, default=1500,
                        help='Size of first hidden layer.')
    parser.add_argument('--hidden2-dim', '-G', type=int, default=0,
                        help='Size of second hidden layer. If zero, layer is not instantiated.')
    parser.add_argument('--learning-rate', '-L', type=float, default=0.001,
                        help='Learning rate for optimizer.')
    parser.add_argument('--n-batches', '-N', type=int, default=2700,
                        help='Number of batches of size `batch-size` to train on.')
    parser.add_argument('--batches-per-epoch', '-E', type=int, default=50,
                        help='Number of batches per epoch. Data does not repeat, so the epochs are just arbitrary cutoffs for logging.')
    parser.add_argument('--model-outdir', type=str, default="models/",
                        help='Directory to save models.')
    parser.add_argument('--no-optimizer-save', action='store_true',
                        help='Flag to turn off optimizer saving on the model. This will halve the size'
                       'of the saved model, but training cannot be resumed.')
    parser.add_argument('--log-dir', type=str, default="logs/",
                        help='Directory to log output details.')
    parser.add_argument('--validation-size', type=int, default=1024,
                        help='Number of pages to use for cross-validation.')    

    parser.add_argument('training_path', type=str, help="Location of TFRecords for training.")
    parser.add_argument('cross_validation_path', type=str, help="Location of TFRecords for cross-validation.")
    
    args = parser.parse_args()
    main(args)