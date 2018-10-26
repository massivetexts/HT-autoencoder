def main(args, vae_model=None, callbacks=[]):
    import tensorflow as tf
    import numpy as np
    from vae import create_vae
    import time
    import json
    import os
    from vaeArgs import param_string
    from data_utils import get_train_dataset, get_validation_dataset

    epochs = np.floor(args.n_batches / args.batches_per_epoch).astype(int)
    params = param_string(args)
    assert epochs > 0
    print("Running with params", params)
    
    trim_head = 200
    trim_dim = args.dims[0] if (args.dims[0] < args.vocab_size) else None
    
    if not trim_dim:
        args.dims[0] = args.vocab_size - trim_head

    if not vae_model:
        # Compile model
        vae = create_vae(args.dims, args.loss,
                         args.optimizer, linear=args.linear,
                         learning_rate=args.learning_rate, epsilon_std=1.0)
    else:
        vae = vae_model

    # Prepare reference to input data
    if args.input_gzip:
        compression = "GZIP"
    else:
        compression = ""

    train_dataset = get_train_dataset(path=args.training_path + "/*.tfrecor*",
                                      batch_size=args.batch_size, n_batches=args.n_batches,
                                      trim_dim=trim_dim, trim_head=trim_head,
                                      idf_path=args.idf_path, compression=compression, max_path=args.max_path)
    val_dataset = get_validation_dataset(path=args.cross_validation_path + "/*.tfrecord*",
                                         n_pages=args.validation_size, trim_head=trim_head,
                                         trim_dim=trim_dim, idf_path=args.idf_path,
                                         compression=compression, max_path=args.max_path)

    # Initialize Variables
    init = tf.global_variables_initializer()
    init.run()

    # Load Cross-Validation data into memory
    val_iter = val_dataset.make_one_shot_iterator()
    val_data = val_iter.get_next()
    val_data_dense = tf.sparse_to_dense(val_data.indices,
                                                val_data.dense_shape,
                                                val_data.values).eval()

    # Create an Iterator for the training data
    train_iter = train_dataset.make_one_shot_iterator()
    traindata = train_iter.get_next()
    # Loss_iter is the exactExact same thing. Why? because the custom loss can't be compiled without a x_true and x_predict
    # for which AEs are just using the original data. 
    # This can be worked around by using vae.add_loss rather than vae.compile(loss=...), which mean that
    # a x isn't needed for fit - however, that also means that saving models is trickier and converting
    # to TPUs fails.
    ##loss_iter = train_dataset.make_one_shot_iterator()
    ##lossdata = loss_iter.get_next()
    

    # Don't specify number of samples, just the width of the samples
    # For some reason, the Model specifies a large number of random samples,
    # But it doesn't seem to affect the quality of models
    traindata.set_shape({args.batch_size,
                         args.dims[0]})
    #lossdata.set_shape({None,
    #                     args.dims[0]})
    start = time.time()

    # Train the autoencoder
    history = vae.fit(traindata, #lossdata,
            shuffle=False,
            epochs=epochs,
            steps_per_epoch=args.batches_per_epoch,
            validation_data=(val_data_dense, None),
            #validation_data=(val_data_dense, val_data_dense),
            validation_steps = 1,
            callbacks=callbacks
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
    
    return (vae, log)

if __name__ == '__main__':
    from vaeArgs import getParser
    parser = getParser()   
    args = parser.parse_args()
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=args.log_device,
                                                       allow_soft_placement=True))
    main(args)
    sess.close()