
def getParser():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--batch-size', '-b', type=int, default=256,
                        help='Size of training batches')
    parser.add_argument('--vocab-size', '-v', type=int, default=202498,
                        help='Size of the vocabulary.')
    parser.add_argument('--trim-vocab', '-t', type=int, default=0,
                        help='Cut off the original vocabulary. '
                        'Only keeps the top {trim-vocab} words for training.')
    parser.add_argument('--latent-dim', '-l', type=int, default=200,
                        help='Number of output dimensions.')
    parser.add_argument('--hidden-dim', '-H', type=int, default=1500,
                        help='Size of first hidden layer.')
    parser.add_argument('--hidden2-dim', '-G', type=int, default=0,
                        help='Size of second hidden layer. If zero, layer is not instantiated.')
    parser.add_argument('--learning-rate', '-L', type=float, default=0.0001,
                        help='Learning rate for optimizer.')
    parser.add_argument('--n-batches', '-N', type=int, default=5096,
                        help='Number of batches of size `batch-size` to train on.')
    parser.add_argument('--batches-per-epoch', '-E', type=int, default=32,
                        help='Number of batches per epoch. Data does not repeat, so the epochs are just arbitrary cutoffs for logging.')
    parser.add_argument('--model-outdir', type=str, default="models/",
                        help='Directory to save models.')
    parser.add_argument('--no-optimizer-save', action='store_true',
                        help='Flag to turn off optimizer saving on the model. This will halve the size'
                       'of the saved model, but training cannot be resumed.')
    parser.add_argument('--log-dir', type=str, default="logs/",
                        help='Directory to log output details.')
    parser.add_argument('--validation-size', '-V', type=int, default=1024,
                        help='Number of pages to use for cross-validation.')
    parser.add_argument('--log-device', '-d', action='store_true',
                    help='Turn on device logging. Useful for debugging whether computation is on expected'
                        'CPUs or GPUs, otherwise unnecessary.') 
    parser.add_argument('--linear', action='store_true',
                    help='Use linear activations. Closer to PCA.') 
    parser.add_argument('--idf_path', '-i', type=str,
            help='Path to a .npy file with a {vocab-size} 1-D array of IDF term weights.')
    parser.add_argument('--max_path', type=str,
            help='Path to a .npy file with a {vocab-size} 1-D array of Max term counts.')
    parser.add_argument('--input-gzip', '-z', action='store_true',
            help='Specify that input TFRecords are gzip compressed.')
    parser.add_argument('--optimizer', '-o', choices=["RMSProp", "Adam"], default="RMSProp",
            help='Choice of optimizer.')
    parser.add_argument('--loss', '-S', choices=["CrossEntropy", "MSE"], default="CrossEntropy",
            help="Loss function metric. This is combined with KL.")

    parser.add_argument('training_path', type=str, help="Location of TFRecords for training.")
    parser.add_argument('cross_validation_path', type=str, help="Location of TFRecords for cross-validation.")

    return parser

def param_string(args):
    ''' Return a short string of the run parameters'''
    idf = "-idf" if args.idf_path else ""
    maxp = "-max" if args.max_path else ""
    lin = "-lin" if args.linear else ""
    params = "L{}-H{}-G{}-b{}-l{}-N{}-E{}-v{}-t{}-o{}-S{}{}{}".format("%.7f" % args.learning_rate, args.hidden_dim,
                                                             args.hidden2_dim, args.batch_size,
                                                             args.latent_dim, args.n_batches,
                                                             args.batches_per_epoch, args.vocab_size,
                                                             args.trim_vocab, args.optimizer, args.loss, idf, maxp,
                                                                     lin)
    return params