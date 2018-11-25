
def getParser():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--batch-size', '-b', type=int, default=256,
                        help='Size of training batches')
    parser.add_argument('--dims', '-D', type=int, nargs='+', default=[202498, 1000, 200],
                    help='Dimensions of the layers, from input, through hidden dimensions, to output.'
                        'If input is smaller than the vocab_size, training data will be trimmed to top terms.')
    parser.add_argument('--vocab-size', '-v', type=int, default=202498,
                        help='Size of the vocabulary.')
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
    parser.add_argument('--idf_path', '-i', type=str,
            help='Path to a .npy file with a {vocab-size} 1-D array of IDF term weights.')
    parser.add_argument('--max_path', type=str,
            help='Path to a .npy file with a {vocab-size} 1-D array of Max term counts.')
    parser.add_argument('--input-gzip', '-z', action='store_true',
            help='Specify that input TFRecords are gzip compressed.')
    parser.add_argument('--optimizer', '-o', choices=["RMSProp", "Adam"], default="Adam",
            help='Choice of optimizer.')
    parser.add_argument('--distloss', '-k', choices=["ELBO", "MMD", "None"], default="ELBO",
            help="Second loss metric, for addressing the distribution. Usually ELBO (which is KL divergence-based) for Variational Autoencoders.")
    parser.add_argument('--lambda', '-l', type=float, default=1, dest='lambda_',
            help="Lambda, from InfoVAE generalization.")
    parser.add_argument('--alpha', '-a', type=float, default=0,
            help="Alpha, from InfoVAE generalization.")
    parser.add_argument('--restore', action='store_true',
            help="Restore model from checkpoint in /tmp/weights.h5")

    parser.add_argument('training_path', type=str, help="Location of TFRecords for training.")
    parser.add_argument('cross_validation_path', type=str, help="Location of TFRecords for cross-validation.")

    return parser

def param_string(args):
    ''' Return a short string of the run parameters'''
    idf = "-idf" if args.idf_path else ""
    maxp = "-max" if args.max_path else ""
    dims = "_".join([str(d) for d in args.dims])
    params = "L{}-D{}-b{}-N{}-E{}-l{}-a{}-v{}-k{}{}{}".format("%.7f" % args.learning_rate, dims, 
                                                              args.batch_size, args.n_batches,
                                                             args.batches_per_epoch, args.lambda_,
                                                             args.alpha, args.vocab_size,
                                                             args.distloss,
                                                              idf, maxp)
    return params

def parse_param_string(param):
    import re
    parser = getParser()
    arg_string = re.sub('-(\w)', ' -\\1', '-' + param).replace('-D', '-D ').replace('_', ' ')
    arg_string = arg_string.replace('-idf', '--idf_path idf_placeholder')
    arg_list = arg_string.split(' ')[1:]
    args = parser.parse_args(['placeholder', 'placeholder2'] + arg_list)
    return args

def path_to_param(filepath):
    '''
    Extract Autoencoder args from filename
    '''
    import os
    return os.path.split(os.path.splitext(filepath)[0])[-1].replace('-keras-full', '')