from keras.layers import Input, Dense, Lambda, Layer, Multiply, Add
from keras.models import Model
from keras import optimizers, losses, regularizers
from keras import backend as K
from keras.utils import multi_gpu_model

def create_vae(dims, loss_metric="CrossEntropy", optimizer="Adam",
               learning_rate=0.001, epsilon_std=1.,
               print_summary=False, distloss="ELBO", alpha=0, lambda_=1):

    '''
    Alpha and lambda are terms from the generalize VAE form offered in InfoVAE paper -
    see Eq. 6 in https://arxiv.org/pdf/1706.02262.pdf.
    As noted, when alpha is 0 and lambda is 1, the objective is the typical VAE.
    When lambda is >0 and alpha is 1-lambda, the objective is the BetaVAE form, which simply weighs the KL divergence
    part of the objective more highly. Generally, this will mean that alpha will be negative.
    
    One recommendation for setting lambda is so that the loss on the third term is similar in magnitude to the reconstruction
    loss. One way of roughly doing this is running the training with distloss=None, then finding an alpha=0, lambda=X value
    that roughly doubles it.
    '''
        
    assert alpha <= 1
    assert lambda_ >= 0
    
    vae = (distloss != 'None')

    # The dims arg should be a string of format '1000-200-100-50',
    # going from the input size to the hidden layer sizes and finally the latent
    # dimensions
    assert len(dims) > 2
    original_dim, hidden_dims, latent_dim = dims[0], dims[1:-1], dims[-1]
    
    activation = 'relu'
    activation_out = 'sigmoid'
    kernel_regularizer=regularizers.l2(0.00005)
        
    # Build Encoder
    inputs = Input(shape=(original_dim,))

    for i, hdim in enumerate(hidden_dims):
        layer = Dense(hdim, activation=activation, 
                      kernel_regularizer=kernel_regularizer,
                      name='HiddenLayer%d' %i)
        if i == 0:
            h = layer(inputs)
        else:
            h = layer(h)

    if vae:
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        # To allow for a model that can be compiled, use a layer that adds the ELBO loss function
        # If alpha is 1, this will be ignored.
        if distloss in ["ELBO", "MMD"] and (alpha < 1):
            z_mean, z_log_var = KLDivergenceLayer(alpha=alpha)([z_mean, z_log_var])
        
        # The latent codes, z, are sampled from the mean and standard deviation, through random normal noise
        # Since the model uses log variance, the standard deviation (sigma) is derived in another layer
        z_sigma = Lambda(lambda x: K.exp(x/2.))(z_log_var)
        
        def sampling(inputs):
            z_mean, z_sigma = inputs
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                      stddev=epsilon_std)
            return z_mean + z_sigma * epsilon
        
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_sigma])
        
        if (distloss == "MMD"):
            # Fail when the parameters would make this term ignored
            assert (alpha+lambda_) != 1
            z = MMDLayer(alpha, lambda_)(z)
        
        encoder = Model(inputs, outputs=[z], name='encoder')
        
    else:
        z = Dense(latent_dim, activation='relu', name='z')(h)
        encoder = Model(inputs, outputs=[z], name='encoder')
        
    if print_summary:
        encoder.summary()

    # build Decoder
    latent_inputs = Input(shape=(latent_dim,), name='DecoderInput')
    
    for i, hdim in enumerate(hidden_dims[::-1]):
        j = len(hidden_dims) - i
        layer = Dense(hdim,
                      activation=activation, kernel_regularizer=kernel_regularizer,
                      name="DecoderHLayer%d" % j)
        if i == 0:
            h_decoded = layer(latent_inputs)
        else:
            h_decoded = layer(h_decoded)
    
    decoder_outputs = Dense(original_dim, activation=activation_out, name="ReconstructedOutput")(h_decoded)
    
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    if print_summary:
        decoder.summary()

    # instantiate VAE model
    # Use the reconstructed version ('z', index=0) of the input data as the output
    if vae:
        enc = encoder(inputs)
    else:
        enc = encoder(inputs)
    outputs = decoder(enc)
    vae = Model(inputs, outputs, name='vae')
    

    # Automatically use all available GPUs
    try:
        vae = multi_gpu_model(vae)
    except:
        pass
    
    # RMSProp Optimizer. see https://keras.io/optimizers/
    if optimizer == "RMSProp":
        optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer == "Adam":
        # Adam should be better with the sparsity seen in text
        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, 
                                    decay=0.0, amsgrad=False)
        
    reconstruction_loss = K.sum(K.binary_crossentropy(inputs, outputs), axis=-1)
    # I take the mean reconstruction loss, because the full scalar of losses causes problems
    # for callbacks
    vae.add_loss(K.mean(reconstruction_loss)) 
    vae.compile(optimizer=optimizer)
    #vae.compile(optimizer=optimizer, loss=calc_vae_loss)
    
    return vae

class KLDivergenceLayer(Layer):

    """ A Keras-friendly version of the KL Divergence loss in Variational Auto-Encoders.
    The z_mean and z_log_var are filtered through this layer, and added to the loss provided to the model (i.e. the reconstruction
    loss). A clever trick! 
    From: http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    
    Calling takes three terms: mean, log_var, and alpha.
    Alpha is the term from the InfoVAE paper - when zero it is a typical VAE objective.
    """

    def __init__(self, alpha=0, *args, **kwargs):
        self.is_placeholder = True
        self.alpha = alpha
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        
        kl_batch = - .5 *  (1-self.alpha) * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

class MMDLayer(Layer):
    '''
    An implementation of one of the proposed divergence families in the InfoVAE paper for the 3rd
    term in the generalized objective. This layer applies Maximum-Mean Discrepency.
    Code is adapted from http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html
    
    '''
    def __init__(self, alpha, lambda_, *args, **kwargs):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.is_placeholder = True
        super(MMDLayer, self).__init__(*args, **kwargs)
        
    def compute_kernel(self, x, y):
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))

    def compute_mmd(self, x, y):
        ## Compute MMD (Maximum Mean Discrepancy)
        ## Used in https://arxiv.org/pdf/1706.02262.pdf
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

    def call(self, z):
        latent_dim = z.shape[1]
        # Generate 200 samples from a Gaussian distribution
        gaussian = K.random_normal(K.stack([200, latent_dim]))
        mmd = self.compute_mmd(gaussian, z)

        self.add_loss((1 - self.alpha - self.lambda_) * mmd, inputs=z)
        
        return z

    
def restore_model(filename, trim_head=200):
    '''
    Grab params for a saved model from its filename, instantiate the model and load the weights
    '''
    from vaeArgs import path_to_param, parse_param_string
    
    param = path_to_param(filename)
    args = parse_param_string(param)

    trim_dim = args.dims[0] if (args.dims[0] < args.vocab_size) else None
    if not trim_dim:
        args.dims[0] = args.vocab_size - trim_head

    vae = create_vae(args.dims,
                         distloss=args.distloss,
                         alpha = args.alpha, lambda_ = args.lambda_,
                         learning_rate=args.learning_rate, epsilon_std=1.0)

    vae.load_weights(filename)
    return args, vae