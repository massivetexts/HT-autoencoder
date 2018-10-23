from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers, losses, regularizers
from keras import backend as K
from keras.utils import multi_gpu_model

def create_vae(dims, loss_metric="CrossEntropy", optimizer="RMSProp",
               learning_rate=0.001, epsilon_std=1., linear=False,
               print_summary=False):

    if linear:
        print("LINEAR AE not tested and likely broken.")
        
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # The dims arg should be a string of format '1000-200-100-50',
    # going from the input size to the hidden layer sizes and finally the latent
    # dimensions
    diml = [int(d) for d in dims.split('-')]
    assert len(diml) > 2
    original_dim, hidden_dims, latent_dim = diml[0], diml[1:-1], diml[-1]
    
    if linear:
        activation = 'linear'
        activation_out = 'linear'
        kernel_regularizer=regularizers.l2(0.00005)
    else:
        activation = 'relu'
        activation_out = 'sigmoid'
        kernel_regularizer=regularizers.l2(0.00005)
        
    # Build Encoder
    inputs = Input(shape=(original_dim,))

    for i, hdim in hidden_dims:
        layer = Dense(hdim, activation=activation, 
                      kernel_regularizer=kernel_regularizer,
                      name='HiddenLayer%d' %i)
        if i == 0:
            h = layer(inputs)
        else:
            h = layer(h)

    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)
    
    z = Lambda(sampling, output_shape=(latent_dim,), name='EncodedOutput')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    if print_summary:
        encoder.summary()

    # build Decoder
    latent_inputs = Input(shape=(latent_dim,), name='DecoderInput')
    
    for i, hdim in hidden_dims[::-1]:
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
    # Use the reconstructed version ('z', index=2) of the input data as the output
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # Automatically use all available GPUs
    try:
        vae = multi_gpu_model(vae)
    except:
        pass
    
    # Loss Function, comparing the decoded mean to the original data
    def calc_vae_loss(inputs, outputs):
        if loss_metric == "CrossEntropy":
            reconstruction_loss = original_dim * losses.binary_crossentropy(inputs, outputs)
        elif loss_metric == "MSE":
            reconstruction_loss = original_dim * losses.mse(inputs, outputs)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)
    
    # RMSProp Optimizer. see https://keras.io/optimizers/
    if optimizer == "RMSProp":
        optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer == "Adam":
        # Adam should be better with the sparsity seen in text
        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, 
                                    decay=0.0, amsgrad=False)
        
    
    vae_loss = calc_vae_loss(inputs, outputs)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=optimizer)
    #vae.compile(optimizer=optimizer, loss=calc_vae_loss)
    
    return vae