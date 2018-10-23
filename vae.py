
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.utils import multi_gpu_model

def create_vae(original_dim, intermediate_dim, latent_dim, loss_metric="CrossEntropy", optimizer="RMSProp",
               intermediate_dim2=False, learning_rate=0.001, epsilon_std=1.):

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # Build Encoder
    inputs = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu', name='HiddenLayer1')(inputs)
    if intermediate_dim2:
        h = Dense(intermediate_dim2, activation='relu', name='HiddenLayer2')(h)

    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)
    z = Lambda(sampling, output_shape=(latent_dim,), name='EncodedOutput')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build Decoder
    latent_inputs = Input(shape=(latent_dim,), name='DecoderInput')
    if intermediate_dim2:
        h2_decoded = Dense(intermediate_dim2, activation='relu', name="DecoderHLayer2")(latent_inputs)
        h_decoded = Dense(intermediate_dim, activation='relu', name="DecoderHLayer1")(h2_decoded)
    else:
        h_decoded = Dense(intermediate_dim, activation='relu', name="DecoderHLayer1")(latent_inputs)
    decoder_outputs = Dense(original_dim, activation='sigmoid', name="ReconstructedOutput")(h_decoded)
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
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
            reconstruction_loss = original_dim * binary_crossentropy(inputs, outputs)
        elif loss_metric == "MSE":
            reconstruction_loss = original_dim * mse(inputs, outputs)
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
    
    return vae, decoder