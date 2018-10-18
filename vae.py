
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import metrics, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model

def create_vae(original_dim, intermediate_dim, latent_dim, intermediate_dim2=False, learning_rate=0.001,
               epsilon_std=1.0):
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)

    # Adding a second hidden layer
    if intermediate_dim2:
        h = Dense(intermediate_dim2, activation='relu')(h)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # For decoding.
    if intermediate_dim2:
        h_decoded2 = Dense(intermediate_dim2, activation='relu')(z)
        h_decoded = Dense(intermediate_dim, activation='relu')(h_decoded2)
    else:
        h_decoded = Dense(intermediate_dim, activation='relu')(z)

    x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

    vae = Model(x, x_decoded_mean)

    # Loss Function, comparing the decoded mean to the original data
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    # Loss Function, comparing the decoded mean to the original data
    def calc_vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    
    # RMSProp Optimizer. see https://keras.io/optimizers/
    optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)

    # Automatically use all available GPUs
    try:
        model = multi_gpu_model(model)
    except:
        pass

    vae_loss = calc_vae_loss(x, x_decoded_mean)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=optimizer)
    #vae.compile(optimizer=optimizer, loss=calc_vae_loss)
    
    return vae