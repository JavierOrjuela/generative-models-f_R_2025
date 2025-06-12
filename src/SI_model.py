"""
Code written by H.J.H for the paper:
Conditional Diffusion-Flow models for generating 3D cosmic
density fields: applications to 𝑓(𝑅) cosmologies
https://arxiv.org/abs/2502.17087

"""


import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from sdeIntegrator import SDEIntegrator
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus
print(tf.config.list_physical_devices('GPU'))

dir_root = ''
num_epochs = 21
total_timesteps = 1000
norm_groups =8  
learning_rate = 1e-4
img_size = 64
depth = 64
height = 64
width = 64
img_channels = 1
clip_min = 0.0
clip_max = 1.0

first_conv_channels = 8
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
num_res_blocks = 6 
activation_fn = tf.keras.activations.swish


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )
class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float64) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float64)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

def ResidualBlock(width, groups=8, activation_fn=tf.keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[4]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv3D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0),kernel_regularizer=regularizers.l2(0.001)
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0),kernel_regularizer=regularizers.l2(0.001))(temb)
        temb = temb[
            :, None, None,None, :
        ]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
       
        x = layers.Conv3D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0),kernel_regularizer=regularizers.l2(0.001)
        )(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv3D(
            width, kernel_size=(3,3,3), padding="same", kernel_initializer=kernel_init(0.0),\
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply

def DownSample(width):
    def apply(x):
        x = layers.Conv3D(
            width,
            kernel_size=3,
            strides=(2,2,2),
            padding="same",
            kernel_initializer=kernel_init(1.0),
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        return x

    return apply


def UpSample(width):
    def apply(x):
        x = layers.UpSampling3D(size=(2,2,2))(x)
        x = layers.Conv3D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0),kernel_regularizer=regularizers.l2(0.001)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=tf.keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0),kernel_regularizer=regularizers.l2(0.001)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

def Targetnn(units, activation_fn=tf.keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0),kernel_regularizer=regularizers.l2(0.001)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

class tile_nn(layers.Layer):
    def __init__(self,units):
        super(tile_nn, self).__init__()
        self.units= units
    def call(self, inputs):
        return tf.tile(inputs,(1,1,1,1,self.units))
    
class MyDenseLabel(layers.Layer):
  def __init__(self,):
    super(MyDenseLabel, self).__init__()
    reconstructed_model_pk = tf.keras.models.load_model(f"model_bis_pdf_pk_tot.keras",safe_mode=False)
    self.model_val_pk=tf.keras.Model(reconstructed_model_pk.input, reconstructed_model_pk.layers[12].output)
    
  def call(self, inputs):
    x= self.model_val_pk(inputs)
    return x


def build_model(
    img_size,
    img_channels,
    widths,
    num_res_blocks=2,
    norm_groups=8,
    activation_fn=tf.keras.activations.swish,
):
    image_input = layers.Input(
        shape=(img_size, img_size, img_size, img_channels), name="image_input"
    )
    label_input = layers.Input(
        shape=(4,), name="label_input"
    )
    time_input = tf.keras.Input(shape=(), dtype=tf.int64, name="time_input")

    x = layers.Conv3D( 
        first_conv_channels,
        kernel_size=(3, 3, 3), 
        padding="same",
        activation=activation_fn,
        kernel_initializer=kernel_init(1.0),
        kernel_regularizer=regularizers.l2(0.001)
    )(image_input)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    label=MyDenseLabel()(label_input)
    skips = [x]
    tile1=tile_nn(x.shape[-1])
    x = tf.keras.layers.Concatenate()([x,tile1(label)])
    
    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            skips.append(x)
        
        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)
            label = tf.keras.layers.MaxPool3D(2)(label)
    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    tile1=tile_nn(x.shape[-1])
    x = tf.keras.layers.Concatenate()([x,tile1(label)])
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    tile2=tile_nn(x.shape[-1])
    x = tf.keras.layers.Concatenate()([x,tile2(label)])
    
    # UpBlock
    
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
    
        if i != 0:
            x = UpSample(widths[i])(x)
            label= layers.UpSampling3D(size=(2,2,2))(label)
    
    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv3D(1, (3, 3, 3), padding="same",kernel_initializer=kernel_init(0.0),kernel_regularizer=regularizers.l2(0.001))(x)
    return tf.keras.Model([image_input, label_input,time_input], x, name="unet")

"""# **Training**"""


class SDEModel(tf.keras.Model):
    def __init__(self, network, timesteps, total_iterations=100):
        super().__init__()
        self.network = network
        self.total_iterations =total_iterations
        self.timesteps = timesteps
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.get_label_box=MyDenseLabel()
        

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]
    
    @tf.function
    def get_x_t(self, x0, x1, eps, t):
        return t**2 * x1 + (1. - t) * x0 + tf.math.sqrt(t) * (1. - t) * eps
    @tf.function
    def compute_x_t_velocity(self,x1,x0_label,t=None):
        if t is None:
            t = tf.random.uniform(shape=x1.shape[0], minval=0., maxval=1.,dtype=tf.float64)
        t2 = t[:, None, None, None, None]
        x0=tf.cast(self.get_label_box(x0_label),tf.float64)
        eps = tf.random.normal(shape=x1.shape,dtype=tf.float64)
        xt = self.get_x_t(x0, x1, eps, t2)
        bt = 2.0 * t2 * x1 - x0 - tf.math.sqrt(t2) * eps
        return xt,bt
    
    @tf.function
    def train_step(self, images_total):
        images,label = images_total
        batch_size = tf.shape(images)[0]

        t = tf.random.uniform(
            minval=0., maxval=1., shape=(batch_size,), dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            xt, bt= self.compute_x_t_velocity(images,label,t)
            vt = self.network([xt, label, t], training=True)
            loss = tf.keras.losses.Huber()(vt, bt)
            losses_weights=sum(self.network.losses)*0.4
            loss += losses_weights
        
        gradients = tape.gradient(loss, self.network.trainable_weights)
        
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(vt, bt)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}
    
    def test_step(self, images_total):
        images, label = images_total
        batch_size = tf.shape(images)[0]

        t = tf.random.uniform(
            minval=0., maxval=1., shape=(batch_size,), dtype=tf.float64)

        xt, bt= self.compute_x_t_velocity(images,label,t)
        vt = self.network([xt, label, t], training=True)
        loss = tf.keras.losses.Huber()(vt, bt)
        
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(vt, bt)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}
    
    def sample(self, x0_label, n_steps=100, n_save=5, method="heun"):
        t_span = np.linspace(0.0, 1.0, n_steps)
        sde = SDEIntegrator(self.model, t_span, n_steps, n_save)
        x0=tf.cast(self.get_label_box(x0_label),tf.float64)
        traj = sde.rollout_forward(x0, method)
        return traj
    
network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=tf.keras.activations.swish,
)

model = SDEModel(
    network=network,
    timesteps=total_timesteps,
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

checkpoint_path = "./results/si_model.weights.h5"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="loss", 
    mode="min",
    save_best_only=True)
history=model.fit(
    Train_dataset,
    epochs=num_epochs,
    validation_data=Validation_dataset,
    callbacks=[checkpoint_callback],
    verbose = 1
)