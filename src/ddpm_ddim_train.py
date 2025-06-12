"""
Code written by H.J.H for the paper:
Conditional Diffusion-Flow models for generating 3D cosmic
density fields: applications to 𝑓(𝑅) cosmologies
https://arxiv.org/abs/2502.17087

"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from data_prep import normalization_features, normalization_data_64
from nn_model import build_model
from ddpm_model import GaussianDiffusion
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus
print(tf.config.list_physical_devices('GPU'))
dir_root = ''


class DiffusionModel(tf.keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, total_iterations=100,ema=0.999):
        super().__init__()
        self.network = network
        self.total_iterations =total_iterations
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")


    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]
    @tf.function
    def train_step(self, images_total):
        KLDiv=False
        images,label = images_total

        batch_size = tf.shape(images)[0]

        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        if tf.random.uniform(minval=0, maxval=1, shape=()) < 0.1:
            cls_ = tf.fill(dims=tf.shape(images),value=0.)
        else:
            cls_ = tf.fill(dims=tf.shape(images),value=1.)

        with tf.GradientTape() as tape:
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            images_t = self.gdf_util.q_sample(images, t, noise)
            values1_coef,values2_coef =self.gdf_util.q_x0_arguments(images_t, t)

            pred_noise = self.network([images_t, label, t,cls_], training=True)
            image_pred= values1_coef - values2_coef*tf.cast(pred_noise,tf.float64)
            loss = tf.keras.losses.Huber()(noise, pred_noise)
            losses_weights=sum(self.network.losses)
            loss += losses_weights
            if self.optimizer.iterations >= int(self.total_iterations*0.5) and KLDiv:
                for indx in range(batch_size):
                    pdf_img= tf.histogram_fixed_width(images[indx],[0,1],nbins=50)
                    pdf_img = tf.cast(pdf_img ,tf.float32)
                    pdf_pred= tf.histogram_fixed_width(image_pred[indx],[0,1],nbins=50)
                    pdf_pred = tf.cast(pdf_pred ,tf.float32)
                    loss_pdf = tf.keras.losses.KLDivergence()(pdf_img,pdf_pred)
                    loss += tf.cast(loss_pdf,tf.float32)*0.01

        gradients = tape.gradient(loss, self.network.trainable_weights)
        
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        if self.optimizer.iterations >= int(self.total_iterations*0.9):
        
            for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
                ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
                
        else:
            for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
                ema_weight.assign(weight)
                
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(noise, pred_noise)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    def test_step(self, images_total):
        images, label = images_total
        
        batch_size = tf.shape(images)[0]

        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)


        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

        images_t = self.gdf_util.q_sample(images, t, noise)
        cls_ = tf.fill(dims=tf.shape(images),value=1.)

        pred_noise = self.network([images_t, label, t,cls_], training=False)

        loss = tf.keras.losses.MSE(noise, pred_noise)
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(noise, pred_noise)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}
    
    def simulation_ddim(self, epoch=None, logs=None,epochs_=10,num_images=5,predict=False,labels=None,labels_var=False,cfg_weight=0.8,inference_timesteps=10,eta=0):
        if not labels_var:
            pd_random=pd.read_csv('sample.csv')
            mylabels = list(normalization_features(pd_random[['Om' ,'h' ,'sigma8' ,'fR0_scaled']].values[0]))
            labels = tf.constant([mylabels]*num_images)
            true_box = normalization_data_64(np.load(pd_random['filename_path'].values[0]))
        
        samples = tf.random.normal(
            shape=(num_images, img_size, img_size, img_size, img_channels), dtype=tf.float64)
        #inference_range = range(0, self.timesteps, self.timesteps // inference_timesteps)
        method = "quadratic"
        if method == "linear":
            a = self.timesteps // inference_timesteps
            time_steps = np.asarray(list(range(0, self.timesteps, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.timesteps * 0.9), inference_timesteps) ** 2)
        
        if epoch or predict:
          if True:
        
            cls_true = tf.fill(dims=tf.shape(samples),value=1.)
            cls_false = tf.fill(dims=tf.shape(samples),value=0.)

            for t in reversed(range(inference_timesteps)):

                tt = tf.cast(tf.fill(num_images, time_steps[t]), dtype=tf.int64)
                pred_noise_true = self.network.predict(
                      [samples,labels,tt,cls_true], verbose=0, batch_size=num_images  )
                pred_noise_false = self.network.predict(
                      [samples,labels,tt,cls_false], verbose=0, batch_size=num_images  )

                pred_noise = (pred_noise_true +
                                   cfg_weight * (-pred_noise_true + pred_noise_false))

                samples = self.gdf_util.p_sample_ddim_new(
                    pred_noise, samples, tt,eta=eta)
               
        if epoch:
          if epoch % epochs_ == 0:
            sim=samples.numpy()
            fig = plt.figure()
            for i in range(1,num_images+1):
              ax = fig.add_subplot(1,num_images,i)
              ax.hist(true_box.flatten(), bins='auto', color='red', density=True);
              sim_test = np.squeeze(sim[i-1])
              ax.hist(sim_test.astype('float32').flatten(), bins='auto', color='b', alpha=0.4, density=True)
            plt.savefig("./results/imagen_labels_{}.png".format(epoch))
            plt.close()
        if predict:
            return samples


    def simulation(self, epoch=None, logs=None,epochs_=10,num_images=5,predict=False,\
                   labels=None,labels_var=False,cfg_weight=0.8):
        Test_LSS=pd.read_csv('sample.csv')
        if not labels_var:
            pd_random=Test_LSS.sample(1)
            mylabels = list(normalization_features(pd_random[['Om' ,'h' ,'sigma8' ,'fR0_scaled']].values[0]))
            labels = tf.constant([mylabels]*num_images)
            true_box = normalization_data_64(np.load(pd_random['filename_path'].values[0]))
        
        samples = tf.random.normal(
            shape=(num_images,64,64,64,1), dtype=tf.float64)
        if epoch or predict:
          if True:
            samples_difussion_list=[]
            cls_true = tf.fill(dims=tf.shape(samples),value=1.)
            cls_false = tf.fill(dims=tf.shape(samples),value=0.)
            values_times=range(0, self.timesteps)
            for t in reversed(values_times):
                tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
                pred_noise_true = self.network.predict(
                      [samples,labels,tt,cls_true], verbose=0, batch_size=num_images  )
                pred_noise_false = self.network.predict(
                      [samples,labels,tt,cls_false], verbose=0, batch_size=num_images  )
                pred_noise = (pred_noise_true +
                                   cfg_weight * (pred_noise_false- pred_noise_true ))
                pred_noise = pred_noise_true
                samples = self.gdf_util.p_sample(
                    pred_noise, samples, tt, clip_denoised=True)
                samples_difussion=True
                if samples_difussion:
                    if (t%50==0 and t<500):
                        samples_difussion_list.append(samples)
                        continue

        if epoch:
          if epoch % epochs_ == 0:
            sim=samples.numpy()
    
            fig = plt.figure()
            for i in range(1,num_images+1):
              ax = fig.add_subplot(1,num_images,i)
          
              ax.hist(true_box.flatten(), bins='auto', color='red', density=True);
              sim_test = np.squeeze(sim[i-1])
        
              ax.hist(sim_test.astype('float32').flatten(), bins='auto', color='b', alpha=0.4, density=True)
         
            plt.savefig("./imagen_labels_{}.png".format(epoch))
            plt.close()
        if predict:
            return samples,samples_difussion_list


first_conv_channels = 8
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False,  False,False]
num_res_blocks = 6 
activation_fn = tf.keras.activations.swish
img_size=64
img_channels=1
dataset_repetitions = 5
num_epochs = 50  
total_timesteps = 1000
norm_groups =8  
learning_rate = 1e-4


network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=activation_fn,
)
ema_network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=activation_fn,
)
ema_network.set_weights(network.get_weights())  
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
)

checkpoint_path = "./results/diffusion_model.weights.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="loss", 
    mode="min",
    save_best_only=True,
)
model.load_weights(checkpoint_path)

history=model.fit(
    Train_dataset,
    epochs=num_epochs,
    validation_data=Validation_dataset,
    callbacks=[ tf.keras.callbacks.LambdaCallback(on_epoch_end=model.simulation), checkpoint_callback],
    verbose = 1
)
history_pd=pd.DataFrame.from_dict(history.history)
history_pd.to_csv(dir_root+'history_ddpm.csv',index=False)
