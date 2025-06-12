"""
Code written by H.J.H for the paper:
Conditional Diffusion-Flow models for generating 3D cosmic
density fields: applications to 𝑓(𝑅) cosmologies
https://arxiv.org/abs/2502.17087

"""


import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from volumentations import *
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os

dataset_repetitions = 5
num_epochs = 50  
total_timesteps = 1000
norm_groups = 8 
learning_rate = 1e-4

img_size = 64
depth = 64
height = 64
width = 64
img_channels = 1
clip_min = 0.0
clip_max = 1.0


class GaussianDiffusion:
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=0.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.num_timesteps = int(timesteps)
        
        def cosine_beta_schedule(time_steps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end, s = 0.008,shift=0.93):
            """
            cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            """
            x = tf.linspace(0, time_steps, time_steps +1)
            def f(t):
                return tf.math.cos(shift*0.5 * np.pi*(t / time_steps + s) / (1 + s) ) ** 2
            alphas_cumprod = f(x) / f(tf.constant([0]))
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            beta_schedule=tf.clip_by_value(betas,beta_start, 0.999)
            return beta_schedule
        def sigmoid_beta_schedule(timesteps=self.timesteps):
            betas = np.linspace(-6, 6, timesteps)
            def sigmoid_np(Z):
                return 1/(1+(np.exp((-Z))))
            return sigmoid_np(betas) * (beta_end - beta_start) + beta_start

        self.betas = betas = cosine_beta_schedule()
        #self.betas = betas = sigmoid_beta_schedule()

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float64)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float64)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float64)
         
        self.sqrt_alphas_cumprod_prev = tf.constant(
            np.sqrt(alphas_cumprod_prev), dtype=tf.float64
        )

        self.sqrt_one_minus_alphas_cumprod_prev = tf.constant(
            np.sqrt(1.0 - alphas_cumprod_prev), dtype=tf.float64
        )
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float64
        )

        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float64
        )

        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float64
        )

        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float64
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float64
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float64)

        # Log calculation clipped
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float64
        )

        sigma_1= (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        sigma_2= 1.- (alphas_cumprod/alphas_cumprod_prev)
        posterior_variance_ddim_new = sigma_1*sigma_2
        self.posterior_variance_ddim_new = tf.constant(posterior_variance_ddim_new, dtype=tf.float64)
        self.posterior_log_variance_clipped_ddim_new = tf.constant(
            np.log(np.maximum(posterior_variance_ddim_new, 1e-20)), dtype=tf.float64
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float64,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float64,
        )
        self.x_0_pred_coef1 =tf.constant(1. / np.sqrt(alphas_cumprod),dtype=tf.float64)
        self.x_0_pred_coef2 = tf.constant(np.sqrt(1.0 - alphas_cumprod) / np.sqrt(alphas_cumprod),dtype=tf.float64)
        

    def _extract(self, a, t, x_shape):
        
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise,shift=1.0):
        
        x_start_shape = tf.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise*shift
        )

    def predict_start_from_noise(self, x_t, t, noise,shift=1.0):
        x_t_shape = tf.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise*shift
        )

    def q_posterior(self, x_start, x_t, t):

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_posterior_ddim_new(self, x_start, x_t, t,eta=1.):
        
        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.sqrt_alphas_cumprod_prev, t, x_t_shape) * x_start
            + self._extract(1.- self.alphas_cumprod_prev -self.posterior_variance_ddim_new**2*eta, t, x_t_shape) * x_t
        )
        
        posterior_variance = self._extract(self.posterior_variance_ddim_new*eta, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped_ddim_new*eta, t, x_t_shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_x0_arguments(self, x_t, t):
        x_t_shape = tf.shape(x_t)
        val1= self._extract(self.x_0_pred_coef1, t, x_t_shape) * x_t
        val2=self._extract(self.x_0_pred_coef2, t, x_t_shape)
        return val1, val2
        

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance_ddim_new(self, pred_noise, x, t, clip_denoised=True,eta=1.):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_ddim_new(
            x_start=x_recon, x_t=x, t=t,eta=eta
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):

        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float64), [tf.shape(x)[0], 1, 1, 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
    
    def p_sample_ddim_new(self, pred_noise, x, t, clip_denoised=True,eta=1.):

        model_mean,_, model_log_variance= self.p_mean_variance_ddim_new(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised,eta=1.
        )
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float64), [tf.shape(x)[0], 1, 1, 1, 1])
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise*eta
    


