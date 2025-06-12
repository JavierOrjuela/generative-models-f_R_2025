import numpy as np
import tensorflow as tf
class SDEIntegrator:
    def __init__(self, drift_model, t_span, n_step, n_save, label):
        self.drift_model = drift_model
        self.t_span = t_span
        self.dt = self.t_span[1] - self.t_span[0]
        self.n_step = n_step
        self.n_save = n_save
        self.label = label

    def step_forward_heun(self, t, x):
        dt_batch=tf.fill(dims=x.shape[0],value= self.dt+t)
  
        dW = tf.math.sqrt(self.dt) * tf.random.normal(shape=x.shape,dtype=tf.float64)
        xhat = x + (1.0 - t) * dW
        K1 = self.drift_model([xhat, self.label, dt_batch], training=False)
        K1=tf.cast(K1,tf.float64)
        xp = xhat + self.dt * K1
        K2 = self.drift_model([xp, self.label, dt_batch], training=False)
        K2=tf.cast(K2,tf.float64)
        return xhat + 0.5 * self.dt * (K1 + K2)

    def step_forward(self, t, x):
        t_batch=tf.fill(dims=x.shape[0],value= t)
        dW = tf.math.sqrt(self.dt) * tf.random.normal(shape=x.shape,dtype=tf.float64)
        return x + tf.cast(self.drift_model([x, self.label, t_batch], training=False),tf.float64) * tf.cast(self.dt,tf.float64) + (1.0 - t) * dW

    def rollout_forward(self, x0, method="heun"):
        x = x0
        for ii, t in enumerate(self.t_span[:-1]):
            if method == "heun":
                x = self.step_forward_heun(t,x)
            else:
                x = self.step_forward(t,x)
        return x